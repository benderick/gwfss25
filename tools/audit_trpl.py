#!/usr/bin/env python3
"""Audit TRPL pseudo-label reliability on labeled validation data."""

import argparse
import csv
import heapq
import json
import re
import sys
import time
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.add_cfg import add_ssl_config  # noqa: E402
from data import DatasetCatalog, MetadataCatalog  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import detection_utils as detection_utils  # noqa: E402
from detectron2.data import transforms as T  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from mask2former import add_maskformer2_config  # noqa: E402
from mask2former.topowheat.audit import (  # noqa: E402
    CalibrationAccumulator,
    SelectiveSegmentationAccumulator,
    TopologyAlignmentAccumulator,
    binary_dilate,
    matched_topk_mask,
    matched_topk_mask_by_class,
)
from mask2former.topowheat.topology import hard_skeletonize  # noqa: E402


DEFAULT_THRESHOLDS = "0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Treat validation images as unlabeled inputs, then use their GT only "
            "to measure pseudo-label quality."
        )
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="gwfss_sem_seg_val")
    parser.add_argument(
        "--checkpoint-branch",
        choices=("auto", "plain", "teacher", "student"),
        default="auto",
        help="Auto prefers the teacher branch in Stage II ensemble checkpoints.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--reliability-threshold", type=float, default=None)
    parser.add_argument("--view-scale", type=float, default=None)
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--topology-tolerance", type=int, default=1)
    parser.add_argument("--visualize-worst", type=int, default=20)
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Additional config overrides in KEY VALUE form.",
    )
    return parser.parse_args()


def parse_thresholds(raw, configured_threshold):
    if not 0.0 < float(configured_threshold) < 1.0:
        raise ValueError("configured reliability threshold must be in (0, 1)")
    thresholds = []
    for value in raw.split(","):
        value = value.strip()
        if value:
            threshold = float(value)
            if not 0.0 < threshold < 1.0:
                raise ValueError("thresholds must be in (0, 1)")
            thresholds.append(threshold)
    thresholds.append(float(configured_threshold))
    return sorted(set(round(value, 8) for value in thresholds))


def build_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ssl_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.TOPOWHEAT.TRPL.ENABLED = True
    cfg.MODEL.TOPOWHEAT.TCPM.ENABLED = False
    cfg.MODEL.TOPOWHEAT.BAZR.ENABLED = False
    cfg.MODEL.TOPOWHEAT.BAZR.AUX_HEADS_ENABLED = False
    cfg.SSL.TRAIN_SSL = False
    cfg.DATASETS.TEST = (args.dataset,)
    if args.reliability_threshold is not None:
        cfg.MODEL.TOPOWHEAT.TRPL.RELIABILITY_THRESHOLD = (
            args.reliability_threshold
        )
    if args.view_scale is not None:
        cfg.MODEL.TOPOWHEAT.TRPL.VIEW_SCALE = args.view_scale
    cfg.freeze()
    return cfg


def _strip_optional_module_prefix(state):
    if state and all(key.startswith("module.") for key in state):
        return OrderedDict(
            (key[len("module.") :], value) for key, value in state.items()
        )
    return state


def load_checkpoint_branch(model, checkpoint_path, requested_branch):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    if not hasattr(state, "items"):
        raise ValueError("checkpoint does not contain a model state dictionary")
    state = _strip_optional_module_prefix(OrderedDict(state.items()))

    prefixes = {
        "teacher": "modelTeacher.",
        "student": "modelStudent.",
    }
    available = {
        branch: any(key.startswith(prefix) for key in state)
        for branch, prefix in prefixes.items()
    }
    if requested_branch == "auto":
        if available["teacher"]:
            branch = "teacher"
        elif available["student"]:
            branch = "student"
        else:
            branch = "plain"
    else:
        branch = requested_branch

    if branch in prefixes:
        if not available[branch]:
            raise ValueError(
                "checkpoint has no {} branch; available branches: {}".format(
                    branch,
                    [name for name, present in available.items() if present],
                )
            )
        prefix = prefixes[branch]
        selected_state = OrderedDict(
            (key[len(prefix) :], value)
            for key, value in state.items()
            if key.startswith(prefix)
        )
    else:
        if any(available.values()):
            raise ValueError(
                "plain branch was requested for an ensemble checkpoint"
            )
        selected_state = OrderedDict(state.items())

    selected_state = _strip_optional_module_prefix(selected_state)
    model_state = model.state_dict()
    compatible = OrderedDict()
    unexpected = []
    shape_mismatch = []
    for key, value in selected_state.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatch.append(
                {
                    "key": key,
                    "checkpoint": list(value.shape),
                    "model": list(model_state[key].shape),
                }
            )
            continue
        compatible[key] = value

    incompatible = model.load_state_dict(compatible, strict=False)
    model_numel = sum(value.numel() for value in model_state.values())
    loaded_numel = sum(value.numel() for value in compatible.values())
    loaded_fraction = loaded_numel / float(max(model_numel, 1))
    if loaded_fraction < 0.90:
        raise RuntimeError(
            "only {:.2%} of model state matched the checkpoint; verify the "
            "config and checkpoint architecture".format(loaded_fraction)
        )

    return {
        "path": str(Path(checkpoint_path).resolve()),
        "requested_branch": requested_branch,
        "loaded_branch": branch,
        "iteration": checkpoint.get("iteration"),
        "loaded_state_fraction": loaded_fraction,
        "loaded_keys": len(compatible),
        "model_keys": len(model_state),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": unexpected + list(incompatible.unexpected_keys),
        "shape_mismatch": shape_mismatch,
    }


def read_label(path):
    label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if label is None:
        raise OSError("failed to read semantic label: {}".format(path))
    if label.ndim == 3:
        label = label[:, :, 0]
    return label


def prepare_image(
    record,
    cfg,
    min_size,
    max_size,
    alignment,
    pixel_mean,
):
    image = detection_utils.read_image(
        record["file_name"],
        format=cfg.INPUT.FORMAT,
    )
    transform = T.ResizeShortestEdge(
        [min_size, min_size],
        max_size,
    ).get_transform(image)
    resized_image = transform.apply_image(image)
    image_tensor = torch.as_tensor(
        np.ascontiguousarray(resized_image.transpose(2, 0, 1)),
        dtype=torch.float32,
    )
    content_height, content_width = image_tensor.shape[-2:]
    padded_height = (
        (content_height + alignment - 1) // alignment * alignment
    )
    padded_width = (
        (content_width + alignment - 1) // alignment * alignment
    )
    if (padded_height, padded_width) != (content_height, content_width):
        padded = pixel_mean.expand(
            image_tensor.shape[0],
            padded_height,
            padded_width,
        ).clone()
        padded[:, :content_height, :content_width] = image_tensor
        image_tensor = padded
    return (
        resized_image,
        image_tensor,
        transform,
        image.shape[:2],
        (content_height, content_width),
    )


def prepare_target(record, transform, image_shape):
    target = read_label(record["sem_seg_file_name"])
    if tuple(image_shape) != tuple(target.shape[:2]):
        raise ValueError(
            "image and target dimensions differ for {}".format(
                record["file_name"]
            )
        )
    resized_target = transform.apply_segmentation(target)
    target_tensor = torch.as_tensor(
        np.ascontiguousarray(resized_target),
        dtype=torch.long,
    )
    return target_tensor


def new_segmentation_accumulator(num_classes, class_names, ignore_label):
    return SelectiveSegmentationAccumulator(
        num_classes,
        class_names=class_names,
        ignore_label=ignore_label,
    )


def one_image_summary(
    prediction,
    target,
    selected,
    num_classes,
    class_names,
    ignore_label,
):
    accumulator = new_segmentation_accumulator(
        num_classes,
        class_names,
        ignore_label,
    )
    accumulator.update(prediction, target, selected)
    return accumulator.summary()


def component_statistics(mask, target_stem, tolerance):
    mask_np = np.ascontiguousarray(
        mask.detach().to(device="cpu", dtype=torch.uint8).numpy()
    )
    target_support = binary_dilate(target_stem, tolerance)
    support_np = target_support.detach().to(device="cpu").numpy().astype(bool)
    component_count, component_labels = cv2.connectedComponents(
        mask_np,
        connectivity=8,
    )
    false_components = 0
    false_component_pixels = 0
    for component_id in range(1, component_count):
        component = component_labels == component_id
        if not np.any(component & support_np):
            false_components += 1
            false_component_pixels += int(component.sum())
    return {
        "components": max(component_count - 1, 0),
        "false_components": false_components,
        "pixels": int(mask_np.sum()),
        "false_component_pixels": false_component_pixels,
    }


def add_component_statistics(totals, update):
    for key, value in update.items():
        totals[key] += int(value)


def finalize_component_statistics(totals):
    components = totals["components"]
    pixels = totals["pixels"]
    return {
        **totals,
        "false_component_fraction": (
            totals["false_components"] / float(components)
            if components
            else None
        ),
        "false_component_pixel_fraction": (
            totals["false_component_pixels"] / float(pixels)
            if pixels
            else None
        ),
    }


def safe_delta(first, second):
    if first is None or second is None:
        return None
    return float(first) - float(second)


def class_balance_pixel_multipliers(summary):
    counts = {
        name: int(metrics["selected_prediction_count"])
        for name, metrics in summary["per_class"].items()
    }
    selected_total = sum(counts.values())
    present_classes = sum(count > 0 for count in counts.values())
    return OrderedDict(
        (
            name,
            (
                selected_total / float(present_classes * count)
                if present_classes and count
                else None
            ),
        )
        for name, count in counts.items()
    )


def flatten_selector_rows(selector_summaries, class_names):
    rows = []
    for selector, summary in selector_summaries.items():
        multipliers = class_balance_pixel_multipliers(summary)
        for class_name in class_names:
            row = {
                "selector": selector,
                "coverage": summary["coverage"],
                "accepted_accuracy": summary["accepted_accuracy"],
                "mean_iou": summary["mean_iou"],
                "class_name": class_name,
                "class_balance_pixel_multiplier": multipliers[class_name],
            }
            row.update(summary["per_class"][class_name])
            rows.append(row)
    return rows


def threshold_rows(sweep_accumulators, class_names, stem_name):
    rows = []
    for (score_name, threshold), accumulator in sweep_accumulators.items():
        summary = accumulator.summary()
        row = {
            "score": score_name,
            "threshold": threshold,
            "coverage": summary["coverage"],
            "accepted_accuracy": summary["accepted_accuracy"],
            "mean_iou": summary["mean_iou"],
        }
        for class_name in class_names:
            class_summary = summary["per_class"][class_name]
            for metric in ("precision", "recall", "iou"):
                row["{}_{}".format(class_name, metric)] = class_summary[metric]
        row["stem_precision"] = summary["per_class"][stem_name]["precision"]
        row["stem_recall"] = summary["per_class"][stem_name]["recall"]
        row["stem_iou"] = summary["per_class"][stem_name]["iou"]
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def colorize_labels(labels, colors, selected=None):
    labels = np.asarray(labels)
    result = np.full((*labels.shape, 3), 40, dtype=np.uint8)
    for class_id, color in enumerate(colors):
        class_mask = labels == class_id
        if selected is not None:
            class_mask &= selected
        result[class_mask] = np.asarray(color, dtype=np.uint8)
    return result


def caption_panel(image, caption):
    header = np.full((28, image.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        header,
        caption,
        (7, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([header, image], axis=0)


def save_visualization(path, payload, colors, stem_class):
    image = payload["image"].copy()
    if payload["input_format"] == "BGR":
        image = image[:, :, ::-1]
    target = payload["target"]
    prediction = payload["prediction"]
    selected = payload["selected"]
    reliability = payload["reliability"]

    target_color = colorize_labels(target, colors)
    prediction_color = colorize_labels(prediction, colors)
    selected_color = colorize_labels(prediction, colors, selected=selected)

    target_stem = target == stem_class
    predicted_stem = (prediction == stem_class) & selected
    valid = target != 255
    error = (image.astype(np.float32) * 0.28).astype(np.uint8)
    error[predicted_stem & target_stem & valid] = (40, 190, 70)
    error[predicted_stem & ~target_stem & valid] = (225, 45, 45)
    error[~predicted_stem & target_stem & valid] = (45, 110, 230)

    heatmap = cv2.applyColorMap(
        np.clip(reliability * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )[:, :, ::-1]
    panels = [
        caption_panel(image, "input"),
        caption_panel(target_color, "ground truth"),
        caption_panel(prediction_color, "teacher mean"),
        caption_panel(selected_color, "TRPL accepted"),
        caption_panel(error, "stem: TP green / FP red / FN blue"),
        caption_panel(heatmap, "TRPL reliability"),
    ]
    canvas = np.concatenate(panels, axis=1)
    cv2.imwrite(str(path), canvas[:, :, ::-1])


def sanitized_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def format_metric(value):
    return "n/a" if value is None else "{:.6f}".format(value)


def aligned_view_multiple(size_divisibility, view_scale):
    size_divisibility = max(int(size_divisibility), 1)
    scale = Fraction(str(float(view_scale))).limit_denominator(64)
    return size_divisibility * scale.denominator


def main():
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args)
    configured_threshold = float(
        cfg.MODEL.TOPOWHEAT.TRPL.RELIABILITY_THRESHOLD
    )
    thresholds = parse_thresholds(args.thresholds, configured_threshold)
    min_size = args.min_size or int(cfg.INPUT.MIN_SIZE_TEST)
    max_size = args.max_size or int(cfg.INPUT.MAX_SIZE_TEST)
    if min_size <= 0 or max_size <= 0:
        raise ValueError("input resize dimensions must be positive")

    model = build_model(cfg)
    checkpoint_info = load_checkpoint_branch(
        model,
        args.checkpoint,
        args.checkpoint_branch,
    )
    print(
        "Loaded {} checkpoint branch ({:.2%} of model state): {}".format(
            checkpoint_info["loaded_branch"],
            checkpoint_info["loaded_state_fraction"],
            checkpoint_info["path"],
        ),
        flush=True,
    )
    model.eval()
    view_alignment = aligned_view_multiple(
        model.size_divisibility,
        cfg.MODEL.TOPOWHEAT.TRPL.VIEW_SCALE,
    )
    pixel_mean = torch.as_tensor(
        cfg.MODEL.PIXEL_MEAN,
        dtype=torch.float32,
    ).view(-1, 1, 1)

    metadata = MetadataCatalog.get(args.dataset)
    records = sorted(
        DatasetCatalog.get(args.dataset),
        key=lambda record: record["file_name"],
    )
    if args.max_images > 0:
        records = records[: args.max_images]
    if not records:
        raise ValueError("dataset contains no records: {}".format(args.dataset))

    num_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
    class_names = list(getattr(metadata, "stuff_classes", []))
    if len(class_names) != num_classes:
        class_names = ["class_{}".format(index) for index in range(num_classes)]
    ignore_label = int(getattr(metadata, "ignore_label", 255))
    stem_class = int(cfg.MODEL.TOPOWHEAT.STEM_CLASS)
    stem_name = class_names[stem_class]

    selector_names = (
        "all_teacher_pixels",
        "confidence_same_threshold",
        "trpl_reliability",
        "confidence_class_matched_coverage",
    )
    selectors = OrderedDict(
        (
            name,
            new_segmentation_accumulator(
                num_classes,
                class_names,
                ignore_label,
            ),
        )
        for name in selector_names
    )
    sweep = OrderedDict()
    for score_name in ("confidence", "reliability"):
        for threshold in thresholds:
            sweep[(score_name, threshold)] = new_segmentation_accumulator(
                num_classes,
                class_names,
                ignore_label,
            )

    calibration = {
        "confidence": CalibrationAccumulator(args.calibration_bins),
        "reliability": CalibrationAccumulator(args.calibration_bins),
    }
    topology = {
        "teacher_stem_skeleton": TopologyAlignmentAccumulator(
            args.topology_tolerance
        ),
        "confidence_matched_teacher_skeleton": TopologyAlignmentAccumulator(
            args.topology_tolerance
        ),
        "stable_multiview_skeleton": TopologyAlignmentAccumulator(
            args.topology_tolerance
        ),
    }
    component_totals = {
        name: {
            "components": 0,
            "false_components": 0,
            "pixels": 0,
            "false_component_pixels": 0,
        }
        for name in (
            "all_teacher_stem",
            "trpl_reliable_stem",
            "confidence_class_matched_stem",
        )
    }
    disagreement_totals = {
        "correct_sum": 0.0,
        "correct_count": 0,
        "incorrect_sum": 0.0,
        "incorrect_count": 0,
    }
    per_image_rows = []
    worst_heap = []

    # Some supported PyTorch versions reject CPU reductions or accumulator
    # updates on tensors created by inference_mode. no_grad keeps normal tensor
    # semantics while still disabling autograd for the audit forward passes.
    with torch.no_grad():
        for index, record in enumerate(records):
            (
                resized_image,
                image_tensor,
                transform,
                image_shape,
                content_size,
            ) = prepare_image(
                record,
                cfg,
                min_size,
                max_size,
                view_alignment,
                pixel_mean,
            )
            generated = model.generate_ssl_targets(
                [{"image": image_tensor}]
            )["trpl"]
            target = prepare_target(record, transform, image_shape)
            content_height, content_width = content_size
            prediction = generated["labels"][
                0, :content_height, :content_width
            ].detach().cpu()
            confidence = generated["confidence"][
                0, :content_height, :content_width
            ].detach().cpu()
            reliability = generated["reliability"][
                0, :content_height, :content_width
            ].detach().cpu()
            disagreement = generated["view_disagreement"][
                0, :content_height, :content_width
            ].detach().cpu()
            stable_skeleton = generated["stable_skeleton"][
                0, :content_height, :content_width
            ].detach().cpu()
            target = target.cpu()

            if prediction.shape != target.shape:
                raise RuntimeError(
                    "TRPL target and GT shapes differ for {}: {} vs {}".format(
                        record["file_name"],
                        tuple(prediction.shape),
                        tuple(target.shape),
                    )
                )
            invalid_target = (
                target.ne(ignore_label)
                & (target.lt(0) | target.ge(num_classes))
            )
            if invalid_target.any():
                raise ValueError(
                    "GT contains class ids outside [0, {}) for {}: {}".format(
                        num_classes,
                        record["sem_seg_file_name"],
                        sorted(target[invalid_target].unique().tolist()),
                    )
                )
            valid = target.ne(ignore_label)
            if not valid.any():
                raise ValueError(
                    "GT contains no valid pixels for {}".format(
                        record["sem_seg_file_name"]
                    )
                )
            correct = prediction.eq(target) & valid
            trpl_selected = reliability.ge(configured_threshold) & valid
            confidence_selected = confidence.ge(configured_threshold) & valid
            confidence_matched = matched_topk_mask_by_class(
                confidence,
                prediction,
                trpl_selected,
                num_classes,
                valid,
            )

            selection_masks = {
                "all_teacher_pixels": valid,
                "confidence_same_threshold": confidence_selected,
                "trpl_reliability": trpl_selected,
                "confidence_class_matched_coverage": confidence_matched,
            }
            for name, selected in selection_masks.items():
                selectors[name].update(prediction, target, selected)
            for threshold in thresholds:
                sweep[("confidence", threshold)].update(
                    prediction,
                    target,
                    confidence.ge(threshold) & valid,
                )
                sweep[("reliability", threshold)].update(
                    prediction,
                    target,
                    reliability.ge(threshold) & valid,
                )

            calibration["confidence"].update(confidence, correct, valid)
            calibration["reliability"].update(reliability, correct, valid)
            disagreement_totals["correct_sum"] += float(
                disagreement[correct].sum().item()
            )
            disagreement_totals["correct_count"] += int(correct.sum().item())
            incorrect = valid & ~correct
            disagreement_totals["incorrect_sum"] += float(
                disagreement[incorrect].sum().item()
            )
            disagreement_totals["incorrect_count"] += int(incorrect.sum().item())

            target_stem = target.eq(stem_class) & valid
            predicted_stem = prediction.eq(stem_class) & valid
            target_skeleton = hard_skeletonize(
                target_stem.to(model.device)[None, None],
            ).squeeze(0).squeeze(0).cpu()
            teacher_skeleton = hard_skeletonize(
                predicted_stem.to(model.device)[None, None],
            ).squeeze(0).squeeze(0).cpu()
            stable_skeleton = stable_skeleton.bool() & valid
            confidence_matched_skeleton = matched_topk_mask(
                confidence,
                stable_skeleton,
                teacher_skeleton,
            )
            topology["teacher_stem_skeleton"].update(
                teacher_skeleton,
                target_stem,
                target_skeleton,
            )
            topology["confidence_matched_teacher_skeleton"].update(
                confidence_matched_skeleton,
                target_stem,
                target_skeleton,
            )
            topology["stable_multiview_skeleton"].update(
                stable_skeleton,
                target_stem,
                target_skeleton,
            )

            component_masks = {
                "all_teacher_stem": predicted_stem,
                "trpl_reliable_stem": predicted_stem & trpl_selected,
                "confidence_class_matched_stem": (
                    predicted_stem & confidence_matched
                ),
            }
            image_component_stats = {}
            for name, mask in component_masks.items():
                stats = component_statistics(
                    mask,
                    target_stem,
                    args.topology_tolerance,
                )
                image_component_stats[name] = stats
                add_component_statistics(component_totals[name], stats)

            image_summaries = {
                name: one_image_summary(
                    prediction,
                    target,
                    selected,
                    num_classes,
                    class_names,
                    ignore_label,
                )
                for name, selected in selection_masks.items()
            }
            trpl_image = image_summaries["trpl_reliability"]
            matched_image = image_summaries[
                "confidence_class_matched_coverage"
            ]
            trpl_stem = trpl_image["per_class"][stem_name]
            matched_stem = matched_image["per_class"][stem_name]
            row = {
                "image_id": record.get("image_id", index),
                "file_name": record["file_name"],
                "valid_pixels": int(valid.sum().item()),
                "mean_confidence": float(confidence[valid].mean().item()),
                "mean_reliability": float(reliability[valid].mean().item()),
                "mean_view_disagreement": float(
                    disagreement[valid].mean().item()
                ),
                "trpl_coverage": trpl_image["coverage"],
                "trpl_accuracy": trpl_image["accepted_accuracy"],
                "trpl_stem_precision": trpl_stem["precision"],
                "trpl_stem_recall": trpl_stem["recall"],
                "trpl_stem_iou": trpl_stem["iou"],
                "matched_confidence_accuracy": matched_image[
                    "accepted_accuracy"
                ],
                "matched_confidence_stem_precision": matched_stem[
                    "precision"
                ],
                "matched_confidence_stem_recall": matched_stem["recall"],
                "stem_precision_gain": safe_delta(
                    trpl_stem["precision"],
                    matched_stem["precision"],
                ),
                "teacher_stem_pixels": int(predicted_stem.sum().item()),
                "target_stem_pixels": int(target_stem.sum().item()),
                "stable_skeleton_pixels": int(stable_skeleton.sum().item()),
                "trpl_false_stem_pixels": int(
                    ((predicted_stem & trpl_selected) & ~target_stem).sum().item()
                ),
                "trpl_false_stem_components": image_component_stats[
                    "trpl_reliable_stem"
                ]["false_components"],
            }
            per_image_rows.append(row)

            if args.visualize_worst > 0:
                false_stem_fraction = row["trpl_false_stem_pixels"] / float(
                    max(row["target_stem_pixels"], 1)
                )
                payload = {
                    "image": resized_image.copy(),
                    "input_format": cfg.INPUT.FORMAT,
                    "target": target.numpy(),
                    "prediction": prediction.numpy(),
                    "selected": trpl_selected.numpy(),
                    "reliability": reliability.numpy(),
                    "image_id": row["image_id"],
                }
                heap_item = (false_stem_fraction, index, payload)
                if len(worst_heap) < args.visualize_worst:
                    heapq.heappush(worst_heap, heap_item)
                elif heap_item[:2] > worst_heap[0][:2]:
                    heapq.heapreplace(worst_heap, heap_item)

            if (index + 1) % 10 == 0 or index + 1 == len(records):
                print(
                    "Audited {}/{} images".format(index + 1, len(records)),
                    flush=True,
                )

    selector_summaries = OrderedDict(
        (name, accumulator.summary())
        for name, accumulator in selectors.items()
    )
    sweep_rows = threshold_rows(sweep, class_names, stem_name)
    topology_summaries = {
        name: accumulator.summary()
        for name, accumulator in topology.items()
    }
    calibration_summaries = {
        name: accumulator.summary()
        for name, accumulator in calibration.items()
    }
    component_summaries = {
        name: finalize_component_statistics(totals)
        for name, totals in component_totals.items()
    }

    trpl_summary = selector_summaries["trpl_reliability"]
    matched_summary = selector_summaries[
        "confidence_class_matched_coverage"
    ]
    trpl_stem = trpl_summary["per_class"][stem_name]
    matched_stem = matched_summary["per_class"][stem_name]
    teacher_topology = topology_summaries["teacher_stem_skeleton"]
    matched_topology = topology_summaries[
        "confidence_matched_teacher_skeleton"
    ]
    stable_topology = topology_summaries["stable_multiview_skeleton"]
    evidence = {
        "criteria": {
            "minimum_precision_gain": 0.03,
            "minimum_stable_skeleton_sensitivity": 0.10,
            "minimum_cldice_gain": 0.0,
        },
        "overall_accuracy_gain_vs_class_matched_confidence": safe_delta(
            trpl_summary["accepted_accuracy"],
            matched_summary["accepted_accuracy"],
        ),
        "stem_precision_gain_vs_class_matched_confidence": safe_delta(
            trpl_stem["precision"],
            matched_stem["precision"],
        ),
        "stem_iou_gain_vs_class_matched_confidence": safe_delta(
            trpl_stem["iou"],
            matched_stem["iou"],
        ),
        "semantic_loss_class_balance_pixel_multipliers": (
            class_balance_pixel_multipliers(trpl_summary)
        ),
        "stable_skeleton_precision_gain_vs_teacher_skeleton": safe_delta(
            stable_topology["precision"],
            teacher_topology["precision"],
        ),
        "stable_skeleton_precision_gain_vs_matched_confidence": safe_delta(
            stable_topology["precision"],
            matched_topology["precision"],
        ),
        "stable_skeleton_cldice_gain_vs_matched_confidence": safe_delta(
            stable_topology["cldice"],
            matched_topology["cldice"],
        ),
    }
    reliability_gain = evidence[
        "stem_precision_gain_vs_class_matched_confidence"
    ]
    skeleton_gain = evidence[
        "stable_skeleton_precision_gain_vs_matched_confidence"
    ]
    skeleton_cldice_gain = evidence[
        "stable_skeleton_cldice_gain_vs_matched_confidence"
    ]
    reliability_gate = (
        reliability_gain is not None and reliability_gain >= 0.03
    )
    topology_gate = (
        skeleton_gain is not None
        and skeleton_gain >= 0.03
        and skeleton_cldice_gain is not None
        and skeleton_cldice_gain >= 0.0
        and stable_topology["sensitivity"] is not None
        and stable_topology["sensitivity"] >= 0.10
    )
    evidence["reliability_gate_passed"] = reliability_gate
    evidence["topology_gate_passed"] = topology_gate
    if (
        reliability_gain is None
        or skeleton_gain is None
        or skeleton_cldice_gain is None
    ):
        evidence["verdict"] = "inconclusive_insufficient_stem_evidence"
    elif reliability_gate and topology_gate:
        evidence["verdict"] = "mechanistic_signal_present"
    else:
        evidence["verdict"] = "current_trpl_not_supported"

    disagreement_summary = {
        "mean_when_correct": (
            disagreement_totals["correct_sum"]
            / disagreement_totals["correct_count"]
            if disagreement_totals["correct_count"]
            else None
        ),
        "mean_when_incorrect": (
            disagreement_totals["incorrect_sum"]
            / disagreement_totals["incorrect_count"]
            if disagreement_totals["incorrect_count"]
            else None
        ),
        **disagreement_totals,
    }
    summary = {
        "audit_version": 2,
        "dataset": args.dataset,
        "images": len(records),
        "config_file": str(Path(args.config_file).resolve()),
        "checkpoint": checkpoint_info,
        "input_resize": {
            "short_edge": min_size,
            "max_size": max_size,
            "view_aligned_padding_multiple": view_alignment,
        },
        "trpl": {
            "view_scale": float(cfg.MODEL.TOPOWHEAT.TRPL.VIEW_SCALE),
            "reliability_threshold": configured_threshold,
            "stem_class": stem_class,
            "stem_name": stem_name,
        },
        "class_names": class_names,
        "selectors": selector_summaries,
        "calibration": calibration_summaries,
        "view_disagreement": disagreement_summary,
        "topology": topology_summaries,
        "stem_components": component_summaries,
        "threshold_sweep": sweep_rows,
        "evidence": evidence,
        "elapsed_seconds": time.time() - started,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")
    write_csv(output_dir / "threshold_sweep.csv", sweep_rows)
    write_csv(
        output_dir / "per_class.csv",
        flatten_selector_rows(selector_summaries, class_names),
    )
    write_csv(output_dir / "per_image.csv", per_image_rows)

    if worst_heap:
        visualization_dir = output_dir / "worst_stem"
        visualization_dir.mkdir(parents=True, exist_ok=True)
        colors = list(getattr(metadata, "stuff_colors", []))
        if len(colors) != num_classes:
            colors = [
                (0, 0, 0),
                (50, 255, 132),
                (50, 132, 255),
                (214, 255, 50),
            ][:num_classes]
        ranked = sorted(worst_heap, key=lambda item: item[:2], reverse=True)
        for rank, (score, _, payload) in enumerate(ranked, start=1):
            filename = "{:02d}_{}_fp_{:.4f}.png".format(
                rank,
                sanitized_name(payload["image_id"]),
                score,
            )
            save_visualization(
                visualization_dir / filename,
                payload,
                colors,
                stem_class,
            )

    full_summary = selector_summaries["all_teacher_pixels"]
    print("\nTRPL audit complete")
    print("  images: {}".format(len(records)))
    print(
        "  teacher mean-view mIoU: {}".format(
            format_metric(full_summary["mean_iou"])
        )
    )
    print(
        "  TRPL coverage / accuracy: {} / {}".format(
            format_metric(trpl_summary["coverage"]),
            format_metric(trpl_summary["accepted_accuracy"]),
        )
    )
    print(
        "  stem precision, TRPL / matched confidence: {} / {}".format(
            format_metric(trpl_stem["precision"]),
            format_metric(matched_stem["precision"]),
        )
    )
    print(
        "  stem precision gain: {}".format(
            format_metric(reliability_gain)
        )
    )
    print(
        "  stem class-balance pixel multiplier: {}".format(
            format_metric(
                evidence["semantic_loss_class_balance_pixel_multipliers"][
                    stem_name
                ]
            )
        )
    )
    print(
        "  stable-skeleton precision gain vs matched confidence: {}".format(
            format_metric(skeleton_gain)
        )
    )
    print("  verdict: {}".format(evidence["verdict"]))
    print("  artifacts: {}".format(output_dir.resolve()))


if __name__ == "__main__":
    main()
