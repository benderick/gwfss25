#!/usr/bin/env python3
"""Audit whether native-image crop magnification can correct model errors."""

import argparse
import heapq
import json
import math
import sys
import time
import zlib
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audit_trpl import (  # noqa: E402
    caption_panel,
    colorize_labels,
    format_metric,
    load_checkpoint_branch,
    one_image_summary,
    read_label,
    sanitized_name,
    write_csv,
)
from config.add_cfg import add_ssl_config  # noqa: E402
from data import DatasetCatalog, MetadataCatalog  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import detection_utils  # noqa: E402
from detectron2.data import transforms as T  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from mask2former import add_maskformer2_config  # noqa: E402
from mask2former.topowheat.audit import (  # noqa: E402
    SelectiveSegmentationAccumulator,
)
from mask2former.topowheat.topology import endpoint_map  # noqa: E402


DEFAULT_BUDGETS = "1,2,4,8"
PRIMARY_FUSION = "mean"
ROUTING_POLICIES = ("entropy", "margin", "stem", "bazr")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use validation labels only for measurement: compare the global "
            "prediction with native-image crops recomputed at larger model scale."
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
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--window-stride", type=int, default=128)
    parser.add_argument("--zoom-size", type=int, default=512)
    parser.add_argument(
        "--dense-short-edge",
        type=int,
        default=768,
        help="Fixed full-image scale control; set to 0 to disable.",
    )
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=2025)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--visualize-best", type=int, default=10)
    parser.add_argument(
        "--expected-baseline-miou",
        type=float,
        default=None,
        help="Expected mIoU as a fraction, for example 0.7310942.",
    )
    parser.add_argument(
        "--baseline-tolerance",
        type=float,
        default=0.001,
        help="Maximum absolute baseline discrepancy as an mIoU fraction.",
    )
    parser.add_argument(
        "--minimum-oracle-gain",
        type=float,
        default=0.005,
        help="Required K=4 oracle gain as an mIoU fraction.",
    )
    parser.add_argument(
        "--minimum-dense-scale-gain",
        type=float,
        default=0.005,
        help="Material full-image scale gain as an mIoU fraction.",
    )
    parser.add_argument(
        "--minimum-recovery",
        type=float,
        default=0.5,
        help="Required K=4 fraction of the positive oracle-all gain.",
    )
    parser.add_argument(
        "--minimum-selector-gain",
        type=float,
        default=0.001,
        help="Required label-free selector gain over random as an mIoU fraction.",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Additional config overrides in KEY VALUE form.",
    )
    return parser.parse_args()


def parse_budgets(raw):
    budgets = []
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        budget = int(value)
        if budget <= 0:
            raise ValueError("budgets must be positive integers")
        budgets.append(budget)
    budgets = sorted(set(budgets))
    if not budgets:
        raise ValueError("at least one budget is required")
    return budgets


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
    cfg.MODEL.TOPOWHEAT.TRPL.ENABLED = False
    cfg.MODEL.TOPOWHEAT.TCPM.ENABLED = False
    cfg.MODEL.TOPOWHEAT.BAZR.ENABLED = False
    cfg.MODEL.TOPOWHEAT.BAZR.AUX_HEADS_ENABLED = False
    cfg.SSL.TRAIN_SSL = False
    cfg.DATASETS.TEST = (args.dataset,)
    cfg.freeze()
    return cfg


def configured_test_size(value):
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("configured test size is empty")
        return int(value[0])
    return int(value)


def prepare_image(record, cfg, min_size, max_size):
    original = detection_utils.read_image(
        record["file_name"],
        format=cfg.INPUT.FORMAT,
    )
    target = read_label(record["sem_seg_file_name"])
    if tuple(original.shape[:2]) != tuple(target.shape[:2]):
        raise ValueError(
            "image and target dimensions differ for {}".format(
                record["file_name"]
            )
        )
    transform = T.ResizeShortestEdge(
        [min_size, min_size],
        max_size,
    ).get_transform(original)
    resized = transform.apply_image(original)
    image_tensor = torch.as_tensor(
        np.ascontiguousarray(resized.transpose(2, 0, 1)),
        dtype=torch.float32,
    )
    target_tensor = torch.as_tensor(
        np.ascontiguousarray(target),
        dtype=torch.long,
    )
    return original, resized, image_tensor, target_tensor


def resized_image_tensor(image, short_edge, max_size):
    transform = T.ResizeShortestEdge(
        [int(short_edge), int(short_edge)],
        int(max_size),
    ).get_transform(image)
    resized = transform.apply_image(image)
    return torch.as_tensor(
        np.ascontiguousarray(resized.transpose(2, 0, 1)),
        dtype=torch.float32,
    )


def probabilities_from_scores(scores, eps=1e-6):
    scores = scores.float()
    if bool(scores.lt(0).any().item()):
        return F.softmax(scores, dim=0)
    scores = scores.clamp_min(0.0)
    return scores / scores.sum(dim=0, keepdim=True).clamp_min(eps)


def run_model(model, image, output_size):
    output_height, output_width = (int(value) for value in output_size)
    result = model(
        [
            {
                "image": image,
                "height": output_height,
                "width": output_width,
            }
        ]
    )[0]
    scores = result["sem_seg"].float()
    if tuple(scores.shape[-2:]) != (output_height, output_width):
        scores = F.interpolate(
            scores.unsqueeze(0),
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return scores


def axis_positions(length, window, stride):
    window = min(int(window), int(length))
    end = int(length) - window
    positions = list(range(0, end + 1, max(int(stride), 1)))
    if not positions or positions[-1] != end:
        positions.append(end)
    return positions, window


def candidate_windows(height, width, window_size, stride):
    y_positions, window_height = axis_positions(height, window_size, stride)
    x_positions, window_width = axis_positions(width, window_size, stride)
    return [
        (x1, y1, x1 + window_width, y1 + window_height)
        for y1 in y_positions
        for x1 in x_positions
    ]


def map_window(window, source_size, destination_size):
    source_height, source_width = source_size
    destination_height, destination_width = destination_size
    x1, y1, x2, y2 = window
    mapped_x1 = int(round(x1 * destination_width / float(source_width)))
    mapped_x2 = int(round(x2 * destination_width / float(source_width)))
    mapped_y1 = int(round(y1 * destination_height / float(source_height)))
    mapped_y2 = int(round(y2 * destination_height / float(source_height)))
    mapped_x1 = min(max(mapped_x1, 0), destination_width - 1)
    mapped_y1 = min(max(mapped_y1, 0), destination_height - 1)
    mapped_x2 = min(max(mapped_x2, mapped_x1 + 1), destination_width)
    mapped_y2 = min(max(mapped_y2, mapped_y1 + 1), destination_height)
    return mapped_x1, mapped_y1, mapped_x2, mapped_y2


def raised_cosine(height, width, device, dtype, floor=0.05):
    if height <= 1:
        window_y = torch.ones(height, device=device, dtype=dtype)
    else:
        window_y = torch.hann_window(
            height,
            periodic=False,
            device=device,
            dtype=dtype,
        )
    if width <= 1:
        window_x = torch.ones(width, device=device, dtype=dtype)
    else:
        window_x = torch.hann_window(
            width,
            periodic=False,
            device=device,
            dtype=dtype,
        )
    return float(floor) + (1.0 - float(floor)) * torch.outer(
        window_y,
        window_x,
    )


def class_balance_weights(target, num_classes, ignore_label):
    valid = (
        target.ne(ignore_label)
        & target.ge(0)
        & target.lt(num_classes)
    )
    counts = torch.bincount(target[valid], minlength=num_classes).float()
    present = counts.gt(0)
    class_weights = torch.zeros_like(counts)
    if present.any():
        class_weights[present] = (
            float(valid.sum().item())
            / (float(present.sum().item()) * counts[present])
        )
    pixel_weights = torch.zeros_like(target, dtype=torch.float32)
    pixel_weights[valid] = class_weights[target[valid]]
    return pixel_weights, valid


def nll_sum(probabilities, target, pixel_weights, valid):
    safe_target = target.clamp(0, probabilities.shape[0] - 1)
    target_probability = probabilities.gather(
        0,
        safe_target.unsqueeze(0),
    ).squeeze(0).clamp_min(1e-6)
    return (
        -target_probability.log() * pixel_weights * valid.float()
    ).sum()


def balanced_nll(probabilities, target, pixel_weights, valid):
    denominator = pixel_weights[valid].sum().clamp_min(1e-6)
    return nll_sum(
        probabilities,
        target,
        pixel_weights,
        valid,
    ) / denominator


def tensor_mean_iou(prediction, target, num_classes, ignore_label):
    valid = (
        target.ne(ignore_label)
        & target.ge(0)
        & target.lt(num_classes)
    )
    if not valid.any():
        return None
    encoded = target[valid] * num_classes + prediction[valid]
    confusion = torch.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    true_positive = confusion.diag().float()
    union = (
        confusion.sum(dim=0).float()
        + confusion.sum(dim=1).float()
        - true_positive
    )
    present = union.gt(0)
    if not present.any():
        return None
    return float((true_positive[present] / union[present]).mean().item())


def tensor_class_iou(
    prediction,
    target,
    class_id,
    num_classes,
    ignore_label,
):
    valid = (
        target.ne(ignore_label)
        & target.ge(0)
        & target.lt(num_classes)
    )
    predicted = prediction.eq(class_id) & valid
    expected = target.eq(class_id) & valid
    union = (predicted | expected).sum().item()
    if not union:
        return None
    return float((predicted & expected).sum().item()) / float(union)


def fused_window(global_window, local_probabilities, weight, mode):
    if mode == "mean":
        return (
            global_window + local_probabilities * weight.unsqueeze(0)
        ) / (1.0 + weight.unsqueeze(0))
    if mode == "replace":
        return local_probabilities
    raise ValueError("unknown fusion mode: {}".format(mode))


def fuse_probabilities(global_probabilities, candidates, selected, mode):
    if not selected:
        return global_probabilities.clone()
    local_sum = torch.zeros_like(global_probabilities)
    local_weight = torch.zeros_like(global_probabilities[:1])
    for candidate_index in selected:
        candidate = candidates[candidate_index]
        x1, y1, x2, y2 = candidate["output_window"]
        weight = candidate["fusion_weight"].unsqueeze(0)
        local_sum[:, y1:y2, x1:x2] += (
            candidate["local_probabilities"] * weight
        )
        local_weight[:, y1:y2, x1:x2] += weight
    if mode == "mean":
        return (global_probabilities + local_sum) / (1.0 + local_weight)
    if mode == "replace":
        local_average = local_sum / local_weight.clamp_min(1e-6)
        return torch.where(
            local_weight.gt(0),
            local_average,
            global_probabilities,
        )
    raise ValueError("unknown fusion mode: {}".format(mode))


def window_iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / float(
        max(first_area + second_area - intersection, 1)
    )


def ordered_selection(candidates, order, budget, nms_threshold):
    selected = []
    for index in order:
        if any(
            window_iou(
                candidates[index]["grid_window"],
                candidates[kept]["grid_window"],
            )
            > nms_threshold
            for kept in selected
        ):
            continue
        selected.append(index)
        if len(selected) >= int(budget):
            break
    return selected


def ranked_selection(candidates, score_name, budget, nms_threshold):
    order = sorted(
        range(len(candidates)),
        key=lambda index: candidates[index][score_name],
        reverse=True,
    )
    return ordered_selection(
        candidates,
        order,
        budget,
        nms_threshold,
    )


def greedy_utility_sequence(
    global_probabilities,
    candidates,
    target,
    pixel_weights,
    valid,
):
    """Greedily add crops that reduce full-image class-balanced NLL."""
    local_sum = torch.zeros_like(global_probabilities)
    local_weight = torch.zeros_like(global_probabilities[:1])
    current = global_probabilities.clone()
    remaining = set(range(len(candidates)))
    selected = []
    gains = []
    denominator = float(pixel_weights[valid].sum().clamp_min(1e-6).item())

    while remaining:
        best_index = None
        best_gain_sum = 0.0
        for index in sorted(remaining):
            candidate = candidates[index]
            x1, y1, x2, y2 = candidate["output_window"]
            region = (slice(y1, y2), slice(x1, x2))
            weight = candidate["fusion_weight"].unsqueeze(0)
            proposed_sum = (
                local_sum[:, region[0], region[1]]
                + candidate["local_probabilities"] * weight
            )
            proposed_weight = (
                local_weight[:, region[0], region[1]] + weight
            )
            proposed = (
                global_probabilities[:, region[0], region[1]]
                + proposed_sum
            ) / (1.0 + proposed_weight)
            current_loss = nll_sum(
                current[:, region[0], region[1]],
                target[region],
                pixel_weights[region],
                valid[region],
            )
            proposed_loss = nll_sum(
                proposed,
                target[region],
                pixel_weights[region],
                valid[region],
            )
            gain_sum = float((current_loss - proposed_loss).item())
            if gain_sum > best_gain_sum:
                best_gain_sum = gain_sum
                best_index = index

        if best_index is None or best_gain_sum <= 0.0:
            break
        candidate = candidates[best_index]
        x1, y1, x2, y2 = candidate["output_window"]
        weight = candidate["fusion_weight"].unsqueeze(0)
        local_sum[:, y1:y2, x1:x2] += (
            candidate["local_probabilities"] * weight
        )
        local_weight[:, y1:y2, x1:x2] += weight
        current[:, y1:y2, x1:x2] = (
            global_probabilities[:, y1:y2, x1:x2]
            + local_sum[:, y1:y2, x1:x2]
        ) / (1.0 + local_weight[:, y1:y2, x1:x2])
        selected.append(best_index)
        gains.append(best_gain_sum / denominator)
        remaining.remove(best_index)
    return selected, gains


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(first, second):
    if len(first) < 2 or len(second) != len(first):
        return None
    first_rank = rankdata(first)
    second_rank = rankdata(second)
    if first_rank.std() <= 0.0 or second_rank.std() <= 0.0:
        return None
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def paired_bootstrap_mean_difference(
    first,
    second,
    repeats,
    seed,
    confidence=0.95,
):
    pairs = [
        (float(left), float(right))
        for left, right in zip(first, second)
        if left is not None and right is not None
    ]
    if not pairs:
        return {
            "images": 0,
            "mean_difference": None,
            "confidence": float(confidence),
            "ci_lower": None,
            "ci_upper": None,
            "positive_interval": False,
        }
    differences = np.asarray(
        [left - right for left, right in pairs],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0,
        len(differences),
        size=(int(repeats), len(differences)),
    )
    bootstrap_means = differences[samples].mean(axis=1)
    tail = 0.5 * (1.0 - float(confidence))
    lower, upper = np.quantile(
        bootstrap_means,
        [tail, 1.0 - tail],
    )
    return {
        "images": len(differences),
        "mean_difference": float(differences.mean()),
        "confidence": float(confidence),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "positive_interval": bool(lower > 0.0),
    }


def new_accumulator(num_classes, class_names, ignore_label):
    return SelectiveSegmentationAccumulator(
        num_classes,
        class_names=class_names,
        ignore_label=ignore_label,
    )


def update_accumulator(accumulator, probabilities, target):
    prediction = probabilities.argmax(dim=0).detach().cpu()
    accumulator.update(prediction, target)
    return prediction


def metric_delta(summary, baseline):
    value = summary["mean_iou"]
    reference = baseline["mean_iou"]
    if value is None or reference is None:
        return None
    return float(value) - float(reference)


def attach_delta(summary, baseline, mean_selected=None):
    result = dict(summary)
    result["mean_iou_gain"] = metric_delta(summary, baseline)
    if mean_selected is not None:
        result["mean_local_forwards"] = float(mean_selected)
        result["mean_total_forwards"] = 1.0 + float(mean_selected)
    return result


def candidate_correlations(rows):
    utility = [row["utility"] for row in rows]
    score_names = (
        "entropy_score",
        "margin_score",
        "stem_score",
        "bazr_score",
        "global_local_disagreement",
    )
    result = OrderedDict()
    for score_name in score_names:
        result[score_name] = spearman(
            [row[score_name] for row in rows],
            utility,
        )
    return result


def display_image(image, input_format):
    if input_format == "BGR":
        return image[:, :, ::-1].copy()
    return image.copy()


def utility_heatmap(shape, candidates):
    utility_sum = np.zeros(shape, dtype=np.float32)
    utility_count = np.zeros(shape, dtype=np.float32)
    for candidate in candidates:
        x1, y1, x2, y2 = candidate["output_window"]
        utility_sum[y1:y2, x1:x2] += float(candidate["utility"])
        utility_count[y1:y2, x1:x2] += 1.0
    values = utility_sum / np.maximum(utility_count, 1.0)
    scale = float(np.percentile(np.abs(values), 95))
    if scale <= 1e-12:
        normalized = np.full(shape, 0.5, dtype=np.float32)
    else:
        normalized = np.clip(0.5 + 0.5 * values / scale, 0.0, 1.0)
    return cv2.applyColorMap(
        np.rint(normalized * 255.0).astype(np.uint8),
        cv2.COLORMAP_JET,
    )[:, :, ::-1]


def save_visualization(path, payload, colors):
    image = display_image(payload["image"], payload["input_format"])
    routed = image.copy()
    for rank, window in enumerate(payload["selected_windows"], start=1):
        x1, y1, x2, y2 = window
        cv2.rectangle(routed, (x1, y1), (x2 - 1, y2 - 1), (255, 65, 35), 2)
        cv2.putText(
            routed,
            str(rank),
            (x1 + 5, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 65, 35),
            2,
            cv2.LINE_AA,
        )
    target = colorize_labels(payload["target"], colors)
    global_prediction = colorize_labels(payload["global_prediction"], colors)
    refined_prediction = colorize_labels(payload["refined_prediction"], colors)
    heatmap = utility_heatmap(payload["target"].shape, payload["candidates"])
    panels = [
        caption_panel(routed, "source image and oracle crops"),
        caption_panel(target, "ground truth"),
        caption_panel(global_prediction, "global"),
        caption_panel(refined_prediction, "oracle K=4"),
        caption_panel(heatmap, "realized utility: blue low / red high"),
    ]
    canvas = np.concatenate(panels, axis=1)
    cv2.imwrite(str(path), canvas[:, :, ::-1])


def main():
    args = parse_args()
    started = time.time()
    budgets = parse_budgets(args.budgets)
    if args.window_size <= 0 or args.window_stride <= 0 or args.zoom_size <= 0:
        raise ValueError("window and zoom dimensions must be positive")
    if args.dense_short_edge < 0:
        raise ValueError("dense short edge must be non-negative")
    if not 0.0 <= args.nms_threshold <= 1.0:
        raise ValueError("NMS threshold must be in [0, 1]")
    if args.random_repeats <= 0:
        raise ValueError("random repeats must be positive")
    if args.bootstrap_repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    if args.max_images < 0 or args.visualize_best < 0:
        raise ValueError("image and visualization limits must be non-negative")
    if args.baseline_tolerance < 0.0:
        raise ValueError("baseline tolerance must be non-negative")
    if args.minimum_oracle_gain < 0.0:
        raise ValueError("minimum oracle gain must be non-negative")
    if not 0.0 <= args.minimum_recovery <= 1.0:
        raise ValueError("minimum recovery must be in [0, 1]")
    if args.minimum_selector_gain < 0.0:
        raise ValueError("minimum selector gain must be non-negative")
    if args.minimum_dense_scale_gain < 0.0:
        raise ValueError("minimum dense scale gain must be non-negative")
    if (
        args.expected_baseline_miou is not None
        and not 0.0 <= args.expected_baseline_miou <= 1.0
    ):
        raise ValueError("expected baseline mIoU must be in [0, 1]")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg(args)
    min_size = args.min_size or configured_test_size(cfg.INPUT.MIN_SIZE_TEST)
    max_size = args.max_size or configured_test_size(cfg.INPUT.MAX_SIZE_TEST)

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

    metadata = MetadataCatalog.get(args.dataset)
    records = sorted(
        DatasetCatalog.get(args.dataset),
        key=lambda record: record["file_name"],
    )
    full_dataset_images = len(records)
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
    if num_classes < 2:
        raise ValueError("the audit requires at least two classes")
    if not 0 <= stem_class < num_classes:
        raise ValueError("stem class must index a configured class")
    stem_name = class_names[stem_class]

    global_raw = new_accumulator(num_classes, class_names, ignore_label)
    global_normalized = new_accumulator(num_classes, class_names, ignore_label)
    dense_scale_accumulator = (
        new_accumulator(num_classes, class_names, ignore_label)
        if args.dense_short_edge > 0
        else None
    )
    scale_mean_accumulator = (
        new_accumulator(num_classes, class_names, ignore_label)
        if args.dense_short_edge > 0
        else None
    )
    scale_confidence_accumulator = (
        new_accumulator(num_classes, class_names, ignore_label)
        if args.dense_short_edge > 0
        else None
    )
    scale_oracle_accumulator = (
        new_accumulator(num_classes, class_names, ignore_label)
        if args.dense_short_edge > 0
        else None
    )
    scale_transition_counts = {
        name: torch.zeros(num_classes, dtype=torch.int64)
        for name in (
            "both_correct",
            "recovered_by_dense",
            "damaged_by_dense",
            "both_wrong",
            "prediction_disagreement",
        )
    }
    dense_accumulators = {
        mode: new_accumulator(num_classes, class_names, ignore_label)
        for mode in ("mean", "replace")
    }
    policy_names = ROUTING_POLICIES + ("oracle_utility",)
    policy_accumulators = OrderedDict(
        (
            (policy, budget),
            new_accumulator(num_classes, class_names, ignore_label),
        )
        for policy in policy_names
        for budget in budgets
    )
    oracle_all_accumulator = new_accumulator(
        num_classes,
        class_names,
        ignore_label,
    )
    random_accumulators = OrderedDict(
        (
            (repeat, budget),
            new_accumulator(num_classes, class_names, ignore_label),
        )
        for repeat in range(args.random_repeats)
        for budget in budgets
    )
    selected_counts = {
        key: 0 for key in policy_accumulators
    }
    random_selected_counts = {
        key: 0 for key in random_accumulators
    }
    dense_counts = {mode: 0 for mode in dense_accumulators}
    oracle_all_count = 0
    oracle_nll_gain_sums = {budget: 0.0 for budget in budgets}
    oracle_all_nll_gain_sum = 0.0

    candidate_rows = []
    per_image_rows = []
    best_heap = []
    source_shape_counts = OrderedDict()
    global_shape_counts = OrderedDict()
    source_global_shape_matches = 0

    with torch.no_grad():
        for image_index, record in enumerate(records):
            original, resized, image_tensor, target = prepare_image(
                record,
                cfg,
                min_size,
                max_size,
            )
            original_height, original_width = original.shape[:2]
            resized_height, resized_width = resized.shape[:2]
            source_shape_key = "{}x{}".format(
                original_width,
                original_height,
            )
            global_shape_key = "{}x{}".format(
                resized_width,
                resized_height,
            )
            source_shape_counts[source_shape_key] = (
                source_shape_counts.get(source_shape_key, 0) + 1
            )
            global_shape_counts[global_shape_key] = (
                global_shape_counts.get(global_shape_key, 0) + 1
            )
            if (original_height, original_width) == (
                resized_height,
                resized_width,
            ):
                source_global_shape_matches += 1
            scores = run_model(
                model,
                image_tensor,
                (original_height, original_width),
            )
            probabilities = probabilities_from_scores(scores)
            target_device = target.to(probabilities.device)
            pixel_weights, valid = class_balance_weights(
                target_device,
                num_classes,
                ignore_label,
            )
            invalid_target = (
                target_device.ne(ignore_label)
                & (target_device.lt(0) | target_device.ge(num_classes))
            )
            if invalid_target.any():
                raise ValueError(
                    "GT contains invalid classes for {}: {}".format(
                        record["sem_seg_file_name"],
                        sorted(
                            target_device[invalid_target].unique().tolist()
                        ),
                    )
                )
            if not valid.any():
                raise ValueError(
                    "GT contains no valid pixels for {}".format(
                        record["sem_seg_file_name"]
                    )
                )

            raw_prediction = scores.argmax(dim=0).detach().cpu()
            normalized_prediction = update_accumulator(
                global_normalized,
                probabilities,
                target,
            )
            global_raw.update(raw_prediction, target)

            dense_scale_prediction = None
            scale_mean_prediction = None
            scale_confidence_prediction = None
            scale_oracle_prediction = None
            if dense_scale_accumulator is not None:
                dense_scale_image = resized_image_tensor(
                    original,
                    args.dense_short_edge,
                    max_size,
                )
                dense_scale_scores = run_model(
                    model,
                    dense_scale_image,
                    (original_height, original_width),
                )
                dense_scale_probabilities = probabilities_from_scores(
                    dense_scale_scores
                )
                dense_scale_prediction_device = (
                    dense_scale_probabilities.argmax(dim=0)
                )
                dense_scale_prediction = (
                    dense_scale_prediction_device.detach().cpu()
                )
                dense_scale_accumulator.update(
                    dense_scale_prediction,
                    target,
                )
                scale_mean_probabilities = 0.5 * (
                    probabilities + dense_scale_probabilities
                )
                scale_mean_prediction = update_accumulator(
                    scale_mean_accumulator,
                    scale_mean_probabilities,
                    target,
                )
                prefer_dense = dense_scale_probabilities.max(dim=0).values.gt(
                    probabilities.max(dim=0).values
                )
                scale_confidence_probabilities = torch.where(
                    prefer_dense.unsqueeze(0),
                    dense_scale_probabilities,
                    probabilities,
                )
                scale_confidence_prediction = update_accumulator(
                    scale_confidence_accumulator,
                    scale_confidence_probabilities,
                    target,
                )

                global_prediction_device = probabilities.argmax(dim=0)
                global_correct = global_prediction_device.eq(target_device)
                dense_correct = dense_scale_prediction_device.eq(target_device)
                scale_oracle_prediction_device = torch.where(
                    dense_correct & ~global_correct,
                    dense_scale_prediction_device,
                    global_prediction_device,
                )
                scale_oracle_prediction = (
                    scale_oracle_prediction_device.detach().cpu()
                )
                scale_oracle_accumulator.update(
                    scale_oracle_prediction,
                    target,
                )
                transitions = {
                    "both_correct": global_correct & dense_correct,
                    "recovered_by_dense": ~global_correct & dense_correct,
                    "damaged_by_dense": global_correct & ~dense_correct,
                    "both_wrong": ~global_correct & ~dense_correct,
                    "prediction_disagreement": global_prediction_device.ne(
                        dense_scale_prediction_device
                    ),
                }
                for class_id in range(num_classes):
                    class_valid = valid & target_device.eq(class_id)
                    for transition_name, transition_mask in transitions.items():
                        scale_transition_counts[transition_name][class_id] += int(
                            (transition_mask & class_valid).sum().item()
                        )

            entropy = -(
                probabilities.clamp_min(1e-6)
                * probabilities.clamp_min(1e-6).log()
            ).sum(dim=0) / math.log(num_classes)
            top_two = probabilities.topk(k=2, dim=0).values
            margin_uncertainty = 1.0 - (top_two[0] - top_two[1])
            stem_probability = probabilities[stem_class]
            predicted_stem = probabilities.argmax(dim=0).eq(stem_class)
            endpoints = endpoint_map(
                predicted_stem.unsqueeze(0).unsqueeze(0),
                skeleton_iterations=20,
                dilation=7,
            ).squeeze(0).squeeze(0)
            pooled_stem = F.avg_pool2d(
                stem_probability.unsqueeze(0).unsqueeze(0),
                kernel_size=5,
                stride=1,
                padding=2,
            ).squeeze(0).squeeze(0)
            legacy_disagreement = (stem_probability - pooled_stem).abs()
            bazr_risk = (
                entropy
                + 2.0 * endpoints.float()
                + legacy_disagreement
                + 0.5 * stem_probability
            )

            original_tensor = torch.as_tensor(
                np.ascontiguousarray(original.transpose(2, 0, 1)),
            )
            candidates = []
            grid_windows = candidate_windows(
                resized_height,
                resized_width,
                args.window_size,
                args.window_stride,
            )
            for candidate_index, grid_window in enumerate(grid_windows):
                output_window = map_window(
                    grid_window,
                    (resized_height, resized_width),
                    (original_height, original_width),
                )
                x1, y1, x2, y2 = output_window
                source_crop = original_tensor[:, y1:y2, x1:x2]
                zoomed_crop = F.interpolate(
                    source_crop.unsqueeze(0).float(),
                    size=(args.zoom_size, args.zoom_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                if not original_tensor.dtype.is_floating_point:
                    zoomed_crop = zoomed_crop.round().clamp(0, 255).to(
                        original_tensor.dtype
                    )
                local_scores = run_model(
                    model,
                    zoomed_crop,
                    (y2 - y1, x2 - x1),
                )
                local_probabilities = probabilities_from_scores(local_scores)
                fusion_weight = raised_cosine(
                    y2 - y1,
                    x2 - x1,
                    probabilities.device,
                    probabilities.dtype,
                )
                global_window = probabilities[:, y1:y2, x1:x2]
                mean_fused = fused_window(
                    global_window,
                    local_probabilities,
                    fusion_weight,
                    PRIMARY_FUSION,
                )
                window_target = target_device[y1:y2, x1:x2]
                window_weights = pixel_weights[y1:y2, x1:x2]
                window_valid = valid[y1:y2, x1:x2]
                global_risk = balanced_nll(
                    global_window,
                    window_target,
                    window_weights,
                    window_valid,
                )
                fused_risk = balanced_nll(
                    mean_fused,
                    window_target,
                    window_weights,
                    window_valid,
                )
                local_risk = balanced_nll(
                    local_probabilities,
                    window_target,
                    window_weights,
                    window_valid,
                )
                global_window_prediction = global_window.argmax(dim=0)
                fused_window_prediction = mean_fused.argmax(dim=0)
                local_window_prediction = local_probabilities.argmax(dim=0)
                global_window_miou = tensor_mean_iou(
                    global_window_prediction,
                    window_target,
                    num_classes,
                    ignore_label,
                )
                fused_window_miou = tensor_mean_iou(
                    fused_window_prediction,
                    window_target,
                    num_classes,
                    ignore_label,
                )
                global_stem_iou = tensor_class_iou(
                    global_window_prediction,
                    window_target,
                    stem_class,
                    num_classes,
                    ignore_label,
                )
                fused_stem_iou = tensor_class_iou(
                    fused_window_prediction,
                    window_target,
                    stem_class,
                    num_classes,
                    ignore_label,
                )
                local_correct = (
                    local_window_prediction.eq(window_target) & window_valid
                )
                global_correct = (
                    global_window_prediction.eq(window_target) & window_valid
                )
                valid_count = int(window_valid.sum().item())
                pixel_accuracy_gain = (
                    float(
                        (
                            local_correct.sum() - global_correct.sum()
                        ).item()
                    )
                    / float(max(valid_count, 1))
                )
                global_local_disagreement = float(
                    (
                        local_probabilities - global_window
                    ).abs().mean().item()
                )
                utility = float((global_risk - fused_risk).item())
                candidate = {
                    "grid_window": grid_window,
                    "output_window": output_window,
                    "local_probabilities": local_probabilities,
                    "fusion_weight": fusion_weight,
                    "entropy_score": float(
                        entropy[y1:y2, x1:x2].mean().item()
                    ),
                    "margin_score": float(
                        margin_uncertainty[y1:y2, x1:x2].mean().item()
                    ),
                    "stem_score": float(
                        stem_probability[y1:y2, x1:x2].mean().item()
                    ),
                    "bazr_score": float(
                        bazr_risk[y1:y2, x1:x2].mean().item()
                    ),
                    "utility": utility,
                }
                candidates.append(candidate)
                candidate_rows.append(
                    {
                        "image_id": record.get("image_id", image_index),
                        "file_name": record["file_name"],
                        "candidate": candidate_index,
                        "grid_x1": grid_window[0],
                        "grid_y1": grid_window[1],
                        "grid_x2": grid_window[2],
                        "grid_y2": grid_window[3],
                        "source_x1": x1,
                        "source_y1": y1,
                        "source_x2": x2,
                        "source_y2": y2,
                        "utility": utility,
                        "local_only_nll_gain": float(
                            (global_risk - local_risk).item()
                        ),
                        "window_miou_gain": (
                            None
                            if global_window_miou is None
                            or fused_window_miou is None
                            else fused_window_miou - global_window_miou
                        ),
                        "stem_iou_gain": (
                            None
                            if global_stem_iou is None
                            or fused_stem_iou is None
                            else fused_stem_iou - global_stem_iou
                        ),
                        "pixel_accuracy_gain": pixel_accuracy_gain,
                        "entropy_score": candidate["entropy_score"],
                        "margin_score": candidate["margin_score"],
                        "stem_score": candidate["stem_score"],
                        "bazr_score": candidate["bazr_score"],
                        "global_local_disagreement": (
                            global_local_disagreement
                        ),
                    }
                )

            dense_predictions = {}
            all_indices = list(range(len(candidates)))
            for mode, accumulator in dense_accumulators.items():
                dense_probabilities = fuse_probabilities(
                    probabilities,
                    candidates,
                    all_indices,
                    mode,
                )
                dense_predictions[mode] = update_accumulator(
                    accumulator,
                    dense_probabilities,
                    target,
                )
                dense_counts[mode] += len(all_indices)

            oracle_sequence, oracle_step_gains = greedy_utility_sequence(
                probabilities,
                candidates,
                target_device,
                pixel_weights,
                valid,
            )
            for budget in budgets:
                oracle_nll_gain_sums[budget] += float(
                    sum(oracle_step_gains[:budget])
                )
            oracle_all_nll_gain_sum += float(sum(oracle_step_gains))
            oracle_all_probabilities = fuse_probabilities(
                probabilities,
                candidates,
                oracle_sequence,
                PRIMARY_FUSION,
            )
            oracle_all_prediction = update_accumulator(
                oracle_all_accumulator,
                oracle_all_probabilities,
                target,
            )
            oracle_all_count += len(oracle_sequence)
            image_selections = {}
            score_fields = {
                "entropy": "entropy_score",
                "margin": "margin_score",
                "stem": "stem_score",
                "bazr": "bazr_score",
            }
            for budget in budgets:
                for policy, score_field in score_fields.items():
                    selection = ranked_selection(
                        candidates,
                        score_field,
                        budget,
                        args.nms_threshold,
                    )
                    image_selections[(policy, budget)] = selection
                image_selections[("oracle_utility", budget)] = (
                    oracle_sequence[:budget]
                )

            policy_predictions = {}
            for key, accumulator in policy_accumulators.items():
                selection = image_selections[key]
                fused = fuse_probabilities(
                    probabilities,
                    candidates,
                    selection,
                    PRIMARY_FUSION,
                )
                policy_predictions[key] = update_accumulator(
                    accumulator,
                    fused,
                    target,
                )
                selected_counts[key] += len(selection)

            image_crc = zlib.crc32(record["file_name"].encode("utf-8"))
            random_image_mious = {budget: [] for budget in budgets}
            for repeat in range(args.random_repeats):
                rng = np.random.default_rng(
                    args.random_seed + image_crc + repeat * 1000003
                )
                random_order = rng.permutation(len(candidates)).tolist()
                for budget in budgets:
                    selection = ordered_selection(
                        candidates,
                        random_order,
                        budget,
                        args.nms_threshold,
                    )
                    fused = fuse_probabilities(
                        probabilities,
                        candidates,
                        selection,
                        PRIMARY_FUSION,
                    )
                    key = (repeat, budget)
                    random_prediction = update_accumulator(
                        random_accumulators[key],
                        fused,
                        target,
                    )
                    random_selected_counts[key] += len(selection)
                    random_image_summary = one_image_summary(
                        random_prediction,
                        target,
                        None,
                        num_classes,
                        class_names,
                        ignore_label,
                    )
                    random_image_mious[budget].append(
                        random_image_summary["mean_iou"]
                    )

            global_image = one_image_summary(
                normalized_prediction,
                target,
                None,
                num_classes,
                class_names,
                ignore_label,
            )
            dense_scale_image_summary = (
                one_image_summary(
                    dense_scale_prediction,
                    target,
                    None,
                    num_classes,
                    class_names,
                    ignore_label,
                )
                if dense_scale_prediction is not None
                else None
            )
            scale_mean_image_summary = (
                one_image_summary(
                    scale_mean_prediction,
                    target,
                    None,
                    num_classes,
                    class_names,
                    ignore_label,
                )
                if scale_mean_prediction is not None
                else None
            )
            scale_confidence_image_summary = (
                one_image_summary(
                    scale_confidence_prediction,
                    target,
                    None,
                    num_classes,
                    class_names,
                    ignore_label,
                )
                if scale_confidence_prediction is not None
                else None
            )
            scale_oracle_image_summary = (
                one_image_summary(
                    scale_oracle_prediction,
                    target,
                    None,
                    num_classes,
                    class_names,
                    ignore_label,
                )
                if scale_oracle_prediction is not None
                else None
            )
            dense_image = one_image_summary(
                dense_predictions[PRIMARY_FUSION],
                target,
                None,
                num_classes,
                class_names,
                ignore_label,
            )
            oracle_budget = 4 if 4 in budgets else budgets[-1]
            oracle_prediction = policy_predictions[
                ("oracle_utility", oracle_budget)
            ]
            oracle_image = one_image_summary(
                oracle_prediction,
                target,
                None,
                num_classes,
                class_names,
                ignore_label,
            )
            positive_candidates = sum(
                candidate["utility"] > 0.0 for candidate in candidates
            )
            row = {
                "image_id": record.get("image_id", image_index),
                "file_name": record["file_name"],
                "source_height": original_height,
                "source_width": original_width,
                "global_height": resized_height,
                "global_width": resized_width,
                "candidate_count": len(candidates),
                "positive_utility_candidates": positive_candidates,
                "positive_utility_fraction": (
                    positive_candidates / float(max(len(candidates), 1))
                ),
                "global_miou": global_image["mean_iou"],
                "dense_scale_miou": (
                    dense_scale_image_summary["mean_iou"]
                    if dense_scale_image_summary is not None
                    else None
                ),
                "dense_scale_miou_gain": (
                    dense_scale_image_summary["mean_iou"]
                    - global_image["mean_iou"]
                    if dense_scale_image_summary is not None
                    else None
                ),
                "scale_mean_miou": (
                    scale_mean_image_summary["mean_iou"]
                    if scale_mean_image_summary is not None
                    else None
                ),
                "scale_mean_miou_gain": (
                    scale_mean_image_summary["mean_iou"]
                    - global_image["mean_iou"]
                    if scale_mean_image_summary is not None
                    else None
                ),
                "scale_confidence_miou": (
                    scale_confidence_image_summary["mean_iou"]
                    if scale_confidence_image_summary is not None
                    else None
                ),
                "scale_confidence_miou_gain": (
                    scale_confidence_image_summary["mean_iou"]
                    - global_image["mean_iou"]
                    if scale_confidence_image_summary is not None
                    else None
                ),
                "scale_oracle_miou": (
                    scale_oracle_image_summary["mean_iou"]
                    if scale_oracle_image_summary is not None
                    else None
                ),
                "scale_oracle_miou_gain": (
                    scale_oracle_image_summary["mean_iou"]
                    - global_image["mean_iou"]
                    if scale_oracle_image_summary is not None
                    else None
                ),
                "dense_mean_miou": dense_image["mean_iou"],
                "dense_mean_gain": (
                    dense_image["mean_iou"] - global_image["mean_iou"]
                ),
                "oracle_budget": oracle_budget,
                "oracle_selected": len(
                    image_selections[("oracle_utility", oracle_budget)]
                ),
                "oracle_miou": oracle_image["mean_iou"],
                "oracle_miou_gain": (
                    oracle_image["mean_iou"] - global_image["mean_iou"]
                ),
                "oracle_positive_steps": len(oracle_sequence),
                "oracle_total_nll_gain": float(sum(oracle_step_gains)),
            }
            oracle_all_image = one_image_summary(
                oracle_all_prediction,
                target,
                None,
                num_classes,
                class_names,
                ignore_label,
            )
            row["oracle_all_selected"] = len(oracle_sequence)
            row["oracle_all_miou"] = oracle_all_image["mean_iou"]
            row["oracle_all_miou_gain"] = (
                oracle_all_image["mean_iou"] - global_image["mean_iou"]
            )
            row["random_mean_miou"] = float(
                np.mean(random_image_mious[oracle_budget])
            )
            row["random_mean_miou_gain"] = (
                row["random_mean_miou"] - global_image["mean_iou"]
            )
            for policy in ROUTING_POLICIES:
                prediction = policy_predictions[(policy, oracle_budget)]
                image_summary = one_image_summary(
                    prediction,
                    target,
                    None,
                    num_classes,
                    class_names,
                    ignore_label,
                )
                row["{}_miou".format(policy)] = image_summary["mean_iou"]
                row["{}_miou_gain".format(policy)] = (
                    image_summary["mean_iou"] - global_image["mean_iou"]
                )
            per_image_rows.append(row)

            if args.visualize_best > 0:
                selected = image_selections[
                    ("oracle_utility", oracle_budget)
                ]
                payload = {
                    "image": original.copy(),
                    "input_format": cfg.INPUT.FORMAT,
                    "target": target.numpy(),
                    "global_prediction": normalized_prediction.numpy(),
                    "refined_prediction": oracle_prediction.numpy(),
                    "selected_windows": [
                        candidates[index]["output_window"] for index in selected
                    ],
                    "candidates": [
                        {
                            "output_window": candidate["output_window"],
                            "utility": candidate["utility"],
                        }
                        for candidate in candidates
                    ],
                    "image_id": row["image_id"],
                }
                heap_item = (row["oracle_miou_gain"], image_index, payload)
                if len(best_heap) < args.visualize_best:
                    heapq.heappush(best_heap, heap_item)
                elif heap_item[:2] > best_heap[0][:2]:
                    heapq.heapreplace(best_heap, heap_item)

            if (image_index + 1) % 10 == 0 or image_index + 1 == len(records):
                print(
                    "Audited {}/{} images ({} local forwards)".format(
                        image_index + 1,
                        len(records),
                        len(candidates),
                    ),
                    flush=True,
                )

    global_raw_summary = global_raw.summary()
    global_summary = global_normalized.summary()
    dense_scale_summary = (
        attach_delta(
            dense_scale_accumulator.summary(),
            global_summary,
        )
        if dense_scale_accumulator is not None
        else None
    )
    if dense_scale_summary is not None:
        dense_scale_summary["mean_total_forwards"] = 1.0
    scale_mean_summary = (
        attach_delta(scale_mean_accumulator.summary(), global_summary)
        if scale_mean_accumulator is not None
        else None
    )
    scale_confidence_summary = (
        attach_delta(scale_confidence_accumulator.summary(), global_summary)
        if scale_confidence_accumulator is not None
        else None
    )
    scale_oracle_summary = (
        attach_delta(scale_oracle_accumulator.summary(), global_summary)
        if scale_oracle_accumulator is not None
        else None
    )
    for scale_summary in (
        scale_mean_summary,
        scale_confidence_summary,
        scale_oracle_summary,
    ):
        if scale_summary is not None:
            scale_summary["mean_total_forwards"] = 2.0

    scale_transition_summary = OrderedDict()
    if dense_scale_summary is not None:
        for class_id, class_name in enumerate(class_names):
            counts = {
                name: int(values[class_id].item())
                for name, values in scale_transition_counts.items()
            }
            valid_pixels = sum(
                counts[name]
                for name in (
                    "both_correct",
                    "recovered_by_dense",
                    "damaged_by_dense",
                    "both_wrong",
                )
            )
            counts["valid_pixels"] = valid_pixels
            counts["net_corrected_pixels"] = (
                counts["recovered_by_dense"] - counts["damaged_by_dense"]
            )
            counts["net_corrected_fraction"] = (
                counts["net_corrected_pixels"] / float(valid_pixels)
                if valid_pixels
                else None
            )
            scale_transition_summary[class_name] = counts
    dense_summaries = OrderedDict(
        (
            mode,
            attach_delta(
                accumulator.summary(),
                global_summary,
                dense_counts[mode] / float(len(records)),
            ),
        )
        for mode, accumulator in dense_accumulators.items()
    )
    policy_summaries = OrderedDict()
    for policy in policy_names:
        policy_summaries[policy] = OrderedDict()
        for budget in budgets:
            key = (policy, budget)
            policy_summaries[policy][str(budget)] = attach_delta(
                policy_accumulators[key].summary(),
                global_summary,
                selected_counts[key] / float(len(records)),
            )
    policy_summaries["oracle_utility"]["all"] = attach_delta(
        oracle_all_accumulator.summary(),
        global_summary,
        oracle_all_count / float(len(records)),
    )

    random_summaries = OrderedDict()
    for budget in budgets:
        repeats = []
        for repeat in range(args.random_repeats):
            key = (repeat, budget)
            repeats.append(
                attach_delta(
                    random_accumulators[key].summary(),
                    global_summary,
                    random_selected_counts[key] / float(len(records)),
                )
            )
        values = [item["mean_iou"] for item in repeats]
        gains = [item["mean_iou_gain"] for item in repeats]
        random_summaries[str(budget)] = {
            "repeats": args.random_repeats,
            "mean_iou": float(np.mean(values)),
            "std_iou": float(np.std(values)),
            "mean_iou_gain": float(np.mean(gains)),
            "std_iou_gain": float(np.std(gains)),
            "minimum_iou": float(np.min(values)),
            "maximum_iou": float(np.max(values)),
        }

    routing_budget = 4 if 4 in budgets else budgets[-1]
    routing_evidence = OrderedDict()
    random_image_gains = [
        row["random_mean_miou_gain"] for row in per_image_rows
    ]
    simultaneous_confidence = 1.0 - 0.05 / float(len(ROUTING_POLICIES))
    for policy_index, policy in enumerate(ROUTING_POLICIES):
        policy_image_gains = [
            row["{}_miou_gain".format(policy)] for row in per_image_rows
        ]
        comparison = paired_bootstrap_mean_difference(
            policy_image_gains,
            random_image_gains,
            args.bootstrap_repeats,
            args.random_seed + 10000019 * (policy_index + 1),
            confidence=simultaneous_confidence,
        )
        comparison["ci_estimand"] = "paired_mean_per_image_miou_gain"
        comparison["budget"] = routing_budget
        comparison["aggregate_miou_gain_vs_random"] = (
            policy_summaries[policy][str(routing_budget)]["mean_iou_gain"]
            - random_summaries[str(routing_budget)]["mean_iou_gain"]
        )
        comparison["minimum_aggregate_miou_gain"] = (
            args.minimum_selector_gain
        )
        comparison["gate_passed"] = bool(
            comparison["positive_interval"]
            and comparison["aggregate_miou_gain_vs_random"]
            >= args.minimum_selector_gain
        )
        routing_evidence[policy] = comparison

    utility_values = [row["utility"] for row in candidate_rows]
    positive_utility_fraction = (
        sum(value > 0.0 for value in utility_values)
        / float(max(len(utility_values), 1))
    )
    oracle_four = policy_summaries["oracle_utility"].get("4")
    oracle_all = policy_summaries["oracle_utility"]["all"]
    oracle_four_gain = (
        oracle_four["mean_iou_gain"] if oracle_four is not None else None
    )
    oracle_all_gain = oracle_all["mean_iou_gain"]
    oracle_four_nll_gain = (
        oracle_nll_gain_sums[4] / float(len(records))
        if 4 in oracle_nll_gain_sums
        else None
    )
    oracle_all_nll_gain = oracle_all_nll_gain_sum / float(len(records))
    recovery = (
        oracle_four_nll_gain / oracle_all_nll_gain
        if oracle_four_nll_gain is not None and oracle_all_nll_gain > 0.0
        else None
    )

    is_full_dataset = len(records) == full_dataset_images
    baseline_difference = (
        None
        if args.expected_baseline_miou is None or not is_full_dataset
        else global_raw_summary["mean_iou"] - args.expected_baseline_miou
    )
    parity_passed = (
        None
        if baseline_difference is None
        else abs(baseline_difference) <= args.baseline_tolerance
    )
    dense_scale_gain = (
        dense_scale_summary["mean_iou_gain"]
        if dense_scale_summary is not None
        else None
    )
    dense_scale_gain_passed = (
        None
        if dense_scale_gain is None or not is_full_dataset
        else dense_scale_gain >= args.minimum_dense_scale_gain
    )
    oracle_gain_passed = (
        oracle_four_gain is not None
        and oracle_four_gain >= args.minimum_oracle_gain
    )
    recovery_passed = (
        recovery is not None and recovery >= args.minimum_recovery
    )
    routing_signal_passed = any(
        result["gate_passed"] for result in routing_evidence.values()
    )
    if not is_full_dataset:
        sparse_verdict = "smoke_only_no_verdict"
    elif parity_passed is False:
        sparse_verdict = "invalid_baseline_mismatch"
    elif not oracle_gain_passed or not recovery_passed:
        sparse_verdict = "sparse_magnification_not_supported"
    elif routing_signal_passed:
        sparse_verdict = "sparse_magnification_mechanism_supported"
    else:
        sparse_verdict = "sparse_headroom_without_observable_routing_signal"

    if dense_scale_summary is None:
        dense_scale_verdict = "dense_scale_control_disabled"
    elif not is_full_dataset:
        dense_scale_verdict = "smoke_only_no_verdict"
    elif parity_passed is False:
        dense_scale_verdict = "invalid_baseline_mismatch"
    elif dense_scale_gain_passed:
        dense_scale_verdict = "dense_scale_materially_supported"
    else:
        dense_scale_verdict = "dense_scale_not_materially_supported"

    if not is_full_dataset:
        next_branch = "complete_full_validation_audit"
    elif parity_passed is False:
        next_branch = "repair_evaluation_parity"
    elif dense_scale_gain_passed and not oracle_gain_passed:
        next_branch = "dense_scale_only_no_sparse_router"
    elif not dense_scale_gain_passed and not oracle_gain_passed:
        next_branch = "abandon_resolution_direction"
    elif oracle_gain_passed and not recovery_passed:
        next_branch = "magnification_gain_is_too_diffuse"
    elif not routing_signal_passed:
        next_branch = "sparse_headroom_without_observable_selector"
    else:
        next_branch = "sparse_mechanism_viable_but_not_novel"

    evidence = {
        "criteria": {
            "baseline_tolerance": args.baseline_tolerance,
            "minimum_dense_scale_miou_gain": args.minimum_dense_scale_gain,
            "minimum_oracle_k4_miou_gain": args.minimum_oracle_gain,
            "minimum_oracle_k4_nll_recovery": args.minimum_recovery,
            "minimum_selector_miou_gain_vs_random": (
                args.minimum_selector_gain
            ),
            "selector_familywise_confidence": 0.95,
        },
        "expected_baseline_miou": args.expected_baseline_miou,
        "measured_baseline_miou": global_raw_summary["mean_iou"],
        "baseline_parity_scope": (
            "full_dataset" if is_full_dataset else "not_evaluated_on_subset"
        ),
        "baseline_difference": baseline_difference,
        "baseline_parity_passed": parity_passed,
        "positive_utility_fraction": positive_utility_fraction,
        "dense_mean_miou_gain": dense_summaries["mean"]["mean_iou_gain"],
        "dense_replace_miou_gain": dense_summaries["replace"][
            "mean_iou_gain"
        ],
        "dense_scale_miou_gain": dense_scale_gain,
        "dense_scale_gain_gate_passed": dense_scale_gain_passed,
        "dense_scale_verdict": dense_scale_verdict,
        "scale_mean_miou_gain": (
            scale_mean_summary["mean_iou_gain"]
            if scale_mean_summary is not None
            else None
        ),
        "scale_confidence_miou_gain": (
            scale_confidence_summary["mean_iou_gain"]
            if scale_confidence_summary is not None
            else None
        ),
        "scale_oracle_miou_gain": (
            scale_oracle_summary["mean_iou_gain"]
            if scale_oracle_summary is not None
            else None
        ),
        "oracle_k4_miou_gain": oracle_four_gain,
        "oracle_k4_mean_balanced_nll_gain": oracle_four_nll_gain,
        "oracle_all_mean_local_forwards": oracle_all[
            "mean_local_forwards"
        ],
        "oracle_all_miou_gain": oracle_all_gain,
        "oracle_all_mean_balanced_nll_gain": oracle_all_nll_gain,
        "oracle_k4_nll_recovery": recovery,
        "oracle_gain_gate_passed": oracle_gain_passed,
        "recovery_gate_passed": recovery_passed,
        "routing_signal_gate_passed": routing_signal_passed,
        "routing_signal_comparisons": routing_evidence,
        "interpretation": (
            "This verdict concerns mechanism viability only and is not a "
            "method-novelty claim."
        ),
        "sparse_magnification_verdict": sparse_verdict,
        "next_branch": next_branch,
        "verdict": sparse_verdict,
    }

    summary = {
        "audit_version": 3,
        "dataset": args.dataset,
        "images": len(records),
        "full_dataset_images": full_dataset_images,
        "config_file": str(Path(args.config_file).resolve()),
        "checkpoint": checkpoint_info,
        "input": {
            "global_short_edge": min_size,
            "global_max_size": max_size,
            "local_crop_source": "native_image_file",
            "source_shape_counts": source_shape_counts,
            "global_shape_counts": global_shape_counts,
            "source_global_shape_matches": source_global_shape_matches,
            "source_global_shape_match_fraction": (
                source_global_shape_matches / float(len(records))
            ),
            "magnification_interpretation": (
                "larger effective model sampling; no claim of recovered "
                "sensor information"
            ),
        },
        "candidate_geometry": {
            "coordinate_space": "global_resized_input",
            "window_size": args.window_size,
            "window_stride": args.window_stride,
            "zoom_size": args.zoom_size,
            "legacy_bazr_nms_threshold": args.nms_threshold,
            "budgets": budgets,
        },
        "dense_scale_control": {
            "short_edge": args.dense_short_edge,
            "dense_only": dense_scale_summary,
            "mean_ensemble": scale_mean_summary,
            "confidence_selection": scale_confidence_summary,
            "pixel_oracle": scale_oracle_summary,
            "transitions_by_target_class": scale_transition_summary,
        },
        "class_names": class_names,
        "stem_class": stem_class,
        "stem_name": stem_name,
        "global_raw": global_raw_summary,
        "global_normalized": global_summary,
        "dense_all_windows": dense_summaries,
        "policies": policy_summaries,
        "random_control": random_summaries,
        "candidate_utility": {
            "candidates": len(candidate_rows),
            "positive_fraction": positive_utility_fraction,
            "mean": float(np.mean(utility_values)),
            "median": float(np.median(utility_values)),
            "correlations": candidate_correlations(candidate_rows),
        },
        "evidence": evidence,
        "elapsed_seconds": time.time() - started,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")
    write_csv(output_dir / "candidates.csv", candidate_rows)
    write_csv(output_dir / "per_image.csv", per_image_rows)

    result_rows = [
        {
            "policy": "global_raw",
            "budget": 0,
            "mean_iou": global_raw_summary["mean_iou"],
            "mean_iou_gain": 0.0,
            "mean_total_forwards": 1.0,
        },
        {
            "policy": "global_normalized",
            "budget": 0,
            "mean_iou": global_summary["mean_iou"],
            "mean_iou_gain": 0.0,
            "mean_total_forwards": 1.0,
        },
    ]
    if dense_scale_summary is not None:
        for policy, scale_result in (
            ("dense_scale_{}".format(args.dense_short_edge), dense_scale_summary),
            ("scale_mean_ensemble", scale_mean_summary),
            ("scale_confidence_selection", scale_confidence_summary),
            ("scale_pixel_oracle", scale_oracle_summary),
        ):
            result_rows.append(
                {
                    "policy": policy,
                    "budget": "full_image",
                    "mean_iou": scale_result["mean_iou"],
                    "mean_iou_gain": scale_result["mean_iou_gain"],
                    "mean_total_forwards": scale_result[
                        "mean_total_forwards"
                    ],
                    "stem_iou": scale_result["per_class"][stem_name][
                        "iou"
                    ],
                }
            )
    for mode, result in dense_summaries.items():
        result_rows.append(
            {
                "policy": "dense_{}".format(mode),
                "budget": "all",
                "mean_iou": result["mean_iou"],
                "mean_iou_gain": result["mean_iou_gain"],
                "mean_total_forwards": result["mean_total_forwards"],
            }
        )
    for policy, budget_results in policy_summaries.items():
        for budget, result in budget_results.items():
            result_rows.append(
                {
                    "policy": policy,
                    "budget": budget,
                    "mean_iou": result["mean_iou"],
                    "mean_iou_gain": result["mean_iou_gain"],
                    "mean_total_forwards": result["mean_total_forwards"],
                    "stem_iou": result["per_class"][stem_name]["iou"],
                }
            )
    for budget, result in random_summaries.items():
        result_rows.append(
            {
                "policy": "random_mean",
                "budget": budget,
                "mean_iou": result["mean_iou"],
                "mean_iou_gain": result["mean_iou_gain"],
                "std_iou": result["std_iou"],
            }
        )
    write_csv(output_dir / "policy_results.csv", result_rows)

    if best_heap:
        visualization_dir = output_dir / "best_oracle"
        visualization_dir.mkdir(parents=True, exist_ok=True)
        colors = list(getattr(metadata, "stuff_colors", []))
        if len(colors) != num_classes:
            colors = [
                (0, 0, 0),
                (50, 255, 132),
                (50, 132, 255),
                (214, 255, 50),
            ][:num_classes]
        ranked = sorted(best_heap, key=lambda item: item[:2], reverse=True)
        for rank, (gain, _, payload) in enumerate(ranked, start=1):
            filename = "{:02d}_{}_gain_{:.4f}.png".format(
                rank,
                sanitized_name(payload["image_id"]),
                gain,
            )
            save_visualization(
                visualization_dir / filename,
                payload,
                colors,
            )

    print("\nResolution intervention audit complete")
    print("  images: {}".format(len(records)))
    print(
        "  global raw / normalized mIoU: {} / {}".format(
            format_metric(global_raw_summary["mean_iou"]),
            format_metric(global_summary["mean_iou"]),
        )
    )
    print(
        "  positive-utility candidate fraction: {}".format(
            format_metric(positive_utility_fraction)
        )
    )
    if dense_scale_summary is not None:
        print(
            "  dense {} short-edge mIoU / gain: {} / {}".format(
                args.dense_short_edge,
                format_metric(dense_scale_summary["mean_iou"]),
                format_metric(dense_scale_summary["mean_iou_gain"]),
            )
        )
        print(
            "  scale mean / confidence / oracle gains: {} / {} / {}".format(
                format_metric(scale_mean_summary["mean_iou_gain"]),
                format_metric(scale_confidence_summary["mean_iou_gain"]),
                format_metric(scale_oracle_summary["mean_iou_gain"]),
            )
        )
        print("  dense scale verdict: {}".format(dense_scale_verdict))
    print(
        "  dense mean / replace gain: {} / {}".format(
            format_metric(dense_summaries["mean"]["mean_iou_gain"]),
            format_metric(dense_summaries["replace"]["mean_iou_gain"]),
        )
    )
    print(
        "  oracle K=4 mIoU gain / NLL recovery: {} / {}".format(
            format_metric(oracle_four_gain),
            format_metric(recovery),
        )
    )
    print(
        "  observable routing signal: {}".format(
            "yes" if routing_signal_passed else "no"
        )
    )
    best_routing_policy, best_routing_result = max(
        routing_evidence.items(),
        key=lambda item: item[1]["aggregate_miou_gain_vs_random"],
    )
    print(
        "  best label-free vs random: {} {} (paired CI [{}, {}])".format(
            best_routing_policy,
            format_metric(
                best_routing_result["aggregate_miou_gain_vs_random"]
            ),
            format_metric(best_routing_result["ci_lower"]),
            format_metric(best_routing_result["ci_upper"]),
        )
    )
    print("  sparse verdict: {}".format(sparse_verdict))
    print("  next branch: {}".format(next_branch))
    print("  artifacts: {}".format(output_dir.resolve()))


if __name__ == "__main__":
    main()
