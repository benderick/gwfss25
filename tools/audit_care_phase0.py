#!/usr/bin/env python3
"""Phase-0 falsification audit for CARE-Wheat configuration matching.

The audit never trains the model. A frozen Stage-I model extracts scale-free
soft organ moments from labeled anchors, validation images, and unlabeled donor
images. Validation masks are used only after retrieval to test whether teacher
signatures recover genuinely similar organ configurations.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, OrderedDict
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
from detectron2.data import detection_utils  # noqa: E402
from detectron2.data import transforms as T  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from mask2former import add_maskformer2_config  # noqa: E402


AUDIT_VERSION = 1
DESCRIPTOR_VERSION = "soft_raw_spatial_moments_p2_v1"
MOMENT_NAMES = ("mass", "x", "y", "xx", "xy", "yy")
STYLE_NAMES = (
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
)

# These are decision criteria, not search parameters.
MIN_GT_RETRIEVAL_REDUCTION = 0.30
MIN_DONOR_DISTANCE_REDUCTION = 0.30
MIN_BROAD_SUPPORT_FRACTION = 0.80


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether frozen-teacher organ configurations can safely "
            "match labeled wheat images to cross-domain unlabeled donors."
        )
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchor-dataset", default="gwfss_sem_seg_train")
    parser.add_argument("--validation-dataset", default="gwfss_sem_seg_val")
    parser.add_argument(
        "--donor-dataset",
        default="gwfss_unlabel_random4500_seed2025",
    )
    parser.add_argument(
        "--checkpoint-branch",
        choices=("auto", "plain", "teacher", "student"),
        default="auto",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--max-validation", type=int, default=0)
    parser.add_argument("--max-donors", type=int, default=0)
    parser.add_argument("--visualize", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--expected-validation-miou", type=float, default=None)
    parser.add_argument("--parity-tolerance", type=float, default=0.001)
    parser.add_argument(
        "--recompute-cache",
        action="store_true",
        help="Discard only this audit's descriptor cache before running.",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Additional config overrides in KEY VALUE form.",
    )
    return parser.parse_args()


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
    cfg.SSL.TRAIN_SSL = False
    cfg.MODEL.TOPOWHEAT.TRPL.ENABLED = False
    cfg.MODEL.TOPOWHEAT.TCPM.ENABLED = False
    cfg.MODEL.TOPOWHEAT.BAZR.ENABLED = False
    cfg.MODEL.TOPOWHEAT.BAZR.AUX_HEADS_ENABLED = False
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


def prepare_image(record, cfg, min_size, max_size):
    image = detection_utils.read_image(
        record["file_name"],
        format=cfg.INPUT.FORMAT,
    )
    transform = T.ResizeShortestEdge(
        [min_size, min_size],
        max_size,
    ).get_transform(image)
    resized = transform.apply_image(image)
    tensor = torch.as_tensor(
        np.ascontiguousarray(resized.transpose(2, 0, 1)),
        dtype=torch.float32,
    )
    return image, resized, tensor, transform


def teacher_probabilities(model, image_tensor):
    height, width = image_tensor.shape[-2:]
    result = model(
        [{"image": image_tensor, "height": height, "width": width}]
    )[0]["sem_seg"]
    probabilities = result.float().clamp_min(0.0)
    probabilities = probabilities / probabilities.sum(
        dim=0,
        keepdim=True,
    ).clamp_min(1.0e-6)
    return probabilities.detach().cpu().numpy().astype(np.float32)


def spatial_moment_signature(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape CxHxW")
    _, height, width = probabilities.shape
    x = (np.arange(width, dtype=np.float64) + 0.5) / float(width)
    y = (np.arange(height, dtype=np.float64) + 0.5) / float(height)
    xx, yy = np.meshgrid(x, y)
    bases = (np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy)
    features = []
    for class_probability in probabilities:
        for basis in bases:
            features.append(float(np.mean(class_probability * basis)))
    return np.asarray(features, dtype=np.float64)


def target_probabilities(target, num_classes, ignore_label):
    target = np.asarray(target)
    invalid = (
        (target != ignore_label)
        & ((target < 0) | (target >= num_classes))
    )
    if np.any(invalid):
        raise ValueError(
            "target contains invalid class ids: {}".format(
                sorted(np.unique(target[invalid]).tolist())
            )
        )
    probabilities = np.zeros((num_classes, *target.shape), dtype=np.float32)
    for class_id in range(num_classes):
        probabilities[class_id] = target == class_id
    return probabilities


def canonical_rgb(image, input_format):
    if input_format == "RGB":
        return image
    if input_format == "BGR":
        return image[:, :, ::-1]
    raise ValueError("unsupported INPUT.FORMAT: {}".format(input_format))


def style_signature(resized_image, input_format):
    rgb = canonical_rgb(resized_image, input_format).astype(np.float64) / 255.0
    pixels = rgb.reshape(-1, 3)
    return np.concatenate([pixels.mean(axis=0), pixels.std(axis=0)])


def decoded_image_hash(image, input_format):
    rgb = np.ascontiguousarray(canonical_rgb(image, input_format))
    payload = (
        "{}x{}x{}|".format(*rgb.shape).encode("ascii") + rgb.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def confusion_for_image(prediction, target, num_classes, ignore_label):
    valid = target != ignore_label
    encoded = num_classes * target[valid].astype(np.int64) + prediction[
        valid
    ].astype(np.int64)
    return np.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def summarize_confusion(confusion, class_names):
    confusion = np.asarray(confusion, dtype=np.float64)
    true_positive = np.diag(confusion)
    target_count = confusion.sum(axis=1)
    prediction_count = confusion.sum(axis=0)
    union = target_count + prediction_count - true_positive
    iou = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan),
        where=union > 0,
    )
    accuracy = np.divide(
        true_positive.sum(),
        confusion.sum(),
        out=np.asarray(np.nan),
        where=confusion.sum() > 0,
    )
    return {
        "mean_iou": float(np.nanmean(iou)),
        "pixel_accuracy": float(accuracy),
        "per_class_iou": {
            name: (None if np.isnan(value) else float(value))
            for name, value in zip(class_names, iou)
        },
        "confusion": confusion.astype(np.int64).tolist(),
    }


def descriptor_feature_names(class_names):
    return [
        "{}_{}".format(class_name, moment)
        for class_name in class_names
        for moment in MOMENT_NAMES
    ]


def records_fingerprint(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(Path(record["file_name"]).resolve()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.get("domain_id", "")).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def checkpoint_identity(path):
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(str(temporary), str(path))


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


def load_jsonl(path):
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = {}
    valid_lines = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if line_number == len(lines):
                print(
                    "Ignoring an incomplete final cache line in {}".format(path),
                    flush=True,
                )
                break
            raise
        key = str(Path(row["file_name"]).resolve())
        if key in rows:
            raise ValueError("duplicate cache entry for {}".format(key))
        rows[key] = row
        valid_lines.append(line)
    if len(valid_lines) != len([line for line in lines if line.strip()]):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
        os.replace(str(temporary), str(path))
    return rows


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        handle.flush()


def extract_descriptors(
    role,
    records,
    cache_path,
    model,
    cfg,
    min_size,
    max_size,
    num_classes,
    ignore_label,
    progress_every,
):
    cached = load_jsonl(cache_path)
    expected = {str(Path(record["file_name"]).resolve()) for record in records}
    unknown = sorted(set(cached) - expected)
    if unknown:
        raise ValueError(
            "{} cache contains records outside this run: {}".format(
                role,
                unknown[0],
            )
        )

    started = time.time()
    completed = len(cached)
    newly_processed = 0
    with torch.no_grad():
        for index, record in enumerate(records):
            key = str(Path(record["file_name"]).resolve())
            if key in cached:
                continue

            original, resized, image_tensor, transform = prepare_image(
                record,
                cfg,
                min_size,
                max_size,
            )
            probabilities = teacher_probabilities(model, image_tensor)
            prediction = probabilities.argmax(axis=0).astype(np.int64)
            confidence = probabilities.max(axis=0)
            entropy = -np.sum(
                probabilities * np.log(np.maximum(probabilities, 1.0e-8)),
                axis=0,
            ) / math.log(num_classes)

            row = {
                "role": role,
                "file_name": key,
                "image_id": record.get("image_id", index),
                "domain_id": int(record["domain_id"]),
                "domain_name": str(record["domain_name"]),
                "original_height": int(original.shape[0]),
                "original_width": int(original.shape[1]),
                "resized_height": int(resized.shape[0]),
                "resized_width": int(resized.shape[1]),
                "image_sha256": decoded_image_hash(original, cfg.INPUT.FORMAT),
                "teacher_signature": spatial_moment_signature(
                    probabilities
                ).tolist(),
                "style_signature": style_signature(
                    resized,
                    cfg.INPUT.FORMAT,
                ).tolist(),
                "mean_confidence": float(confidence.mean()),
                "mean_normalized_entropy": float(entropy.mean()),
            }

            if "sem_seg_file_name" in record:
                target = read_label(record["sem_seg_file_name"])
                if target.shape[:2] != original.shape[:2]:
                    raise ValueError(
                        "image and target dimensions differ for {}".format(key)
                    )
                target = transform.apply_segmentation(target).astype(np.int64)
                target_prob = target_probabilities(
                    target,
                    num_classes,
                    ignore_label,
                )
                row["sem_seg_file_name"] = str(
                    Path(record["sem_seg_file_name"]).resolve()
                )
                row["gt_signature"] = spatial_moment_signature(
                    target_prob
                ).tolist()
                row["confusion"] = confusion_for_image(
                    prediction,
                    target,
                    num_classes,
                    ignore_label,
                ).reshape(-1).tolist()

            append_jsonl(cache_path, row)
            cached[key] = row
            completed += 1
            newly_processed += 1

            if (
                completed % max(progress_every, 1) == 0
                or completed == len(records)
            ):
                elapsed = time.time() - started
                rate = elapsed / float(max(newly_processed, 1))
                remaining = len(records) - completed
                print(
                    "{} descriptors: {}/{} (ETA {:.1f} min)".format(
                        role,
                        completed,
                        len(records),
                        remaining * rate / 60.0,
                    ),
                    flush=True,
                )

    return [
        cached[str(Path(record["file_name"]).resolve())]
        for record in records
    ]


def robust_standardizer(values):
    values = np.asarray(values, dtype=np.float64)
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349
    standard_deviation = values.std(axis=0)
    scale = np.where(scale > 1.0e-8, scale, standard_deviation)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    return center, scale


def standardize(values, center, scale):
    return (np.asarray(values, dtype=np.float64) - center) / scale


def rms_distances(query, candidates):
    candidates = np.asarray(candidates, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    return np.sqrt(np.mean(np.square(candidates - query), axis=-1))


def aggregate_reduction(matched, baseline):
    matched = np.asarray(matched, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    denominator = float(baseline.mean())
    if denominator <= 0.0:
        return None
    return 1.0 - float(matched.mean()) / denominator


def bootstrap_reduction(matched, baseline, samples, seed):
    matched = np.asarray(matched, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if len(matched) != len(baseline) or not len(matched):
        raise ValueError("bootstrap arrays must be non-empty and equally sized")
    rng = np.random.RandomState(seed)
    reductions = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.randint(0, len(matched), size=(count, len(matched)))
        matched_mean = matched[indices].mean(axis=1)
        baseline_mean = baseline[indices].mean(axis=1)
        reductions[start : start + count] = 1.0 - matched_mean / np.maximum(
            baseline_mean,
            1.0e-12,
        )
    lower, upper = np.quantile(reductions, (0.025, 0.975))
    return {
        "samples": int(samples),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(first, second):
    first_rank = average_ranks(first)
    second_rank = average_ranks(second)
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    denominator = math.sqrt(
        float(np.square(first_rank).sum() * np.square(second_rank).sum())
    )
    if denominator <= 0.0:
        return None
    return float(np.dot(first_rank, second_rank) / denominator)


def validate_signature_retrieval(rows, teacher_signatures, gt_signatures, args):
    domains = np.asarray([row["domain_id"] for row in rows], dtype=np.int64)
    retrieval_rows = []
    pair_teacher_distances = []
    pair_gt_distances = []

    for first in range(len(rows)):
        for second in range(first + 1, len(rows)):
            if domains[first] == domains[second]:
                continue
            pair_teacher_distances.append(
                rms_distances(
                    teacher_signatures[first],
                    teacher_signatures[second : second + 1],
                )[0]
            )
            pair_gt_distances.append(
                rms_distances(
                    gt_signatures[first],
                    gt_signatures[second : second + 1],
                )[0]
            )

    for index, row in enumerate(rows):
        candidates = np.flatnonzero(domains != domains[index])
        teacher_distance = rms_distances(
            teacher_signatures[index],
            teacher_signatures[candidates],
        )
        gt_distance = rms_distances(
            gt_signatures[index],
            gt_signatures[candidates],
        )
        selected_position = int(np.argmin(teacher_distance))
        oracle_position = int(np.argmin(gt_distance))
        selected_index = int(candidates[selected_position])
        oracle_index = int(candidates[oracle_position])
        retrieval_rows.append(
            {
                "anchor_file": row["file_name"],
                "anchor_domain": row["domain_name"],
                "selected_file": rows[selected_index]["file_name"],
                "selected_domain": rows[selected_index]["domain_name"],
                "oracle_file": rows[oracle_index]["file_name"],
                "teacher_selected_distance": float(
                    teacher_distance[selected_position]
                ),
                "selected_gt_distance": float(gt_distance[selected_position]),
                "random_expected_gt_distance": float(gt_distance.mean()),
                "oracle_gt_distance": float(gt_distance[oracle_position]),
            }
        )

    selected_gt = [row["selected_gt_distance"] for row in retrieval_rows]
    random_gt = [row["random_expected_gt_distance"] for row in retrieval_rows]
    teacher_nn = [row["teacher_selected_distance"] for row in retrieval_rows]
    reduction = aggregate_reduction(selected_gt, random_gt)
    interval = bootstrap_reduction(
        selected_gt,
        random_gt,
        args.bootstrap_samples,
        args.seed,
    )
    gate_passed = (
        reduction is not None
        and reduction >= MIN_GT_RETRIEVAL_REDUCTION
        and interval["lower_95"] > 0.0
    )
    return retrieval_rows, {
        "cross_domain_pairs": len(pair_teacher_distances),
        "teacher_gt_pairwise_spearman": spearman_correlation(
            pair_teacher_distances,
            pair_gt_distances,
        ),
        "selected_gt_distance_mean": float(np.mean(selected_gt)),
        "random_expected_gt_distance_mean": float(np.mean(random_gt)),
        "oracle_gt_distance_mean": float(
            np.mean([row["oracle_gt_distance"] for row in retrieval_rows])
        ),
        "gt_distance_reduction_vs_random": reduction,
        "reduction_bootstrap": interval,
        "teacher_nn_distance_median": float(np.median(teacher_nn)),
        "teacher_nn_distance_q75": float(np.quantile(teacher_nn, 0.75)),
        "criterion_minimum_reduction": MIN_GT_RETRIEVAL_REDUCTION,
        "gate_passed": bool(gate_passed),
    }


def stable_random_index(seed, anchor_file, domain_id, count):
    payload = "{}|{}|{}".format(seed, anchor_file, domain_id).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % count


def normalized_entropy(counts):
    values = np.asarray(list(counts.values()), dtype=np.float64)
    if not len(values) or values.sum() <= 0.0:
        return None
    probabilities = values / values.sum()
    entropy = -float(
        np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-12)))
    )
    maximum = math.log(len(values)) if len(values) > 1 else 0.0
    return entropy / maximum if maximum > 0.0 else 1.0


def audit_donor_support(
    anchor_rows,
    donor_rows,
    anchor_signatures,
    donor_signatures,
    anchor_styles,
    donor_styles,
    compatibility_threshold,
    args,
):
    anchor_domains = np.asarray(
        [row["domain_id"] for row in anchor_rows],
        dtype=np.int64,
    )
    donor_domains = np.asarray(
        [row["domain_id"] for row in donor_rows],
        dtype=np.int64,
    )
    donor_hashes = np.asarray([row["image_sha256"] for row in donor_rows])
    anchor_hashes = {row["image_sha256"] for row in anchor_rows}
    exact_duplicate_mask = np.asarray(
        [image_hash in anchor_hashes for image_hash in donor_hashes],
        dtype=bool,
    )

    domain_ids = sorted(set(donor_domains.tolist()))
    indices_by_domain = {
        domain_id: np.flatnonzero(
            (donor_domains == domain_id) & ~exact_duplicate_mask
        )
        for domain_id in domain_ids
    }
    for domain_id, indices in indices_by_domain.items():
        if not len(indices):
            raise ValueError(
                "donor domain {} has no non-duplicate candidates".format(domain_id)
            )

    match_rows = []
    anchor_support_rows = []
    natural_domain_counts = Counter(
        {
            donor_rows[indices_by_domain[domain_id][0]]["domain_name"]: 0
            for domain_id in domain_ids
        }
    )
    selected_donor_counts = Counter()

    for anchor_index, anchor_row in enumerate(anchor_rows):
        rows_for_anchor = []
        for domain_id in domain_ids:
            if domain_id == anchor_domains[anchor_index]:
                continue
            candidates = indices_by_domain[domain_id]
            distances = rms_distances(
                anchor_signatures[anchor_index],
                donor_signatures[candidates],
            )
            style_distances = rms_distances(
                anchor_styles[anchor_index],
                donor_styles[candidates],
            )
            matched_position = int(np.argmin(distances))
            matched_index = int(candidates[matched_position])
            random_position = stable_random_index(
                args.seed,
                anchor_row["file_name"],
                domain_id,
                len(candidates),
            )
            random_index = int(candidates[random_position])
            row = {
                "anchor_index": anchor_index,
                "anchor_file": anchor_row["file_name"],
                "anchor_domain_id": int(anchor_domains[anchor_index]),
                "anchor_domain": anchor_row["domain_name"],
                "donor_domain_id": int(domain_id),
                "donor_domain": donor_rows[matched_index]["domain_name"],
                "matched_donor_index": matched_index,
                "matched_donor_file": donor_rows[matched_index]["file_name"],
                "random_donor_index": random_index,
                "random_donor_file": donor_rows[random_index]["file_name"],
                "matched_distance": float(distances[matched_position]),
                "random_expected_distance": float(distances.mean()),
                "matched_style_distance": float(style_distances[matched_position]),
                "random_expected_style_distance": float(style_distances.mean()),
                "compatible": bool(
                    distances[matched_position] <= compatibility_threshold
                ),
            }
            rows_for_anchor.append(row)
            match_rows.append(row)
            selected_donor_counts[row["matched_donor_file"]] += 1

        if not rows_for_anchor:
            raise ValueError(
                "anchor {} has no eligible cross-domain donors".format(
                    anchor_row["file_name"]
                )
            )
        natural_match = min(rows_for_anchor, key=lambda row: row["matched_distance"])
        natural_domain_counts[natural_match["donor_domain"]] += 1
        eligible_domains = len(rows_for_anchor)
        required_domains = int(math.ceil(eligible_domains / 2.0))
        compatible_domains = sum(row["compatible"] for row in rows_for_anchor)
        anchor_support_rows.append(
            {
                "anchor_index": anchor_index,
                "anchor_file": anchor_row["file_name"],
                "anchor_domain": anchor_row["domain_name"],
                "eligible_donor_domains": eligible_domains,
                "required_compatible_domains": required_domains,
                "compatible_donor_domains": compatible_domains,
                "broad_support": compatible_domains >= required_domains,
                "mean_matched_distance": float(
                    np.mean([row["matched_distance"] for row in rows_for_anchor])
                ),
                "mean_random_expected_distance": float(
                    np.mean(
                        [row["random_expected_distance"] for row in rows_for_anchor]
                    )
                ),
                "natural_nearest_domain": natural_match["donor_domain"],
                "natural_nearest_donor_file": natural_match[
                    "matched_donor_file"
                ],
                "natural_nearest_distance": natural_match["matched_distance"],
                "natural_random_donor_file": natural_match["random_donor_file"],
            }
        )

    tau = float(np.median([row["matched_distance"] for row in match_rows]))
    for row in match_rows:
        row["compatibility_weight"] = float(
            math.exp(-0.5 * (row["matched_distance"] / max(tau, 1.0e-12)) ** 2)
        )

    domain_rows = []
    for domain_id in domain_ids:
        domain_matches = [
            row for row in match_rows if row["donor_domain_id"] == domain_id
        ]
        if not domain_matches:
            continue
        domain_rows.append(
            {
                "donor_domain_id": int(domain_id),
                "donor_domain": domain_matches[0]["donor_domain"],
                "anchors": len(domain_matches),
                "matched_distance_mean": float(
                    np.mean([row["matched_distance"] for row in domain_matches])
                ),
                "random_expected_distance_mean": float(
                    np.mean(
                        [row["random_expected_distance"] for row in domain_matches]
                    )
                ),
                "distance_reduction_vs_random": aggregate_reduction(
                    [row["matched_distance"] for row in domain_matches],
                    [row["random_expected_distance"] for row in domain_matches],
                ),
                "compatible_fraction": float(
                    np.mean([row["compatible"] for row in domain_matches])
                ),
            }
        )

    anchor_matched = np.asarray(
        [row["mean_matched_distance"] for row in anchor_support_rows]
    )
    anchor_random = np.asarray(
        [row["mean_random_expected_distance"] for row in anchor_support_rows]
    )
    reduction = aggregate_reduction(anchor_matched, anchor_random)
    interval = bootstrap_reduction(
        anchor_matched,
        anchor_random,
        args.bootstrap_samples,
        args.seed + 1,
    )
    broad_support_fraction = float(
        np.mean([row["broad_support"] for row in anchor_support_rows])
    )
    gate_passed = (
        reduction is not None
        and reduction >= MIN_DONOR_DISTANCE_REDUCTION
        and interval["lower_95"] > 0.0
        and broad_support_fraction >= MIN_BROAD_SUPPORT_FRACTION
    )

    matched_style = [row["matched_style_distance"] for row in match_rows]
    random_style = [row["random_expected_style_distance"] for row in match_rows]
    style_retention = (
        float(np.mean(matched_style)) / float(np.mean(random_style))
        if np.mean(random_style) > 0.0
        else None
    )
    return match_rows, anchor_support_rows, domain_rows, {
        "donor_domains": len(domain_ids),
        "eligible_anchor_domain_pairs": len(match_rows),
        "exact_anchor_donor_duplicates_excluded": int(exact_duplicate_mask.sum()),
        "matched_distance_mean": float(anchor_matched.mean()),
        "random_expected_distance_mean": float(anchor_random.mean()),
        "distance_reduction_vs_random": reduction,
        "reduction_bootstrap": interval,
        "compatibility_reference_q75": float(compatibility_threshold),
        "broad_support_fraction": broad_support_fraction,
        "criterion_minimum_broad_support_fraction": (
            MIN_BROAD_SUPPORT_FRACTION
        ),
        "criterion_minimum_distance_reduction": (
            MIN_DONOR_DISTANCE_REDUCTION
        ),
        "recommended_tau_median_nn": tau,
        "selected_unique_donors": len(selected_donor_counts),
        "selected_donor_fraction": len(selected_donor_counts)
        / float(max(len(match_rows), 1)),
        "maximum_donor_reuse": max(selected_donor_counts.values()),
        "natural_nearest_domain_counts": dict(natural_domain_counts),
        "natural_nearest_domain_normalized_entropy": normalized_entropy(
            natural_domain_counts
        ),
        "natural_nearest_domain_max_fraction": max(
            natural_domain_counts.values()
        )
        / float(len(anchor_rows)),
        "matched_style_distance_mean": float(np.mean(matched_style)),
        "random_expected_style_distance_mean": float(np.mean(random_style)),
        "matched_style_distance_retention": style_retention,
        "gate_passed": bool(gate_passed),
    }


def flatten_descriptor_rows(rows, class_names):
    feature_names = descriptor_feature_names(class_names)
    flattened = []
    for row in rows:
        result = {
            "role": row["role"],
            "file_name": row["file_name"],
            "image_id": row["image_id"],
            "domain_id": row["domain_id"],
            "domain_name": row["domain_name"],
            "original_height": row["original_height"],
            "original_width": row["original_width"],
            "resized_height": row["resized_height"],
            "resized_width": row["resized_width"],
            "image_sha256": row["image_sha256"],
            "mean_confidence": row["mean_confidence"],
            "mean_normalized_entropy": row["mean_normalized_entropy"],
        }
        for name, value in zip(feature_names, row["teacher_signature"]):
            result["teacher_{}".format(name)] = value
        if "gt_signature" in row:
            for name, value in zip(feature_names, row["gt_signature"]):
                result["gt_{}".format(name)] = value
        for name, value in zip(STYLE_NAMES, row["style_signature"]):
            result[name] = value
        flattened.append(result)
    return flattened


def sanitize_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def caption_panel(image, caption):
    header = np.full((28, image.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        header,
        caption,
        (7, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([header, image], axis=0)


def colorize_prediction(probabilities, colors):
    labels = probabilities.argmax(axis=0)
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        output[labels == class_id] = np.asarray(color, dtype=np.uint8)
    return output


def colorize_target(target, colors, ignore_label):
    output = np.full((*target.shape, 3), 40, dtype=np.uint8)
    for class_id, color in enumerate(colors):
        output[target == class_id] = np.asarray(color, dtype=np.uint8)
    output[target == ignore_label] = (80, 80, 80)
    return output


def resize_panel(image, size=256):
    interpolation = cv2.INTER_NEAREST if image.ndim == 2 else cv2.INTER_AREA
    return cv2.resize(image, (size, size), interpolation=interpolation)


def inference_visual_payload(record, model, cfg, min_size, max_size):
    original, resized, image_tensor, transform = prepare_image(
        record,
        cfg,
        min_size,
        max_size,
    )
    probabilities = teacher_probabilities(model, image_tensor)
    payload = {
        "image": canonical_rgb(resized, cfg.INPUT.FORMAT).copy(),
        "probabilities": probabilities,
    }
    if "sem_seg_file_name" in record:
        target = read_label(record["sem_seg_file_name"])
        if target.shape[:2] != original.shape[:2]:
            raise ValueError("visualization image and target sizes differ")
        payload["target"] = transform.apply_segmentation(target)
    return payload


def info_panel(lines, size=256):
    panel = np.full((size, size, 3), 245, dtype=np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            str(line),
            (8, 24 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    return panel


def save_retrieval_visualizations(
    output_dir,
    count,
    anchor_records,
    donor_records,
    anchor_support_rows,
    match_rows,
    model,
    cfg,
    min_size,
    max_size,
    colors,
    ignore_label,
):
    if count <= 0:
        return []
    retrieval_dir = output_dir / "retrievals"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    matches_by_anchor = {}
    for row in match_rows:
        matches_by_anchor.setdefault(row["anchor_index"], []).append(row)

    ranked = sorted(
        anchor_support_rows,
        key=lambda row: (
            row["compatible_donor_domains"],
            -row["mean_matched_distance"],
        ),
    )
    selected = ranked[: min(count, len(ranked))]
    saved = []
    with torch.no_grad():
        for support in selected:
            anchor_index = support["anchor_index"]
            natural = min(
                matches_by_anchor[anchor_index],
                key=lambda row: row["matched_distance"],
            )
            anchor_payload = inference_visual_payload(
                anchor_records[anchor_index],
                model,
                cfg,
                min_size,
                max_size,
            )
            matched_payload = inference_visual_payload(
                donor_records[natural["matched_donor_index"]],
                model,
                cfg,
                min_size,
                max_size,
            )
            random_payload = inference_visual_payload(
                donor_records[natural["random_donor_index"]],
                model,
                cfg,
                min_size,
                max_size,
            )

            anchor_image = resize_panel(anchor_payload["image"])
            anchor_gt = resize_panel(
                colorize_target(anchor_payload["target"], colors, ignore_label)
            )
            anchor_pred = resize_panel(
                colorize_prediction(anchor_payload["probabilities"], colors)
            )
            matched_image = resize_panel(matched_payload["image"])
            matched_pred = resize_panel(
                colorize_prediction(matched_payload["probabilities"], colors)
            )
            random_image = resize_panel(random_payload["image"])
            random_pred = resize_panel(
                colorize_prediction(random_payload["probabilities"], colors)
            )
            information = info_panel(
                [
                    "anchor: {}".format(support["anchor_domain"]),
                    "donor: {}".format(natural["donor_domain"]),
                    "matched d: {:.3f}".format(natural["matched_distance"]),
                    "random E[d]: {:.3f}".format(
                        natural["random_expected_distance"]
                    ),
                    "support: {}/{}".format(
                        support["compatible_donor_domains"],
                        support["eligible_donor_domains"],
                    ),
                ]
            )
            panels = [
                caption_panel(anchor_image, "anchor image"),
                caption_panel(anchor_gt, "anchor ground truth"),
                caption_panel(anchor_pred, "anchor teacher"),
                caption_panel(information, "pair statistics"),
                caption_panel(matched_image, "matched donor"),
                caption_panel(matched_pred, "matched teacher"),
                caption_panel(random_image, "random donor (same domain)"),
                caption_panel(random_pred, "random teacher"),
            ]
            canvas = np.concatenate(
                [np.concatenate(panels[:4], axis=1), np.concatenate(panels[4:], axis=1)],
                axis=0,
            )
            filename = "{:03d}_{}.jpg".format(
                anchor_index,
                sanitize_name(Path(support["anchor_file"]).stem),
            )
            path = retrieval_dir / filename
            cv2.imwrite(str(path), canvas[:, :, ::-1])
            saved.append(str(path.resolve()))
    return saved


def select_records(dataset_name, maximum):
    all_records = sorted(
        DatasetCatalog.get(dataset_name),
        key=lambda record: record["file_name"],
    )
    if maximum > 0 and maximum < len(all_records):
        by_domain = OrderedDict()
        for record in all_records:
            by_domain.setdefault(int(record["domain_id"]), []).append(record)
        selected = []
        offset = 0
        domain_ids = sorted(by_domain)
        while len(selected) < maximum:
            added = False
            for domain_id in domain_ids:
                domain_records = by_domain[domain_id]
                if offset < len(domain_records):
                    selected.append(domain_records[offset])
                    added = True
                    if len(selected) == maximum:
                        break
            if not added:
                break
            offset += 1
    else:
        selected = all_records
    if not selected:
        raise ValueError("dataset contains no records: {}".format(dataset_name))
    return all_records, selected


def main():
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "anchors": cache_dir / "anchors.jsonl",
        "validation": cache_dir / "validation.jsonl",
        "donors": cache_dir / "donors.jsonl",
    }
    cache_meta_path = cache_dir / "meta.json"
    if args.recompute_cache:
        for path in list(cache_paths.values()) + [cache_meta_path]:
            if path.exists():
                path.unlink()

    cfg = build_cfg(args)
    min_size = args.min_size or int(cfg.INPUT.MIN_SIZE_TEST)
    max_size = args.max_size or int(cfg.INPUT.MAX_SIZE_TEST)
    if min_size <= 0 or max_size <= 0:
        raise ValueError("input resize dimensions must be positive")

    all_anchors, anchor_records = select_records(
        args.anchor_dataset,
        args.max_anchors,
    )
    all_validation, validation_records = select_records(
        args.validation_dataset,
        args.max_validation,
    )
    all_donors, donor_records = select_records(
        args.donor_dataset,
        args.max_donors,
    )

    model = build_model(cfg)
    checkpoint_info = load_checkpoint_branch(
        model,
        args.checkpoint,
        args.checkpoint_branch,
    )
    model.eval()
    print(
        "Loaded {} checkpoint branch ({:.2%}): {}".format(
            checkpoint_info["loaded_branch"],
            checkpoint_info["loaded_state_fraction"],
            checkpoint_info["path"],
        ),
        flush=True,
    )

    anchor_metadata = MetadataCatalog.get(args.anchor_dataset)
    class_names = list(getattr(anchor_metadata, "stuff_classes", []))
    num_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
    if len(class_names) != num_classes:
        class_names = ["class_{}".format(index) for index in range(num_classes)]
    colors = list(getattr(anchor_metadata, "stuff_colors", []))
    if len(colors) != num_classes:
        colors = [(0, 0, 0), (50, 255, 132), (50, 132, 255), (214, 255, 50)]
    ignore_label = int(getattr(anchor_metadata, "ignore_label", 255))

    cache_meta = {
        "audit_version": AUDIT_VERSION,
        "descriptor_version": DESCRIPTOR_VERSION,
        "config_file": str(Path(args.config_file).resolve()),
        "checkpoint": checkpoint_identity(args.checkpoint),
        "checkpoint_branch": checkpoint_info["loaded_branch"],
        "device": args.device,
        "config_opts": list(args.opts),
        "min_size": min_size,
        "max_size": max_size,
        "num_classes": num_classes,
        "anchor_dataset": args.anchor_dataset,
        "validation_dataset": args.validation_dataset,
        "donor_dataset": args.donor_dataset,
        "anchor_records": len(anchor_records),
        "validation_records": len(validation_records),
        "donor_records": len(donor_records),
        "anchor_fingerprint": records_fingerprint(anchor_records),
        "validation_fingerprint": records_fingerprint(validation_records),
        "donor_fingerprint": records_fingerprint(donor_records),
    }
    if cache_meta_path.exists():
        previous_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        if previous_meta != cache_meta:
            differing = [
                key
                for key in sorted(set(previous_meta) | set(cache_meta))
                if previous_meta.get(key) != cache_meta.get(key)
            ]
            raise RuntimeError(
                "descriptor cache settings changed ({}); use a new OUTPUT_DIR "
                "or set RECOMPUTE_CACHE=1".format(", ".join(differing))
            )
    else:
        write_json(cache_meta_path, cache_meta)

    anchor_rows = extract_descriptors(
        "anchor",
        anchor_records,
        cache_paths["anchors"],
        model,
        cfg,
        min_size,
        max_size,
        num_classes,
        ignore_label,
        args.progress_every,
    )
    validation_rows = extract_descriptors(
        "validation",
        validation_records,
        cache_paths["validation"],
        model,
        cfg,
        min_size,
        max_size,
        num_classes,
        ignore_label,
        args.progress_every,
    )
    donor_rows = extract_descriptors(
        "donor",
        donor_records,
        cache_paths["donors"],
        model,
        cfg,
        min_size,
        max_size,
        num_classes,
        ignore_label,
        args.progress_every,
    )

    donor_raw = np.asarray(
        [row["teacher_signature"] for row in donor_rows],
        dtype=np.float64,
    )
    descriptor_center, descriptor_scale = robust_standardizer(donor_raw)
    donor_signatures = standardize(
        donor_raw,
        descriptor_center,
        descriptor_scale,
    )
    anchor_signatures = standardize(
        [row["teacher_signature"] for row in anchor_rows],
        descriptor_center,
        descriptor_scale,
    )
    validation_signatures = standardize(
        [row["teacher_signature"] for row in validation_rows],
        descriptor_center,
        descriptor_scale,
    )
    validation_gt_signatures = standardize(
        [row["gt_signature"] for row in validation_rows],
        descriptor_center,
        descriptor_scale,
    )

    style_center, style_scale = robust_standardizer(
        [row["style_signature"] for row in donor_rows]
    )
    donor_styles = standardize(
        [row["style_signature"] for row in donor_rows],
        style_center,
        style_scale,
    )
    anchor_styles = standardize(
        [row["style_signature"] for row in anchor_rows],
        style_center,
        style_scale,
    )

    validation_confusion = np.sum(
        [
            np.asarray(row["confusion"], dtype=np.int64).reshape(
                num_classes,
                num_classes,
            )
            for row in validation_rows
        ],
        axis=0,
    )
    validation_metrics = summarize_confusion(validation_confusion, class_names)
    expected_miou = args.expected_validation_miou
    validation_is_full = len(validation_records) == len(all_validation)
    if expected_miou is None or not validation_is_full:
        parity_error = None
        parity_passed = True
        parity_verified = False
    else:
        if expected_miou > 1.0:
            expected_miou /= 100.0
        parity_error = validation_metrics["mean_iou"] - expected_miou
        parity_passed = abs(parity_error) <= args.parity_tolerance
        parity_verified = True
    validation_metrics.update(
        {
            "expected_mean_iou": expected_miou,
            "difference": parity_error,
            "tolerance": args.parity_tolerance,
            "verified": parity_verified,
            "gate_passed": bool(parity_passed),
        }
    )

    labeled_retrieval_rows, signature_validation = validate_signature_retrieval(
        validation_rows,
        validation_signatures,
        validation_gt_signatures,
        args,
    )
    compatibility_threshold = signature_validation[
        "teacher_nn_distance_q75"
    ]
    (
        donor_match_rows,
        anchor_support_rows,
        domain_support_rows,
        donor_support,
    ) = audit_donor_support(
        anchor_rows,
        donor_rows,
        anchor_signatures,
        donor_signatures,
        anchor_styles,
        donor_styles,
        compatibility_threshold,
        args,
    )

    retrieval_visualizations = save_retrieval_visualizations(
        output_dir,
        args.visualize,
        anchor_records,
        donor_records,
        anchor_support_rows,
        donor_match_rows,
        model,
        cfg,
        min_size,
        max_size,
        colors,
        ignore_label,
    )

    write_csv(
        output_dir / "anchors.csv",
        flatten_descriptor_rows(anchor_rows, class_names),
    )
    write_csv(
        output_dir / "validation.csv",
        flatten_descriptor_rows(validation_rows, class_names),
    )
    write_csv(
        output_dir / "donors.csv",
        flatten_descriptor_rows(donor_rows, class_names),
    )
    write_csv(output_dir / "labeled_retrieval.csv", labeled_retrieval_rows)
    write_csv(output_dir / "donor_matches.csv", donor_match_rows)
    write_csv(output_dir / "anchor_support.csv", anchor_support_rows)
    write_csv(output_dir / "domain_support.csv", domain_support_rows)

    full_audit = (
        len(anchor_records) == len(all_anchors)
        and len(validation_records) == len(all_validation)
        and len(donor_records) == len(all_donors)
    )
    gates_passed = (
        parity_passed
        and signature_validation["gate_passed"]
        and donor_support["gate_passed"]
    )
    if not full_audit:
        verdict = "smoke_only"
    elif gates_passed:
        verdict = "care_phase0_supported"
    else:
        verdict = "care_phase0_not_supported"

    summary = {
        "audit_version": AUDIT_VERSION,
        "descriptor_version": DESCRIPTOR_VERSION,
        "full_audit": full_audit,
        "config_file": str(Path(args.config_file).resolve()),
        "checkpoint": checkpoint_info,
        "datasets": {
            "anchors": {
                "name": args.anchor_dataset,
                "selected": len(anchor_records),
                "available": len(all_anchors),
            },
            "validation": {
                "name": args.validation_dataset,
                "selected": len(validation_records),
                "available": len(all_validation),
            },
            "donors": {
                "name": args.donor_dataset,
                "selected": len(donor_records),
                "available": len(all_donors),
            },
        },
        "input_operating_condition": {
            "short_edge": min_size,
            "max_size": max_size,
            "note": (
                "Frozen Stage-I inference setting; all spatial descriptor "
                "coordinates are normalized and resolution-independent."
            ),
        },
        "classes": class_names,
        "descriptor": {
            "moments_per_class": list(MOMENT_NAMES),
            "dimensions": len(class_names) * len(MOMENT_NAMES),
            "standardization": "donor median and robust IQR scale",
            "center": descriptor_center.tolist(),
            "scale": descriptor_scale.tolist(),
        },
        "validation_inference": validation_metrics,
        "signature_validation": signature_validation,
        "donor_support": donor_support,
        "domain_support": domain_support_rows,
        "decision_criteria": {
            "minimum_gt_retrieval_reduction": MIN_GT_RETRIEVAL_REDUCTION,
            "minimum_donor_distance_reduction": MIN_DONOR_DISTANCE_REDUCTION,
            "minimum_broad_support_fraction": MIN_BROAD_SUPPORT_FRACTION,
            "broad_support_definition": (
                "nearest donor falls within validation cross-domain NN Q75 "
                "for at least half of eligible donor domains"
            ),
        },
        "gates": {
            "inference_parity": bool(parity_passed),
            "out_of_sample_signature": bool(
                signature_validation["gate_passed"]
            ),
            "cross_domain_donor_support": bool(donor_support["gate_passed"]),
        },
        "verdict": verdict,
        "retrieval_visualizations": retrieval_visualizations,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "summary.json", summary)

    print("\nCARE Phase-0 audit complete", flush=True)
    print(
        "  anchors / validation / donors: {} / {} / {}".format(
            len(anchor_records),
            len(validation_records),
            len(donor_records),
        ),
        flush=True,
    )
    print(
        "  validation mIoU: {:.6f} (parity: {})".format(
            validation_metrics["mean_iou"],
            "pass" if parity_passed else "FAIL",
        ),
        flush=True,
    )
    print(
        "  teacher/GT pairwise rank correlation: {}".format(
            "n/a"
            if signature_validation["teacher_gt_pairwise_spearman"] is None
            else "{:.6f}".format(
                signature_validation["teacher_gt_pairwise_spearman"]
            )
        ),
        flush=True,
    )
    print(
        "  GT retrieval distance reduction: {:.2%} [{:.2%}, {:.2%}]".format(
            signature_validation["gt_distance_reduction_vs_random"],
            signature_validation["reduction_bootstrap"]["lower_95"],
            signature_validation["reduction_bootstrap"]["upper_95"],
        ),
        flush=True,
    )
    print(
        "  donor matching distance reduction: {:.2%}".format(
            donor_support["distance_reduction_vs_random"]
        ),
        flush=True,
    )
    print(
        "  anchors with broad cross-domain support: {:.2%}".format(
            donor_support["broad_support_fraction"]
        ),
        flush=True,
    )
    print(
        "  matched acquisition-style distance retained: {:.2%}".format(
            donor_support["matched_style_distance_retention"]
        ),
        flush=True,
    )
    print("  verdict: {}".format(verdict), flush=True)
    print("  artifacts: {}".format(output_dir.resolve()), flush=True)


if __name__ == "__main__":
    main()
