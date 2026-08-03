#!/usr/bin/env python3
"""Prepare a portable shallow-feature statistic bank for CARE Phase 1."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectron2.modeling import build_model  # noqa: E402
from detectron2.structures import ImageList  # noqa: E402
from mask2former.care_protocol import (  # noqa: E402
    CARE_BANK_VERSION,
    validate_phase0_protocol,
)


AUDIT_PATH = REPO_ROOT / "tools/audit_care_phase0.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("care_phase0_audit", AUDIT_PATH)
AUDIT_MODULE = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT_MODULE)
build_cfg = AUDIT_MODULE.build_cfg
load_checkpoint_branch = AUDIT_MODULE.load_checkpoint_branch
prepare_image = AUDIT_MODULE.prepare_image
select_records = AUDIT_MODULE.select_records


BANK_VERSION = CARE_BANK_VERSION


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen Stage-I shallow-feature statistics only for "
            "Phase-0-supported CARE donors."
        )
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase0-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint-branch",
        choices=("auto", "plain", "teacher", "student"),
        default="auto",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-name", default="res2")
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def atomic_write_npz(path, **payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(str(temporary), str(path))


def parse_boolean(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def checkpoint_identity(path):
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "basename": resolved.name,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_supported_pairs(phase0_dir, summary, anchor_records, donor_records):
    support_rows = read_csv(phase0_dir / "anchor_support.csv")
    match_rows = read_csv(phase0_dir / "donor_matches.csv")

    supported = {}
    for row in support_rows:
        anchor_index = int(row["anchor_index"])
        if not 0 <= anchor_index < len(anchor_records):
            raise ValueError("Phase-0 anchor index is out of range")
        anchor_key = Path(anchor_records[anchor_index]["file_name"]).name
        if anchor_key != Path(row["anchor_file"]).name:
            raise ValueError("Phase-0 anchor ordering differs from current dataset")
        if parse_boolean(row["broad_support"]):
            supported[anchor_index] = {
                "anchor_key": anchor_key,
                "anchor_domain_id": int(anchor_records[anchor_index]["domain_id"]),
                "choices": [],
            }

    selected_donor_indices = set()
    for row in match_rows:
        anchor_index = int(row["anchor_index"])
        if anchor_index not in supported or not parse_boolean(row["compatible"]):
            continue
        donor_index = int(row["matched_donor_index"])
        if not 0 <= donor_index < len(donor_records):
            raise ValueError("Phase-0 donor index is out of range")
        donor_record = donor_records[donor_index]
        if Path(donor_record["file_name"]).name != Path(row["matched_donor_file"]).name:
            raise ValueError("Phase-0 donor ordering differs from current dataset")
        if int(donor_record["domain_id"]) == supported[anchor_index]["anchor_domain_id"]:
            raise ValueError("CARE donor must be cross-domain")

        weight = float(row["compatibility_weight"])
        if not 0.0 < weight <= 1.0:
            raise ValueError("invalid Phase-0 compatibility weight")
        supported[anchor_index]["choices"].append(
            {
                "source_donor_index": donor_index,
                "donor_domain_id": int(donor_record["domain_id"]),
                "distance": float(row["matched_distance"]),
                "compatibility_weight": weight,
            }
        )
        selected_donor_indices.add(donor_index)

    required_domains = {
        int(row["anchor_index"]): int(row["required_compatible_domains"])
        for row in support_rows
    }
    for anchor_index, entry in supported.items():
        if len(entry["choices"]) < required_domains[anchor_index]:
            raise ValueError(
                "supported anchor {} has too few compatible donors".format(
                    entry["anchor_key"]
                )
            )

    expected_fraction = float(
        summary["donor_support"]["broad_support_fraction"]
    )
    observed_fraction = len(supported) / float(max(len(anchor_records), 1))
    if abs(expected_fraction - observed_fraction) > 1.0e-12:
        raise ValueError("Phase-0 broad-support count is inconsistent")

    donor_indices = sorted(selected_donor_indices)
    bank_index = {source_index: index for index, source_index in enumerate(donor_indices)}
    anchors = OrderedDict()
    for anchor_index in sorted(supported):
        entry = supported[anchor_index]
        choices = []
        for choice in sorted(
            entry["choices"],
            key=lambda value: value["donor_domain_id"],
        ):
            choice = dict(choice)
            source_index = choice.pop("source_donor_index")
            if source_index not in bank_index:
                raise RuntimeError("CARE donor bank index construction failed")
            choice["donor_index"] = bank_index[source_index]
            choices.append(choice)
        anchors[entry["anchor_key"]] = choices
    return anchors, donor_indices


def donor_identifier(record):
    image_id = str(record.get("image_id", ""))
    if image_id:
        return image_id
    return "{}__{}".format(record["domain_name"], Path(record["file_name"]).stem)


def load_cache(path, donor_ids, channels):
    if not path.is_file():
        return (
            np.full((len(donor_ids), channels), np.nan, dtype=np.float32),
            np.full((len(donor_ids), channels), np.nan, dtype=np.float32),
            np.zeros(len(donor_ids), dtype=np.bool_),
        )
    with np.load(str(path), allow_pickle=False) as payload:
        cached_ids = [str(value) for value in payload["donor_ids"]]
        if cached_ids != donor_ids:
            raise ValueError("CARE cache donor ordering changed")
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        completed = np.asarray(payload["completed"], dtype=np.bool_)
    if mean.shape != (len(donor_ids), channels):
        raise ValueError("CARE cache channel count changed")
    if std.shape != mean.shape or completed.shape != (len(donor_ids),):
        raise ValueError("CARE cache shape is invalid")
    return mean, std, completed


def save_cache(path, donor_ids, mean, std, completed):
    atomic_write_npz(
        path,
        donor_ids=np.asarray(donor_ids, dtype=np.str_),
        mean=mean,
        std=std,
        completed=completed,
    )


def extract_feature_statistics(
    role,
    records,
    record_ids,
    cache_path,
    model,
    cfg,
    feature_name,
    channels,
    min_size,
    max_size,
    save_every,
):
    mean, std, completed = load_cache(cache_path, record_ids, channels)
    initially_completed = int(completed.sum())
    with torch.inference_mode():
        for index, record in enumerate(records):
            if completed[index]:
                continue
            _, _, image_tensor, _ = prepare_image(
                record,
                cfg,
                min_size,
                max_size,
            )
            image_tensor = image_tensor.to(model.device)
            normalized = (image_tensor - model.pixel_mean) / model.pixel_std
            images = ImageList.from_tensors(
                [normalized],
                model.size_divisibility,
            )
            feature = model.backbone(images.tensor)[feature_name].float()
            channel_mean = feature.mean(dim=(2, 3))[0]
            channel_std = (
                (feature - channel_mean.view(1, -1, 1, 1))
                .square()
                .mean(dim=(2, 3))[0]
                .add(1.0e-6)
                .sqrt()
            )
            mean[index] = channel_mean.cpu().numpy()
            std[index] = channel_std.cpu().numpy()
            completed[index] = True

            done = int(completed.sum())
            if done % save_every == 0 or done == len(records):
                save_cache(cache_path, record_ids, mean, std, completed)
                print(
                    "Prepared {}/{} CARE {}".format(done, len(records), role),
                    flush=True,
                )

    if not completed.all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("CARE {} feature cache is incomplete".format(role))
    return mean, std, initially_completed


def intervention_preflight(anchors, anchor_ids, anchor_mean, anchor_std, donor_mean, donor_std):
    anchor_bank_index = {anchor_id: index for index, anchor_id in enumerate(anchor_ids)}
    normalized_mean_shifts = []
    std_ratios = []
    weights = []
    for anchor_id, choices in anchors.items():
        anchor_index = anchor_bank_index[anchor_id]
        source_mean = anchor_mean[anchor_index]
        source_std = np.maximum(anchor_std[anchor_index], 1.0e-6)
        for choice in choices:
            donor_index = int(choice["donor_index"])
            weight = float(choice["compatibility_weight"])
            target_mean = (1.0 - weight) * source_mean + weight * donor_mean[donor_index]
            target_std = (1.0 - weight) * source_std + weight * donor_std[donor_index]
            normalized_mean_shifts.extend(
                (np.abs(target_mean - source_mean) / source_std).tolist()
            )
            std_ratios.extend((target_std / source_std).tolist())
            weights.append(weight)

    normalized_mean_shifts = np.asarray(normalized_mean_shifts, dtype=np.float64)
    std_ratios = np.asarray(std_ratios, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (
        np.isfinite(normalized_mean_shifts).all()
        and np.isfinite(std_ratios).all()
        and np.isfinite(weights).all()
        and np.all(std_ratios > 0)
    ):
        raise RuntimeError("CARE intervention preflight produced invalid statistics")

    def quantiles(values):
        return {
            "q01": float(np.quantile(values, 0.01)),
            "q05": float(np.quantile(values, 0.05)),
            "median": float(np.quantile(values, 0.50)),
            "q95": float(np.quantile(values, 0.95)),
            "q99": float(np.quantile(values, 0.99)),
        }

    return {
        "compatibility_weight": quantiles(weights),
        "normalized_channel_mean_shift": quantiles(normalized_mean_shifts),
        "target_to_anchor_channel_std_ratio": quantiles(std_ratios),
        "extreme_std_ratio_fraction": float(
            np.mean((std_ratios < 0.20) | (std_ratios > 5.0))
        ),
    }


def main():
    args = parse_args()
    if args.save_every < 1:
        raise ValueError("--save-every must be positive")
    started = time.time()
    phase0_dir = Path(args.phase0_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase0_paths = {
        "summary": phase0_dir / "summary.json",
        "support": phase0_dir / "anchor_support.csv",
        "matches": phase0_dir / "donor_matches.csv",
    }
    missing = [str(path) for path in phase0_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Phase-0 artifact: {}".format(missing[0]))
    phase0 = read_json(phase0_paths["summary"])
    runtime_calibration = validate_phase0_protocol(phase0)
    if phase0.get("verdict") != "care_phase0_supported":
        raise RuntimeError("Phase 0 did not support CARE")
    if not phase0.get("full_audit"):
        raise RuntimeError("CARE bank requires the full Phase-0 audit")

    cfg_args = SimpleNamespace(
        config_file=args.config_file,
        opts=args.opts,
        device=args.device,
    )
    cfg = build_cfg(cfg_args)
    min_size = args.min_size or int(cfg.INPUT.MIN_SIZE_TEST)
    max_size = args.max_size or int(cfg.INPUT.MAX_SIZE_TEST)

    anchor_dataset = phase0["datasets"]["anchors"]["name"]
    donor_dataset = phase0["datasets"]["donors"]["name"]
    _, anchor_records = select_records(anchor_dataset, 0)
    _, donor_records = select_records(donor_dataset, 0)
    if len(anchor_records) != int(phase0["datasets"]["anchors"]["selected"]):
        raise ValueError("current anchor dataset differs from Phase 0")
    if len(donor_records) != int(phase0["datasets"]["donors"]["selected"]):
        raise ValueError("current donor dataset differs from Phase 0")

    anchors, source_donor_indices = build_supported_pairs(
        phase0_dir,
        phase0,
        anchor_records,
        donor_records,
    )
    selected_records = [donor_records[index] for index in source_donor_indices]
    donor_ids = [donor_identifier(record) for record in selected_records]
    if len(set(donor_ids)) != len(donor_ids):
        raise ValueError("CARE donor identifiers are not unique")

    model = build_model(cfg)
    checkpoint_info = load_checkpoint_branch(
        model,
        args.checkpoint,
        args.checkpoint_branch,
    )
    model.eval()
    output_shapes = model.backbone.output_shape()
    if args.feature_name not in output_shapes:
        raise KeyError(
            "feature {} is not returned by the backbone".format(args.feature_name)
        )
    channels = int(output_shapes[args.feature_name].channels)

    phase0_hashes = {
        name: sha256_file(path) for name, path in phase0_paths.items()
    }
    provenance = {
        "bank_version": BANK_VERSION,
        "phase0_audit_version": int(phase0["audit_version"]),
        "runtime_calibration": runtime_calibration,
        "phase0_hashes": phase0_hashes,
        "config_file": str(Path(args.config_file).resolve()),
        "checkpoint": checkpoint_identity(args.checkpoint),
        "loaded_checkpoint_branch": checkpoint_info["loaded_branch"],
        "feature_name": args.feature_name,
        "channels": channels,
        "min_size": min_size,
        "max_size": max_size,
        "donor_ids": donor_ids,
    }
    meta_path = output_dir / "cache_meta.json"
    cache_path = output_dir / "feature_cache.npz"
    anchor_cache_path = output_dir / "anchor_feature_cache.npz"
    if args.force:
        for path in (
            meta_path,
            cache_path,
            anchor_cache_path,
            output_dir / "manifest.json",
            output_dir / "feature_bank.npz",
        ):
            if path.exists():
                path.unlink()
    if meta_path.is_file():
        previous = read_json(meta_path)
        if previous != provenance:
            raise RuntimeError(
                "CARE bank settings changed; use a new OUTPUT_DIR or --force"
            )
    else:
        atomic_write_json(meta_path, provenance)

    mean, std, initially_completed = extract_feature_statistics(
        "donors",
        selected_records,
        donor_ids,
        cache_path,
        model,
        cfg,
        args.feature_name,
        channels,
        min_size,
        max_size,
        args.save_every,
    )

    anchor_record_by_key = {
        Path(record["file_name"]).name: record for record in anchor_records
    }
    supported_anchor_ids = list(anchors)
    supported_anchor_records = [
        anchor_record_by_key[anchor_id] for anchor_id in supported_anchor_ids
    ]
    anchor_mean, anchor_std, initially_completed_anchors = extract_feature_statistics(
        "anchors",
        supported_anchor_records,
        supported_anchor_ids,
        anchor_cache_path,
        model,
        cfg,
        args.feature_name,
        channels,
        min_size,
        max_size,
        args.save_every,
    )
    preflight = intervention_preflight(
        anchors,
        supported_anchor_ids,
        anchor_mean,
        anchor_std,
        mean,
        std,
    )

    bank_path = output_dir / "feature_bank.npz"
    atomic_write_npz(
        bank_path,
        donor_ids=np.asarray(donor_ids, dtype=np.str_),
        mean=mean,
        std=std,
    )

    donors = []
    for bank_index, record in enumerate(selected_records):
        donors.append(
            {
                "donor_id": donor_ids[bank_index],
                "domain_id": int(record["domain_id"]),
                "domain_name": str(record["domain_name"]),
                "source_basename": Path(record["file_name"]).name,
            }
        )
    manifest = {
        "bank_version": BANK_VERSION,
        "phase0_audit_version": int(phase0["audit_version"]),
        "phase0_verdict": phase0["verdict"],
        "descriptor_version": phase0["descriptor_version"],
        "runtime_calibration": runtime_calibration,
        "phase0_artifact_hashes": phase0_hashes,
        "feature_name": args.feature_name,
        "channels": channels,
        "checkpoint_branch": checkpoint_info["loaded_branch"],
        "input_resize": {"short_edge": min_size, "max_size": max_size},
        "supported_anchor_count": len(anchors),
        "total_anchor_count": len(anchor_records),
        "pair_count": sum(len(choices) for choices in anchors.values()),
        "donor_count": len(donors),
        "feature_bank_sha256": sha256_file(bank_path),
        "intervention_preflight": preflight,
        "donors": donors,
        "anchors": anchors,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)

    elapsed = time.time() - started
    print("\nCARE Phase-1 bank complete", flush=True)
    print(
        "  protocol: Phase-0 v{} / {}".format(
            manifest["phase0_audit_version"],
            runtime_calibration["source"],
        ),
        flush=True,
    )
    print(
        "  supported anchors / pairs / donors: {} / {} / {}".format(
            manifest["supported_anchor_count"],
            manifest["pair_count"],
            manifest["donor_count"],
        ),
        flush=True,
    )
    print("  feature: {} x {}".format(args.feature_name, channels), flush=True)
    print(
        "  compatibility weight median / q95: {:.3f} / {:.3f}".format(
            preflight["compatibility_weight"]["median"],
            preflight["compatibility_weight"]["q95"],
        ),
        flush=True,
    )
    print(
        "  normalized mean shift median / q95: {:.3f} / {:.3f}".format(
            preflight["normalized_channel_mean_shift"]["median"],
            preflight["normalized_channel_mean_shift"]["q95"],
        ),
        flush=True,
    )
    print(
        "  target/anchor std ratio q01 / median / q99: {:.3f} / {:.3f} / {:.3f}".format(
            preflight["target_to_anchor_channel_std_ratio"]["q01"],
            preflight["target_to_anchor_channel_std_ratio"]["median"],
            preflight["target_to_anchor_channel_std_ratio"]["q99"],
        ),
        flush=True,
    )
    print(
        "  extreme std-ratio fraction: {:.4%}".format(
            preflight["extreme_std_ratio_fraction"]
        ),
        flush=True,
    )
    print(
        "  resumed donors / anchors: {} / {}".format(
            initially_completed,
            initially_completed_anchors,
        ),
        flush=True,
    )
    print("  elapsed seconds: {:.2f}".format(elapsed), flush=True)
    print("  artifacts: {}".format(output_dir.resolve()), flush=True)


if __name__ == "__main__":
    main()
