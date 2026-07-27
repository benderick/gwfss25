#!/usr/bin/env python3
"""Quantify organ imbalance and geometric survival across feature strides."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "topowheat-matplotlib"),
)

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize


TARGET_SIZE = 512
THRESHOLDS = np.arange(1, 65, dtype=int)
STRIDES = np.array([2, 4, 8, 16, 32], dtype=int)
CLASS_INFO = {
    1: ("Head", "#2E9D70"),
    2: ("Stem", "#326CB0"),
    3: ("Leaf", "#9AAF32"),
}
RGB_TO_CLASS = {
    (0, 0, 0): 0,
    (50, 255, 132): 1,
    (50, 132, 255): 2,
    (214, 255, 50): 3,
}


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "GWFSS" / "GWFSS_v1.0_labelled",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "analysis_outputs" / "organ_scale_burden",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=repo_root / "Paper_Template" / "figures",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--reuse-manifest",
        action="store_true",
        help="Reuse per_image_organ_geometry.csv instead of recomputing masks.",
    )
    return parser.parse_args()


def find_pairs(dataset_root):
    image_root = dataset_root / "images"
    label_root = dataset_root / "class_id"
    if not label_root.is_dir():
        label_root = dataset_root / "masks"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(
            "Expected images/ and class_id/ or masks/ under {}".format(dataset_root)
        )

    pairs = []
    for image_path in sorted(image_root.glob("*/*.png")):
        relative = image_path.relative_to(image_root)
        mask_path = label_root / relative
        if not mask_path.is_file():
            raise FileNotFoundError("Missing mask for {}".format(image_path))
        pairs.append((image_path.parent.name, image_path, mask_path))
    if len(pairs) != 1096:
        raise RuntimeError("Expected 1,096 image-mask pairs, found {}".format(len(pairs)))
    return pairs


def load_class_mask(path):
    raw = np.asarray(Image.open(path))
    if raw.ndim == 2:
        labels = raw.astype(np.uint8)
    else:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        labels = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        for color, class_id in RGB_TO_CLASS.items():
            labels[np.all(rgb == color, axis=-1)] = class_id
    unknown = np.setdiff1d(np.unique(labels), np.array([0, 1, 2, 3], dtype=np.uint8))
    if unknown.size:
        raise ValueError("Unknown mask values {} in {}".format(unknown.tolist(), path))
    return labels


def normalize_mask(labels):
    if labels.shape == (TARGET_SIZE, TARGET_SIZE):
        return labels
    return cv2.resize(
        labels,
        (TARGET_SIZE, TARGET_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )


def majority_rasterize(binary_mask, stride):
    grid = binary_mask.reshape(
        TARGET_SIZE // stride,
        stride,
        TARGET_SIZE // stride,
        stride,
    ).mean(axis=(1, 3))
    coarse = grid >= 0.5
    return np.repeat(np.repeat(coarse, stride, axis=0), stride, axis=1)


def process_pair(item):
    domain, image_path, mask_path = item
    native_labels = load_class_mask(mask_path)
    native_height, native_width = native_labels.shape
    labels = normalize_mask(native_labels)
    total_pixels = labels.size
    rows = []

    for class_id, (organ, _) in CLASS_INFO.items():
        binary = labels == class_id
        area_pixels = int(binary.sum())
        row = {
            "domain": domain,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "native_width": native_width,
            "native_height": native_height,
            "organ": organ,
            "present": bool(area_pixels),
            "area_pct": 100.0 * area_pixels / total_pixels,
        }

        if not area_pixels:
            row["median_diameter_px"] = float("nan")
            for threshold in THRESHOLDS:
                row["survival_{:02d}".format(threshold)] = float("nan")
            for stride in STRIDES:
                row["retention_s{:02d}".format(stride)] = float("nan")
            rows.append(row)
            continue

        skeleton = skeletonize(binary)
        distance = ndimage.distance_transform_edt(binary)
        local_diameters = 2.0 * distance[skeleton]
        row["median_diameter_px"] = float(np.median(local_diameters))
        for threshold in THRESHOLDS:
            row["survival_{:02d}".format(threshold)] = float(
                np.mean(local_diameters >= threshold)
            )
        for stride in STRIDES:
            rasterized = majority_rasterize(binary, int(stride))
            row["retention_s{:02d}".format(stride)] = float(
                np.mean(rasterized[skeleton])
            )
        rows.append(row)
    return rows


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 9.5,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.75,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_panel_label(axis, label):
    axis.text(
        -0.15,
        1.08,
        label,
        transform=axis.transAxes,
            fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_occupancy(axis, records):
    positions = {"Head": 2, "Stem": 1, "Leaf": 0}
    for organ, position in positions.items():
        _, color = next(info for info in CLASS_INFO.values() if info[0] == organ)
        subset = records[(records["organ"] == organ) & records["present"]]
        log_values = np.log10(subset["area_pct"].to_numpy())
        violin = axis.violinplot(
            [log_values],
            positions=[position],
            orientation="horizontal",
            widths=0.72,
            showextrema=False,
            bw_method=0.22,
        )
        body = violin["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor("#2E3439")
        body.set_linewidth(0.65)
        body.set_alpha(0.70)
        q25, median, q75 = np.percentile(log_values, [25, 50, 75])
        axis.plot([q25, q75], [position, position], color="#20252A", linewidth=3.0)
        axis.scatter(
            [median],
            [position],
            s=24,
            color="white",
            edgecolor="#20252A",
            linewidth=0.8,
            zorder=4,
        )
        median_raw = float(subset["area_pct"].median())
        absent_pct = 100.0 * (1.0 - len(subset) / len(records[records["organ"] == organ]))
        axis.text(
            0.98,
            position + 0.28,
            "median {:.2f}%  |  absent {:.1f}%".format(median_raw, absent_pct),
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8.2,
            color="#343A40",
        )

    tick_values = np.array([0.001, 0.01, 0.1, 1, 10, 100])
    axis.set_xticks(np.log10(tick_values))
    axis.set_xticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])
    axis.set_xlim(-3.1, 2.1)
    axis.set_yticks([2, 1, 0])
    axis.set_yticklabels(["Head", "Stem", "Leaf"])
    axis.set_xlabel("Organ area per image (%)")
    axis.set_title("Per-image pixel occupancy", loc="center", pad=8)
    axis.grid(axis="x", color="#D9DDE1", linewidth=0.55, alpha=0.85)
    axis.set_axisbelow(True)
    add_panel_label(axis, "(a)")


def plot_survival(axis, records):
    for _, (organ, color) in CLASS_INFO.items():
        subset = records[(records["organ"] == organ) & records["present"]]
        survival_columns = ["survival_{:02d}".format(value) for value in THRESHOLDS]
        values = 100.0 * subset[survival_columns].to_numpy(dtype=float)
        median = np.nanmedian(values, axis=0)
        q25 = np.nanpercentile(values, 25, axis=0)
        q75 = np.nanpercentile(values, 75, axis=0)
        axis.fill_between(THRESHOLDS, q25, q75, color=color, alpha=0.13, linewidth=0)
        axis.plot(THRESHOLDS, median, color=color, linewidth=2.0, label=organ)

    for stride in (4, 8, 16, 32):
        axis.axvline(stride, color="#687078", linestyle=(0, (2, 2)), linewidth=0.8)
        # axis.text(
        #     stride,
        #     103,
        #     "s={}".format(stride),
        #     ha="center",
        #     va="bottom",
        #     fontsize=7.1,
        #     color="#51585F",
        # )
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 2, 4, 8, 16, 32, 64])
    axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axis.set_xlim(1, 64)
    axis.set_ylim(0, 108)
    axis.set_xlabel("Diameter threshold (px at 512 x 512)")
    axis.set_ylabel("Centerline above threshold (%)")
    axis.set_title("Medial-diameter survival", loc="center", pad=8)
    axis.grid(color="#D9DDE1", linewidth=0.55, alpha=0.85)
    axis.legend(frameon=False, loc="lower left", ncol=3, handlelength=1.5)
    axis.set_axisbelow(True)
    add_panel_label(axis, "(b)")


def plot_retention(axis, records):
    for _, (organ, color) in CLASS_INFO.items():
        subset = records[(records["organ"] == organ) & records["present"]]
        columns = ["retention_s{:02d}".format(value) for value in STRIDES]
        values = 100.0 * subset[columns].to_numpy(dtype=float)
        median = np.nanmedian(values, axis=0)
        q25 = np.nanpercentile(values, 25, axis=0)
        q75 = np.nanpercentile(values, 75, axis=0)
        axis.fill_between(STRIDES, q25, q75, color=color, alpha=0.13, linewidth=0)
        axis.plot(
            STRIDES,
            median,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=organ,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(STRIDES)
    axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axis.set_xlim(0, 38)
    axis.set_ylim(0, 105)
    axis.set_xlabel("Grid stride (pixels)")
    axis.set_ylabel("Centerline retained (%)")
    axis.set_title("Coarse-grid topology retention", loc="center", pad=8)
    axis.grid(color="#D9DDE1", linewidth=0.55, alpha=0.85)
    axis.legend(frameon=False, loc="lower left", ncol=3, handlelength=0.5)
    axis.set_axisbelow(True)
    add_panel_label(axis, "(c)")


def plot_figure(records, output_pdf, output_png):
    configure_style()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(8.6, 3.0),
        gridspec_kw={"width_ratios": [1.0, 1.18, 1.05], "wspace": 0.31},
    )
    plot_occupancy(axes[0], records)
    plot_survival(axes[1], records)
    plot_retention(axes[2], records)
    figure.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.19)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(output_png, dpi=450, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def derive_findings(records):
    findings = {"normalized_size": TARGET_SIZE, "classes": {}}
    for _, (organ, _) in CLASS_INFO.items():
        all_rows = records[records["organ"] == organ]
        present = all_rows[all_rows["present"]]
        entry = {
            "median_area_pct_present_images": float(present["area_pct"].median()),
            "absent_image_pct": float(100.0 * (1.0 - len(present) / len(all_rows))),
            "median_image_diameter_px": float(present["median_diameter_px"].median()),
            "median_centerline_below_4px_pct": float(
                100.0 * np.nanmedian(1.0 - present["survival_04"])
            ),
            "median_centerline_below_8px_pct": float(
                100.0 * np.nanmedian(1.0 - present["survival_08"])
            ),
        }
        for stride in STRIDES:
            entry["median_retention_stride_{}".format(stride)] = float(
                100.0 * present["retention_s{:02d}".format(stride)].median()
            )
        findings["classes"][organ] = entry
    return findings


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "per_image_organ_geometry.csv"

    if args.reuse_manifest and manifest_path.is_file():
        records = pd.read_csv(manifest_path)
    else:
        pairs = find_pairs(args.dataset_root.resolve())
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                nested_rows = list(executor.map(process_pair, pairs, chunksize=8))
        else:
            nested_rows = [process_pair(pair) for pair in pairs]
        records = pd.DataFrame([row for rows in nested_rows for row in rows])
        records.to_csv(manifest_path, index=False)

    expected_rows = 1096 * len(CLASS_INFO)
    if len(records) != expected_rows:
        raise RuntimeError("Expected {} organ records, found {}".format(expected_rows, len(records)))

    output_pdf = args.figure_dir / "organ_scale_burden.pdf"
    output_png = args.figure_dir / "organ_scale_burden.png"
    plot_figure(records, output_pdf, output_png)

    findings = derive_findings(records)
    findings_path = args.output_dir / "findings.json"
    findings_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")

    summary_rows = []
    for organ, values in findings["classes"].items():
        summary_rows.append({"organ": organ, **values})
    summary_path = args.output_dir / "organ_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("Processed {} images into {} organ records".format(1096, len(records)))
    print("Figure PDF: {}".format(output_pdf))
    print("Figure PNG: {}".format(output_png))
    print("Per-image geometry: {}".format(manifest_path))
    print("Organ summary: {}".format(summary_path))
    print("Findings: {}".format(findings_path))


if __name__ == "__main__":
    main()
