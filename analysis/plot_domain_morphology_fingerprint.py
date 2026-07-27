#!/usr/bin/env python3
"""Build a cross-domain appearance and organ-morphology fingerprint for GWFSS."""

import argparse
import json
import math
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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist, pdist
from skimage.morphology import dilation, disk, erosion, skeletonize


CLASS_COLORS = {
    1: np.array([50, 255, 132], dtype=np.float32),
    2: np.array([50, 132, 255], dtype=np.float32),
    3: np.array([214, 255, 50], dtype=np.float32),
}
RGB_TO_CLASS = {
    (0, 0, 0): 0,
    (50, 255, 132): 1,
    (50, 132, 255): 2,
    (214, 255, 50): 3,
}

ROLE_BY_DOMAIN = {
    "Arvalis": "Source train",
    "CIMMYT": "Source train",
    "ETHZ": "Source train",
    "INRAE": "Source train",
    "NJAU": "Source train",
    "RRES": "Source train",
    "ULiege_CRA-W": "Source train",
    "UTokyo": "Validation",
    "UQ_new": "Target test",
    "USASK": "Unused",
}
ROLE_COLORS = {
    "Source train": "#168C83",
    "Validation": "#E0A11A",
    "Target test": "#C74440",
    "Unused": "#8A9099",
}

FEATURES = [
    ("luminance_mean", "Mean L", "Appearance"),
    ("luminance_std", "L\ncontrast", "Appearance"),
    ("green_excess", "Excess\ngreen", "Appearance"),
    ("chroma_variation", "Chroma\nvar.", "Appearance"),
    ("background_pct", "Background", "Composition"),
    ("head_pct", "Head", "Composition"),
    ("stem_pct", "Stem", "Composition"),
    ("leaf_pct", "Leaf", "Composition"),
    ("stem_width_px", "Stem\nwidth", "Topology"),
    ("skeleton_density", "Skeleton\ndens.", "Topology"),
    ("component_density", "Component\ndens.", "Topology"),
    ("endpoint_density", "Endpoint\ndens.", "Topology"),
    ("leaf_contact_pct", "Leaf--stem\ncontact", "Topology"),
]


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "GWFSS" / "GWFSS_v1.0_labelled",
        help="Directory containing images/ and either class_id/ or masks/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "analysis_outputs" / "domain_fingerprint",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=repo_root / "Paper_Template" / "figures",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--reuse-manifest",
        action="store_true",
        help="Reuse the existing per-image CSV instead of recomputing masks.",
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


def process_pair(item):
    domain, image_path, mask_path = item
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    labels = load_class_mask(mask_path)
    if rgb.shape[:2] != labels.shape:
        raise ValueError("Image-mask size mismatch for {}".format(image_path))

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    channels = rgb.astype(np.float32) / 255.0
    luminance = lab[..., 0] / 255.0
    green_excess = 2.0 * channels[..., 1] - channels[..., 0] - channels[..., 2]
    chroma_variation = math.sqrt(
        float(lab[..., 1].std()) ** 2 + float(lab[..., 2].std()) ** 2
    )

    total_pixels = labels.size
    class_pct = [100.0 * np.count_nonzero(labels == i) / total_pixels for i in range(4)]
    stem = labels == 2
    leaf = labels == 3

    if stem.any():
        skeleton = skeletonize(stem)
        skeleton_length = int(skeleton.sum())
        distance = ndimage.distance_transform_edt(stem)
        # Twice the distance-to-background on the medial axis estimates local diameter.
        stem_width = (
            2.0 * float(np.median(distance[skeleton]))
            if skeleton_length
            else float("nan")
        )

        component_labels, component_count = ndimage.label(
            stem,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        component_sizes = np.bincount(component_labels.ravel())[1:]
        component_count = int(np.count_nonzero(component_sizes >= 4))

        neighbour_count = ndimage.convolve(
            skeleton.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            mode="constant",
            cval=0,
        ) - skeleton.astype(np.uint8)
        endpoint_count = int(np.count_nonzero(skeleton & (neighbour_count <= 1)))

        stem_boundary = stem & ~erosion(stem, footprint=disk(1))
        leaf_near = dilation(leaf, footprint=disk(2))
        boundary_pixels = int(stem_boundary.sum())
        leaf_contact = (
            100.0 * float(np.count_nonzero(stem_boundary & leaf_near)) / boundary_pixels
            if boundary_pixels
            else 0.0
        )
        skeleton_density = 1000.0 * skeleton_length / total_pixels
        component_density = 1000.0 * component_count / max(int(stem.sum()), 1)
        endpoint_density = 1000.0 * endpoint_count / max(skeleton_length, 1)
    else:
        stem_width = float("nan")
        skeleton_length = 0
        component_count = 0
        endpoint_count = 0
        leaf_contact = float("nan")
        skeleton_density = 0.0
        component_density = float("nan")
        endpoint_density = float("nan")

    return {
        "domain": domain,
        "role": ROLE_BY_DOMAIN.get(domain, "Unknown"),
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "image_width": int(labels.shape[1]),
        "image_height": int(labels.shape[0]),
        "luminance_mean": float(luminance.mean()),
        "luminance_std": float(luminance.std()),
        "green_excess": float(green_excess.mean()),
        "chroma_variation": chroma_variation,
        "background_pct": class_pct[0],
        "head_pct": class_pct[1],
        "stem_pct": class_pct[2],
        "leaf_pct": class_pct[3],
        "stem_width_px": stem_width,
        "skeleton_density": skeleton_density,
        "component_density": component_density,
        "endpoint_density": endpoint_density,
        "leaf_contact_pct": leaf_contact,
        "skeleton_length": skeleton_length,
        "component_count": component_count,
        "endpoint_count": endpoint_count,
    }


def robust_zscore(frame):
    values = frame.to_numpy(dtype=float)
    center = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = q75 - q25
    fallback = np.nanstd(values, axis=0)
    scale = np.where(scale > 1e-9, scale, fallback)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (values - center) / scale, center, scale


def select_representatives(records, domain_z, ordered_domains):
    mandatory = [name for name in ("UTokyo", "UQ_new", "USASK") if name in ordered_domains]
    sources = [
        name for name in ordered_domains
        if ROLE_BY_DOMAIN.get(name) == "Source train"
    ]
    selected = list(mandatory)
    while len(selected) < 5 and sources:
        candidates = [name for name in sources if name not in selected]
        if not candidates:
            break
        if selected:
            scores = {
                name: min(
                    float(np.linalg.norm(domain_z[name] - domain_z[chosen]))
                    for chosen in selected
                )
                for name in candidates
            }
        else:
            scores = {name: float(np.linalg.norm(domain_z[name])) for name in candidates}
        selected.append(max(scores, key=scores.get))

    feature_columns = [item[0] for item in FEATURES]
    sample_values = records[feature_columns].copy()
    sample_values = sample_values.fillna(records.groupby("domain")[feature_columns].transform("median"))
    sample_values = sample_values.fillna(sample_values.median())
    sample_z, _, _ = robust_zscore(sample_values)
    records = records.copy()
    records["_feature_index"] = np.arange(len(records))

    representative_rows = []
    for domain in selected:
        domain_rows = records[records["domain"] == domain]
        eligible_rows = domain_rows[
            domain_rows["stem_width_px"].notna()
            & domain_rows["endpoint_density"].notna()
            & (domain_rows["skeleton_length"] >= 40)
        ]
        if not eligible_rows.empty:
            domain_rows = eligible_rows
        indices = domain_rows["_feature_index"].to_numpy(dtype=int)
        target = np.nanmedian(sample_z[indices], axis=0)
        distances = np.linalg.norm(sample_z[indices] - target, axis=1)
        representative_rows.append(domain_rows.iloc[int(np.argmin(distances))])
    return representative_rows


def color_overlay(rgb, labels):
    rgb = rgb.astype(np.float32)
    overlay = rgb.copy()
    for class_id, color in CLASS_COLORS.items():
        selected = labels == class_id
        overlay[selected] = 0.58 * overlay[selected] + 0.42 * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def measurement_crop(image_path, mask_path, crop_size=256):
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    labels = load_class_mask(mask_path)
    stem = labels == 2
    skeleton = skeletonize(stem)
    distance = ndimage.distance_transform_edt(stem)

    neighbour_count = ndimage.convolve(
        skeleton.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    ) - skeleton.astype(np.uint8)
    candidates = np.column_stack(np.nonzero(skeleton & (neighbour_count == 2)))
    if not len(candidates):
        candidates = np.column_stack(np.nonzero(skeleton))
    if not len(candidates):
        raise ValueError("Representative sample contains no stem skeleton: {}".format(mask_path))

    local_widths = 2.0 * distance[candidates[:, 0], candidates[:, 1]]
    median_width = 2.0 * float(np.median(distance[skeleton]))
    errors = np.abs(local_widths - median_width)
    near_median = candidates[errors <= errors.min() + 0.25]
    height, width = labels.shape
    edge_clearance = np.minimum.reduce(
        [
            near_median[:, 0],
            near_median[:, 1],
            height - 1 - near_median[:, 0],
            width - 1 - near_median[:, 1],
        ]
    )
    center_y, center_x = near_median[int(np.argmax(edge_clearance))]

    y0 = int(np.clip(center_y - crop_size // 2, 0, max(height - crop_size, 0)))
    x0 = int(np.clip(center_x - crop_size // 2, 0, max(width - crop_size, 0)))
    y1 = min(y0 + crop_size, height)
    x1 = min(x0 + crop_size, width)
    overlay = color_overlay(rgb[y0:y1, x0:x1], labels[y0:y1, x0:x1])

    skeleton_crop = skeleton[y0:y1, x0:x1]
    skeleton_outline = dilation(skeleton_crop, footprint=disk(1))
    overlay[skeleton_outline] = np.array([32, 36, 40], dtype=np.uint8)
    overlay[skeleton_crop] = np.array([255, 255, 255], dtype=np.uint8)

    return {
        "image": overlay,
        "circle_center": (float(center_x - x0), float(center_y - y0)),
        "circle_radius": float(distance[center_y, center_x]),
        "native_size": (width, height),
    }


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_figure(records, domain_summary, output_pdf, output_png):
    feature_columns = [item[0] for item in FEATURES]
    matrix_raw = domain_summary[feature_columns]
    matrix_z, _, _ = robust_zscore(matrix_raw)
    hierarchy = linkage(matrix_z, method="average", metric="euclidean", optimal_ordering=True)
    leaves = dendrogram(hierarchy, no_plot=True)["leaves"]
    ordered_domains = [domain_summary.index[index] for index in leaves]
    ordered_matrix = matrix_z[leaves]
    domain_z = {
        domain_summary.index[index]: matrix_z[index]
        for index in range(len(domain_summary))
    }
    representatives = select_representatives(records, domain_z, ordered_domains)

    configure_style()
    figure = plt.figure(figsize=(8.6, 5.55), constrained_layout=False)
    outer = figure.add_gridspec(
        2,
        1,
        height_ratios=[2.25, 1.15],
        hspace=0.62,
        left=0.055,
        right=0.97,
        top=0.925,
        bottom=0.055,
    )
    top = outer[0].subgridspec(
        1,
        5,
        width_ratios=[1.05, 0.52, 0.13, 6.7, 0.24],
        wspace=0.035,
    )
    ax_dendro = figure.add_subplot(top[0, 0])
    ax_names = figure.add_subplot(top[0, 1])
    ax_role = figure.add_subplot(top[0, 2])
    ax_heat = figure.add_subplot(top[0, 3])
    ax_color = figure.add_subplot(top[0, 4])

    dendrogram(
        hierarchy,
        orientation="left",
        labels=domain_summary.index.tolist(),
        ax=ax_dendro,
        color_threshold=0,
        above_threshold_color="#343A40",
        link_color_func=lambda _: "#343A40",
    )
    ax_dendro.invert_yaxis()
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)
    ax_dendro.set_title("Domain similarity", loc="left", pad=10, fontweight="bold")

    ax_names.set_xlim(0, 1)
    ax_names.set_ylim(len(ordered_domains) - 0.5, -0.5)
    for row_index, name in enumerate(ordered_domains):
        display_name = name.replace("ULiege_CRA-W", "ULiege").replace("UQ_new", "UQ")
        ax_names.text(1.0, row_index, display_name, ha="right", va="center")
    ax_names.set_xticks([])
    ax_names.set_yticks([])
    for spine in ax_names.spines.values():
        spine.set_visible(False)

    role_rgba = np.array(
        [mpl.colors.to_rgba(ROLE_COLORS[ROLE_BY_DOMAIN[name]]) for name in ordered_domains]
    ).reshape(len(ordered_domains), 1, 4)
    ax_role.imshow(role_rgba, aspect="auto", interpolation="nearest")
    ax_role.set_xticks([])
    ax_role.set_yticks([])
    for spine in ax_role.spines.values():
        spine.set_visible(False)

    norm = mpl.colors.TwoSlopeNorm(vmin=-2.4, vcenter=0.0, vmax=2.4)
    image = ax_heat.imshow(
        ordered_matrix,
        cmap="RdBu_r",
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )
    ax_heat.set_yticks([])
    ax_heat.set_xticks(np.arange(len(FEATURES)))
    ax_heat.set_xticklabels(
        [item[1] for item in FEATURES],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax_heat.tick_params(axis="x", pad=4, length=0)
    ax_heat.set_xticks(np.arange(-0.5, len(FEATURES), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(ordered_domains), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=0.65)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    ax_heat.set_title(
        "Robust domain fingerprint",
        loc="center",
        pad=10,
        fontweight="bold",
    )

    group_colors = {
        "Appearance": "#E0A11A",
        "Composition": "#168C83",
        "Topology": "#C74440",
    }
    groups = [item[2] for item in FEATURES]
    start = 0
    for index in range(1, len(groups) + 1):
        if index == len(groups) or groups[index] != groups[start]:
            end = index - 1
            ax_heat.plot(
                [start - 0.43, end + 0.43],
                [-0.83, -0.83],
                color=group_colors[groups[start]],
                linewidth=4.0,
                solid_capstyle="butt",
                clip_on=False,
            )
            ax_heat.text(
                (start + end) / 2,
                -1.33,
                groups[start],
                ha="center",
                va="bottom",
                color=group_colors[groups[start]],
                fontweight="bold",
                clip_on=False,
            )
            start = index

    colorbar = figure.colorbar(image, cax=ax_color)
    colorbar.set_label("Relative domain level (robust z-score)", rotation=90, labelpad=8)
    colorbar.outline.set_linewidth(0.6)

    legend_handles = [
        mpl.patches.Patch(color=color, label=role)
        for role, color in ROLE_COLORS.items()
    ]
    ax_heat.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.12),
        ncol=4,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.25,
        borderaxespad=0.0,
    )
    figure.text(0.018, 0.965, "(a)", fontsize=15, fontweight="bold", va="top")

    bottom = outer[1].subgridspec(1, 5, wspace=0.12)
    for index, row in enumerate(representatives):
        axis = figure.add_subplot(bottom[0, index])
        crop = measurement_crop(row["image_path"], row["mask_path"])
        axis.imshow(crop["image"])
        axis.add_patch(
            mpl.patches.Circle(
                crop["circle_center"],
                crop["circle_radius"],
                fill=False,
                edgecolor="#202428",
                linewidth=2.4,
            )
        )
        axis.add_patch(
            mpl.patches.Circle(
                crop["circle_center"],
                crop["circle_radius"],
                fill=False,
                edgecolor="white",
                linewidth=1.2,
            )
        )
        axis.plot(
            crop["circle_center"][0],
            crop["circle_center"][1],
            marker="o",
            markersize=2.4,
            markerfacecolor="white",
            markeredgecolor="#202428",
            markeredgewidth=0.5,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        role = row["role"]
        for spine in axis.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color(ROLE_COLORS[role])
        display_domain = row["domain"].replace("ULiege_CRA-W", "ULiege").replace("UQ_new", "UQ")
        axis.set_title(
            "{}  |  {}".format(display_domain, role),
            color=ROLE_COLORS[role],
            fontweight="bold",
            pad=5,
        )
        width = row["stem_width_px"]
        axis.set_xlabel(
            "Diameter: {:.1f} px\nImage: {} x {} px".format(
                width, crop["native_size"][0], crop["native_size"][1]
            ),
            fontsize=9.0,
            labelpad=3,
            linespacing=1.25,
        )
    figure.text(0.018, 0.345, "(b)", fontsize=15, fontweight="bold", va="top")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(output_png, dpi=450, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return matrix_z, ordered_domains, representatives


def derive_findings(domain_summary, matrix_z):
    feature_columns = [item[0] for item in FEATURES]
    domain_names = domain_summary.index.tolist()
    source_indices = [
        i for i, name in enumerate(domain_names)
        if ROLE_BY_DOMAIN.get(name) == "Source train"
    ]
    source_centroid = matrix_z[source_indices].mean(axis=0)
    source_dispersion = float(np.median(pdist(matrix_z[source_indices])))

    findings = {
        "source_pairwise_median_distance": source_dispersion,
        "domain_to_source_centroid": {},
        "feature_extremes": {},
    }
    for name in domain_names:
        index = domain_names.index(name)
        findings["domain_to_source_centroid"][name] = float(
            np.linalg.norm(matrix_z[index] - source_centroid)
        )
    for feature in ("stem_width_px", "endpoint_density", "leaf_contact_pct"):
        series = domain_summary[feature]
        findings["feature_extremes"][feature] = {
            "minimum_domain": str(series.idxmin()),
            "minimum": float(series.min()),
            "maximum_domain": str(series.idxmax()),
            "maximum": float(series.max()),
        }

    target_name = "UQ_new"
    if target_name in domain_names:
        target_index = domain_names.index(target_name)
        difference = matrix_z[target_index] - source_centroid
        ranked = np.argsort(np.abs(difference))[::-1][:4]
        findings["target_largest_shifts"] = [
            {
                "feature": feature_columns[index],
                "signed_robust_z": float(difference[index]),
            }
            for index in ranked
        ]
    return findings


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "per_image_domain_metrics.csv"

    if args.reuse_manifest and manifest_path.is_file():
        records = pd.read_csv(manifest_path)
    else:
        pairs = find_pairs(args.dataset_root.resolve())
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                rows = list(executor.map(process_pair, pairs, chunksize=8))
        else:
            rows = [process_pair(pair) for pair in pairs]
        records = pd.DataFrame(rows)
        records.to_csv(manifest_path, index=False)

    unknown = sorted(set(records["domain"]) - set(ROLE_BY_DOMAIN))
    if unknown:
        raise RuntimeError("Missing split roles for domains: {}".format(unknown))

    feature_columns = [item[0] for item in FEATURES]
    domain_summary = records.groupby("domain", sort=True)[feature_columns].median()
    domain_summary.insert(0, "sample_count", records.groupby("domain").size())
    domain_summary.insert(1, "role", [ROLE_BY_DOMAIN[name] for name in domain_summary.index])
    summary_path = args.output_dir / "domain_summary.csv"
    domain_summary.to_csv(summary_path)

    output_pdf = args.figure_dir / "domain_morphology_fingerprint.pdf"
    output_png = args.figure_dir / "domain_morphology_fingerprint.png"
    matrix_z, ordered_domains, representatives = plot_figure(
        records, domain_summary, output_pdf, output_png
    )
    representative_path = args.output_dir / "representative_samples.csv"
    pd.DataFrame(representatives)[
        [
            "domain",
            "role",
            "image_path",
            "mask_path",
            "stem_width_px",
            "endpoint_density",
            "leaf_contact_pct",
        ]
    ].to_csv(representative_path, index=False)
    findings = derive_findings(domain_summary, matrix_z)
    findings["ordered_domains"] = ordered_domains
    findings_path = args.output_dir / "findings.json"
    findings_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")

    print("Processed {} images across {} labelled domains".format(len(records), len(domain_summary)))
    print("Figure PDF: {}".format(output_pdf))
    print("Figure PNG: {}".format(output_png))
    print("Per-image metrics: {}".format(manifest_path))
    print("Domain summary: {}".format(summary_path))
    print("Representative samples: {}".format(representative_path))
    print("Findings: {}".format(findings_path))


if __name__ == "__main__":
    main()
