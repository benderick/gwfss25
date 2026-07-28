#!/usr/bin/env python3
"""Draw the data-agnostic three-stage TopoWheat method overview."""

import csv
import hashlib
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "topowheat-matplotlib"),
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from PIL import Image
from scipy import ndimage as ndi


INK = "#203139"
MUTED = "#66757C"
LINE = "#CBD5D9"
WHITE = "#FFFFFF"

STAGE1 = "#607C8D"
STAGE1_BG = "#EEF3F6"
STAGE2 = "#138B80"
STAGE2_BG = "#EDF8F5"
STAGE3 = "#D28A0A"
STAGE3_BG = "#FFF6E7"

MODEL = "#3378B9"
TEACHER = "#71879A"
TRPL = "#CE514B"
TRPL_BG = "#FFF4F2"
TCPM = "#158B80"
TCPM_BG = "#F2FBF8"

BACKGROUND = "#27353B"
HEAD = "#28A46D"
STEM = "#3378B9"
LEAF = "#A6B51E"

CLASS_COLORS = {
    0: np.array([39, 53, 59], dtype=np.float32),
    1: np.array([40, 164, 109], dtype=np.float32),
    2: np.array([51, 120, 185], dtype=np.float32),
    3: np.array([166, 181, 30], dtype=np.float32),
}

ASSET_SPECS = (
    {
        "role": "supervised_pair",
        "split": "competition_train",
        "path": "GWFSS/gwfss_competition_train/images/domain8_00003.png",
        "annotation": "GWFSS/gwfss_competition_train/class_id/domain8_00003.png",
        "usage": "Aligned real RGB and annotation used for the supervised-pair example",
    },
    {
        "role": "unlabeled_views",
        "split": "competition_pretrain",
        "path": "GWFSS/gwfss_competition_pretrain/RRES/RRES_!_05630.jpg",
        "annotation": "",
        "usage": "One real unlabeled image rendered with deterministic weak and strong color views",
    },
    {
        "role": "inference_flow",
        "split": "competition_val",
        "path": "GWFSS/gwfss_competition_val/images/domain7_00021.png",
        "annotation": "GWFSS/gwfss_competition_val/class_id/domain7_00021.png",
        "usage": "One aligned real RGB/annotation pair used for deterministic mechanism maps, core sampling, windows, crops, and output geometry",
    },
)


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 8.0,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rounded_box(ax, x, y, width, height, face=WHITE, edge="none", radius=0.010, lw=0.8, zorder=2):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.0025,rounding_size={}".format(radius),
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=INK, lw=1.1, dashed=False, connection="arc3,rad=0", zorder=8):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=lw,
        linestyle=(0, (3, 2)) if dashed else "solid",
        color=color,
        connectionstyle=connection,
        shrinkA=1.0,
        shrinkB=1.0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def orthogonal_arrow(ax, points, color=INK, lw=1.1, dashed=False, zorder=8):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=lw,
            linestyle=(0, (3, 2)) if dashed else "solid",
            solid_capstyle="round",
            zorder=zorder,
        )
    return arrow(ax, points[-2], points[-1], color=color, lw=lw, dashed=dashed, zorder=zorder)


def square_crop(array):
    height, width = array.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return array[y0 : y0 + side, x0 : x0 + side]


def load_rgb(path):
    return square_crop(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))


def load_mask(path):
    raw = square_crop(np.asarray(Image.open(path)))
    if raw.ndim == 2:
        return raw.astype(np.uint8)
    labels = np.zeros(raw.shape[:2], dtype=np.uint8)
    rgb = raw[..., :3]
    for color, class_id in {
        (50, 255, 132): 1,
        (50, 132, 255): 2,
        (214, 255, 50): 3,
    }.items():
        labels[np.all(rgb == color, axis=-1)] = class_id
    return labels


def overlay_labels(rgb, labels, alpha=0.56):
    result = rgb.astype(np.float32).copy()
    for class_id in (1, 2, 3):
        selected = labels == class_id
        result[selected] = (1.0 - alpha) * result[selected] + alpha * CLASS_COLORS[class_id]
    return np.clip(result, 0, 255).astype(np.uint8)


def color_mask(labels):
    result = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        result[labels == class_id] = color.astype(np.uint8)
    return result


def normalize_map(values):
    values = np.asarray(values, dtype=np.float32)
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    if upper - lower < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0)


def heatmap_rgb(values, cmap="magma", background=None, alpha=1.0):
    normalized = normalize_map(values)
    mapped = mpl.colormaps[cmap](normalized)[..., :3]
    if background is not None:
        base = np.asarray(background, dtype=np.float32) / 255.0
        local_alpha = (alpha * normalized)[..., None]
        mapped = (1.0 - local_alpha) * base + local_alpha * mapped
    return np.clip(mapped * 255.0, 0, 255).astype(np.uint8)


def binary_skeleton(mask):
    """Return a deterministic morphological skeleton without an extra dependency."""
    current = np.asarray(mask, dtype=bool)
    skeleton = np.zeros_like(current)
    structure = ndi.generate_binary_structure(2, 1)
    while current.any():
        opened = ndi.binary_dilation(ndi.binary_erosion(current, structure=structure), structure=structure)
        skeleton |= current & ~opened
        current = ndi.binary_erosion(current, structure=structure)
    return skeleton


def mechanism_maps(labels):
    """Construct aligned schematic evidence maps from a real annotated geometry."""
    scale = max(min(labels.shape[:2]) / 192.0, 0.5)

    def scaled_pixels(value):
        return max(1, int(round(value * scale)))

    stem = np.asarray(labels == 2, dtype=np.float32)
    base = ndi.gaussian_filter(stem, sigma=1.3 * scale)
    base = normalize_map(base)
    variants = []
    for shift, sigma, gain in (((-2, 1), 1.0, 0.94), ((1, -1), 1.8, 1.00), ((2, 2), 2.6, 0.90)):
        scaled_shift = tuple(int(round(value * scale)) for value in shift)
        shifted = ndi.shift(stem, shift=scaled_shift, order=0, mode="constant", cval=0.0)
        variants.append(np.clip(gain * ndi.gaussian_filter(shifted, sigma=sigma * scale), 0.0, 1.0))
    stack = np.stack(variants, axis=0)
    mean = stack.mean(axis=0)
    eps = 1e-6
    entropy = -(mean * np.log(mean + eps) + (1.0 - mean) * np.log(1.0 - mean + eps)) / np.log(2.0)
    disagreement = stack.std(axis=0)
    uncertainty = normalize_map(entropy + 1.35 * disagreement)

    skeletons = np.stack([binary_skeleton(item > 0.33) for item in variants], axis=0)
    persistence = skeletons.mean(axis=0)
    reliable_region = (mean > 0.34) & (uncertainty < 0.68)
    stable_core = persistence >= (2.0 / 3.0)
    candidate = mean > 0.12
    uncertain_boundary = candidate & ~reliable_region & ~stable_core

    targets = np.zeros((*mean.shape, 3), dtype=np.uint8)
    targets[:] = np.array([245, 247, 247], dtype=np.uint8)
    targets[uncertain_boundary] = np.array([172, 181, 185], dtype=np.uint8)
    targets[reliable_region] = np.array([21, 139, 128], dtype=np.uint8)
    targets[stable_core] = np.array([51, 120, 185], dtype=np.uint8)

    boundary = ndi.binary_dilation(stem > 0, iterations=scaled_pixels(4)) ^ ndi.binary_erosion(stem > 0, iterations=scaled_pixels(2))
    scale_disagreement = np.abs(ndi.gaussian_filter(stem, sigma=0.8 * scale) - ndi.gaussian_filter(stem, sigma=4.0 * scale))
    skeleton = binary_skeleton(stem > 0)
    neighbors = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - skeleton
    endpoints = skeleton & (neighbors == 1)
    endpoint_density = ndi.gaussian_filter(endpoints.astype(np.float32), sigma=7.0 * scale)
    stem_prior = ndi.gaussian_filter(stem, sigma=2.0 * scale)
    boundary_uncertainty = ndi.gaussian_filter(boundary.astype(np.float32), sigma=3.0 * scale)
    risk = normalize_map(
        0.30 * normalize_map(boundary_uncertainty)
        + 0.28 * normalize_map(endpoint_density)
        + 0.27 * normalize_map(scale_disagreement)
        + 0.15 * normalize_map(stem_prior)
    )
    return {
        "variants": variants,
        "mean": mean,
        "uncertainty": uncertainty,
        "persistence": persistence,
        "targets": targets,
        "region": reliable_region,
        "core": stable_core,
        "boundary": uncertain_boundary,
        "routing_uncertainty": boundary_uncertainty,
        "endpoints": endpoint_density,
        "disagreement": scale_disagreement,
        "stem_prior": stem_prior,
        "risk": risk,
    }


def core_sampling_visual(labels):
    labels = np.asarray(labels, dtype=np.uint8)
    canvas = np.full((*labels.shape, 3), 244, dtype=np.float32)
    for class_id in (0, 1, 2, 3):
        selected = labels == class_id
        tint = CLASS_COLORS[class_id]
        canvas[selected] = 0.78 * canvas[selected] + 0.22 * tint
        if class_id == 2:
            core = binary_skeleton(selected)
            core = ndi.binary_dilation(core, iterations=1)
        else:
            core = ndi.binary_erosion(selected, iterations=3)
        canvas[core] = tint
    return np.clip(canvas, 0, 255).astype(np.uint8)


def phase_label(ax, x, y, text, color, width):
    ax.text(x, y, text, ha="left", va="center", color=color, fontsize=5.3, fontweight="bold", zorder=9)
    ax.plot([x, x + width], [y - 0.012, y - 0.012], color=color, linewidth=0.55, alpha=0.55, zorder=4)


def map_caption(ax, x, y, width, text, color=INK, size=5.2):
    ax.text(x + width / 2, y, text, ha="center", va="center", color=color, fontsize=size, zorder=9)


def strong_color_view(rgb):
    data = rgb.astype(np.float32) / 255.0
    data = np.power(data, 0.78)
    data *= np.array([0.82, 1.08, 0.90], dtype=np.float32)
    luminance = data.mean(axis=2, keepdims=True)
    data = luminance + 1.22 * (data - luminance)
    return np.clip(data * 255.0, 0, 255).astype(np.uint8)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_and_load_assets(repo_root):
    resolved = {}
    rows = []
    for spec in ASSET_SPECS:
        path = repo_root / spec["path"]
        annotation = repo_root / spec["annotation"] if spec["annotation"] else None
        if not path.is_file():
            raise FileNotFoundError(path)
        if annotation is not None and not annotation.is_file():
            raise FileNotFoundError(annotation)
        resolved[spec["role"]] = (path, annotation)
        rows.append({**spec, "exists": True, "sha256": sha256(path)})

    train_rgb = load_rgb(resolved["supervised_pair"][0])
    train_labels = load_mask(resolved["supervised_pair"][1])
    unlabeled_rgb = load_rgb(resolved["unlabeled_views"][0])
    inference_rgb = load_rgb(resolved["inference_flow"][0])
    inference_labels = load_mask(resolved["inference_flow"][1])
    return train_rgb, train_labels, unlabeled_rgb, inference_rgb, inference_labels, rows


def write_manifest(repo_root, rows):
    output = repo_root / "analysis_outputs" / "framework" / "figure_assets_manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["role", "split", "path", "annotation", "usage", "exists", "sha256"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def image_tile(ax, image, x, y, width, height, edge=WHITE, lw=1.0, radius=0.008, zorder=5):
    clip = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0,rounding_size={}".format(radius),
        facecolor="none",
        edgecolor="none",
        zorder=zorder,
    )
    ax.add_patch(clip)
    artist = ax.imshow(
        image,
        extent=(x, x + width, y, y + height),
        interpolation="bilinear",
        aspect="auto",
        zorder=zorder,
    )
    artist.set_clip_path(clip)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0,rounding_size={}".format(radius),
            facecolor="none",
            edgecolor=edge,
            linewidth=lw,
            zorder=zorder + 1,
        )
    )


def stage_panel(ax, x, y, width, height, letter, title, color, background):
    del background
    title_offset = {"I": 0.053, "II": 0.060, "III": 0.071}.get(letter, 0.055)
    ax.plot([x, x + width], [y + height, y + height], color=color, linewidth=1.15, solid_capstyle="butt", zorder=2)
    ax.text(
        x,
        y + height - 0.027,
        "Stage {}".format(letter),
        ha="left",
        va="center",
        color=color,
        fontsize=7.0,
        fontweight="bold",
        zorder=6,
    )
    ax.text(x + title_offset, y + height - 0.027, title, ha="left", va="center", color=INK, fontsize=7.0, zorder=6)
    ax.plot([x, x + width], [y + height - 0.050, y + height - 0.050], color=LINE, linewidth=0.45, zorder=2)


def small_label(ax, x, y, text, color=MUTED, size=6.4, weight="normal", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", color=color, fontsize=size, fontweight=weight, zorder=9)


def loss_chip(ax, x, y, width, text, color):
    rounded_box(ax, x, y, width, 0.034, face=WHITE, edge=color, radius=0.014, lw=0.75, zorder=6)
    ax.text(x + width / 2, y + 0.017, text, ha="center", va="center", color=color, fontsize=6.4, zorder=7)


def draw_segmenter(ax, x, y, width, height, accent=MODEL, compact=False):
    rounded_box(ax, x, y, width, height, face=WHITE, edge="none", radius=0.010, zorder=4)
    title_size = 6.7 if compact else 7.2
    ax.text(
        x + width / 2,
        y + height - 0.025,
        "GLOBAL PASS" if compact else "SHARED SEGMENTER",
        ha="center",
        va="center",
        color=accent,
        fontsize=title_size,
        fontweight="bold",
        zorder=6,
    )
    layers = (
        ("BEiTv2-L", "#EAF1F8", MODEL),
        ("ViT-Adapter", "#E8F6F3", STAGE2),
        ("Mask2Former", "#FFF4DE", STAGE3),
    )
    gap = 0.029 if compact else 0.038
    block_h = 0.021 if compact else 0.029
    start = y + height - (0.057 if compact else 0.067)
    for index, (name, face, edge) in enumerate(layers):
        yy = start - index * gap
        rounded_box(ax, x + 0.011, yy, width - 0.022, block_h, face=face, edge=edge, radius=0.004, lw=0.55, zorder=5)
        if not compact:
            ax.text(x + width / 2, yy + block_h / 2, name, ha="center", va="center", color=INK, fontsize=6.3, zorder=6)


def draw_supervised_stage(ax, rgb, labels):
    x1, x2, y = 0.035, 0.112, 0.665
    image_tile(ax, rgb, x1, y, 0.066, 0.132, radius=0.008)
    image_tile(ax, overlay_labels(rgb, labels), x2, y, 0.066, 0.132, radius=0.008)
    small_label(ax, x1 + 0.033, y - 0.018, r"RGB  $x^l$", size=6.5)
    small_label(ax, x2 + 0.033, y - 0.018, r"mask  $y^l$", size=6.5)
    arrow(ax, (0.107, 0.635), (0.107, 0.610), color=STAGE1)

    draw_segmenter(ax, 0.043, 0.420, 0.132, 0.185, accent=STAGE1)
    arrow(ax, (0.107, 0.420), (0.107, 0.370), color=STAGE1)

    rounded_box(ax, 0.032, 0.190, 0.150, 0.172, face=WHITE, edge="none", radius=0.010, zorder=4)
    ax.text(0.107, 0.337, "SUPERVISED OBJECTIVE", ha="center", va="center", color=STAGE1, fontsize=6.8, fontweight="bold", zorder=6)
    loss_chip(ax, 0.041, 0.280, 0.038, r"$\mathcal{L}_{cls}$", STAGE1)
    loss_chip(ax, 0.086, 0.280, 0.041, r"$\mathcal{L}_{mask}$", STAGE1)
    loss_chip(ax, 0.134, 0.280, 0.039, r"$\mathcal{L}_{topo}$", STAGE1)
    ax.text(0.082, 0.297, "+", color=MUTED, fontsize=8.0, ha="center", va="center", zorder=7)
    ax.text(0.130, 0.297, "+", color=MUTED, fontsize=8.0, ha="center", va="center", zorder=7)
    ax.text(0.107, 0.235, r"$\mathcal{L}_{sup}$", color=INK, fontsize=8.0, ha="center", va="center", fontweight="bold", zorder=7)

    ax.add_patch(Circle((0.200, 0.606), 0.024, facecolor=WHITE, edgecolor=STAGE1, linewidth=1.2, zorder=9))
    ax.text(0.200, 0.610, r"$\theta_0$", ha="center", va="center", color=STAGE1, fontsize=10.5, fontstyle="italic", zorder=10)
    small_label(ax, 0.197, 0.568, "warm start", color=STAGE1, size=6.2)


def draw_teacher_student(ax, x, y, width, height):
    rounded_box(ax, x, y, width, height, face=WHITE, edge="none", radius=0.010, zorder=4)
    teacher_y = y + height - 0.067
    student_y = y + 0.058
    ax.text(x + 0.012, y + height - 0.025, "EMA TEACHER", ha="left", va="center", color=TEACHER, fontsize=7.2, fontweight="bold", zorder=7)
    ax.text(x + 0.012, y + 0.025, "STUDENT", ha="left", va="center", color=MODEL, fontsize=7.2, fontweight="bold", zorder=7)
    for yy, color in ((teacher_y, TEACHER), (student_y, MODEL)):
        ax.plot([x + 0.016, x + width - 0.015], [yy, yy], color=color, linewidth=5.2, solid_capstyle="round", zorder=5)
        for index in range(4):
            xx = x + 0.027 + index * (width - 0.054) / 3
            ax.add_patch(Circle((xx, yy), 0.0062, facecolor=WHITE, edgecolor=color, linewidth=0.7, zorder=6))
    arrow(ax, (x + width * 0.52, student_y + 0.014), (x + width * 0.52, teacher_y - 0.014), color=TEACHER, lw=0.9, dashed=True)
    ax.text(x + width * 0.70, y + height / 2, "EMA", ha="center", va="center", color=TEACHER, fontsize=6.3, fontweight="bold", zorder=7)
    ax.text(x + width / 2, y + height / 2 - 0.028, "shared encoder-decoder", ha="center", va="center", color=MUTED, fontsize=5.9, zorder=7)


def curved_band(ax, cx, cy, height, width, color, lw, alpha=1.0, gap=None, zorder=7):
    values = np.linspace(-1.0, 1.0, 90)
    xx = cx + width * (0.35 * values + 0.16 * values**3)
    yy = cy + height * values / 2
    if gap is None:
        ax.plot(xx, yy, color=color, linewidth=lw, alpha=alpha, solid_capstyle="round", zorder=zorder)
    else:
        for keep in (values < -gap, values > gap):
            ax.plot(xx[keep], yy[keep], color=color, linewidth=lw, alpha=alpha, solid_capstyle="round", zorder=zorder)


def mini_frame(ax, x, y, width, height, face="#FAFCFC"):
    return rounded_box(ax, x, y, width, height, face=face, edge=LINE, radius=0.006, lw=0.55, zorder=5)


def draw_trpl(ax, x, y, width, height):
    rounded_box(ax, x, y, width, height, face=TRPL_BG, edge="none", radius=0.010, zorder=4)
    ax.add_patch(Rectangle((x, y + height - 0.007), width, 0.007, facecolor=TRPL, edgecolor="none", zorder=5))
    ax.text(x + 0.012, y + height - 0.025, "TRPL", ha="left", va="center", color=TRPL, fontsize=8.7, fontweight="bold", zorder=7)
    ax.text(x + 0.012, y + height - 0.048, "topology-reliable targets", ha="left", va="center", color=INK, fontsize=6.2, zorder=7)

    frame_y = y + 0.046
    frame_w = 0.049
    frame_h = height - 0.108
    xs = (x + 0.012, x + 0.079, x + 0.146)
    for xx in xs:
        mini_frame(ax, xx, frame_y, frame_w, frame_h)

    for offset, alpha in ((-0.006, 0.30), (0.0, 0.62), (0.006, 0.30)):
        curved_band(ax, xs[0] + frame_w / 2 + offset, frame_y + frame_h / 2, frame_h * 0.72, frame_w * 0.55, MODEL, 4.2, alpha=alpha)
    curved_band(ax, xs[0] + frame_w / 2, frame_y + frame_h / 2, frame_h * 0.72, frame_w * 0.55, WHITE, 0.8, alpha=0.85)

    curved_band(ax, xs[1] + frame_w / 2, frame_y + frame_h / 2, frame_h * 0.72, frame_w * 0.58, "#B5BEC2", 9.0, alpha=0.34)
    curved_band(ax, xs[1] + frame_w / 2 - 0.004, frame_y + frame_h / 2, frame_h * 0.72, frame_w * 0.58, TRPL, 1.1, alpha=0.95)
    curved_band(ax, xs[1] + frame_w / 2 + 0.004, frame_y + frame_h / 2, frame_h * 0.72, frame_w * 0.58, TRPL, 1.1, alpha=0.95)
    curved_band(ax, xs[1] + frame_w / 2, frame_y + frame_h / 2, frame_h * 0.72, frame_w * 0.58, MODEL, 1.3)

    curved_band(ax, xs[2] + frame_w / 2, frame_y + frame_h / 2, frame_h * 0.72, frame_w * 0.58, "#BFC6C8", 9.0, alpha=0.42)
    curved_band(ax, xs[2] + frame_w / 2, frame_y + frame_h / 2, frame_h * 0.72, frame_w * 0.58, TCPM, 5.0, alpha=0.85, gap=0.15)
    curved_band(ax, xs[2] + frame_w / 2, frame_y + frame_h / 2, frame_h * 0.72, frame_w * 0.58, MODEL, 1.3, gap=0.15)

    arrow(ax, (xs[0] + frame_w + 0.004, frame_y + frame_h / 2), (xs[1] - 0.004, frame_y + frame_h / 2), color=TRPL, lw=0.8)
    arrow(ax, (xs[1] + frame_w + 0.004, frame_y + frame_h / 2), (xs[2] - 0.004, frame_y + frame_h / 2), color=TRPL, lw=0.8)
    for xx, label in zip(xs, ("views", "uncertainty", "targets")):
        small_label(ax, xx + frame_w / 2, y + 0.025, label, color=MUTED, size=5.8)


def draw_tcpm(ax, x, y, width, height):
    rounded_box(ax, x, y, width, height, face=TCPM_BG, edge="none", radius=0.010, zorder=4)
    ax.add_patch(Rectangle((x, y + height - 0.007), width, 0.007, facecolor=TCPM, edgecolor="none", zorder=5))
    ax.text(x + 0.012, y + height - 0.025, "TCPM", ha="left", va="center", color=TCPM, fontsize=8.7, fontweight="bold", zorder=7)
    ax.text(x + 0.012, y + height - 0.048, "curriculum-stabilized organ anchors", ha="left", va="center", color=INK, fontsize=6.2, zorder=7)

    colors = (BACKGROUND, HEAD, STEM, LEAF)
    grid_x = x + 0.019
    for row in range(3):
        for column, color in enumerate(colors):
            ax.add_patch(Circle((grid_x + column * 0.014, y + 0.058 + row * 0.025), 0.0047, facecolor=color, edgecolor=WHITE, linewidth=0.35, zorder=7))

    proto_x = x + 0.096
    for row in range(3):
        yy = y + 0.058 + row * 0.025
        ax.add_patch(Circle((proto_x, yy), 0.011, facecolor=WHITE, edgecolor=TCPM, linewidth=0.75, zorder=6))
        for index, color in enumerate(colors):
            ax.add_patch(Circle((proto_x - 0.005 + index * 0.0033, yy), 0.0025, facecolor=color, edgecolor="none", zorder=7))

    anchor_x = x + width - 0.027
    for index, color in enumerate(colors):
        angle = np.pi / 2 + index * np.pi / 2
        ax.add_patch(Circle((anchor_x + 0.013 * np.cos(angle), y + 0.083 + 0.025 * np.sin(angle)), 0.0052, facecolor=color, edgecolor=WHITE, linewidth=0.4, zorder=7))
    ax.add_patch(Circle((anchor_x, y + 0.083), 0.013, facecolor="none", edgecolor=TCPM, linewidth=0.7, linestyle=(0, (2, 2)), zorder=6))
    ax.plot([anchor_x - 0.013, anchor_x + 0.013], [y + 0.083, y + 0.083], color=TRPL, linewidth=0.7, zorder=6)

    arrow(ax, (x + 0.078, y + 0.083), (proto_x - 0.014, y + 0.083), color=TCPM, lw=0.8)
    arrow(ax, (proto_x + 0.014, y + 0.083), (anchor_x - 0.018, y + 0.083), color=TCPM, lw=0.8)
    ax.text(x + 0.043, y + 0.025, "core\npixels", ha="center", va="center", color=MUTED, fontsize=5.3, linespacing=0.88, zorder=8)
    ax.text(proto_x, y + 0.025, "dual domain\nmemory", ha="center", va="center", color=MUTED, fontsize=5.3, linespacing=0.88, zorder=8)
    ax.text(anchor_x, y + 0.025, "class\nanchors", ha="center", va="center", color=MUTED, fontsize=5.3, linespacing=0.88, zorder=8)


def draw_joint_objective(ax, x, y, width, height):
    rounded_box(ax, x, y, width, height, face=WHITE, edge="none", radius=0.010, zorder=4)
    ax.text(x + 0.014, y + height - 0.024, "JOINT OPTIMIZATION", ha="left", va="center", color=STAGE2, fontsize=7.1, fontweight="bold", zorder=7)
    ax.text(x + width - 0.014, y + height - 0.024, "structure + representation", ha="right", va="center", color=MUTED, fontsize=6.0, fontstyle="italic", zorder=7)
    labels = (
        (r"$\mathcal{L}_{sup}$", STAGE1),
        (r"$\mathcal{L}_{region}$", TRPL),
        (r"$\mathcal{L}_{topology}$", TRPL),
        (r"$\mathcal{L}_{prototype}$", TCPM),
    )
    chip_w = 0.082
    start_x = x + 0.020
    for index, (label, color) in enumerate(labels):
        xx = start_x + index * 0.108
        loss_chip(ax, xx, y + 0.038, chip_w, label, color)
        if index < len(labels) - 1:
            ax.text(xx + chip_w + 0.013, y + 0.055, "+", ha="center", va="center", color=MUTED, fontsize=8.0, zorder=7)


def select_stem_windows(labels, grid=4, count=3):
    height, width = labels.shape
    candidates = []
    for row in range(grid):
        for column in range(grid):
            y0, y1 = row * height // grid, (row + 1) * height // grid
            x0, x1 = column * width // grid, (column + 1) * width // grid
            score = float((labels[y0:y1, x0:x1] == 2).mean())
            candidates.append((score, column, row))
    chosen = []
    for _, column, row in sorted(candidates, reverse=True):
        if all(abs(column - c) + abs(row - r) > 1 for c, r in chosen):
            chosen.append((column, row))
        if len(chosen) == count:
            break
    return chosen


def crop_from_grid(array, column, row, grid=4, size=192):
    height, width = array.shape[:2]
    cell_w, cell_h = width / grid, height / grid
    margin = 0.28
    x0 = int(max(0, (column - margin) * cell_w))
    x1 = int(min(width, (column + 1 + margin) * cell_w))
    y0 = int(max(0, (row - margin) * cell_h))
    y1 = int(min(height, (row + 1 + margin) * cell_h))
    crop = array[y0:y1, x0:x1]
    mode = Image.Resampling.NEAREST if array.ndim == 2 else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(crop).resize((size, size), mode))


def draw_window_image(ax, rgb, labels, x, y, width, height):
    image_tile(ax, rgb, x, y, width, height, edge=STAGE3, lw=0.8, radius=0.007, zorder=6)
    windows = select_stem_windows(labels)
    grid = 4
    for column, row in windows:
        ax.add_patch(
            Rectangle(
                (x + column * width / grid, y + (grid - 1 - row) * height / grid),
                width / grid,
                height / grid,
                fill=False,
                edgecolor=TRPL,
                linewidth=1.0,
                zorder=9,
            )
        )
    return windows


def draw_risk_cues(ax, x, y, width, height):
    rounded_box(ax, x, y, width, height, face=WHITE, edge="none", radius=0.009, zorder=4)
    ax.text(x + 0.011, y + height - 0.022, "BREAK-RISK FIELD", ha="left", va="center", color=STAGE3, fontsize=6.9, fontweight="bold", zorder=7)
    cue_centers = [x + 0.027 + index * 0.035 for index in range(4)]
    labels = (r"$H$", r"$E$", r"$\Delta$", r"$P_s$")
    for index, (cx, label) in enumerate(zip(cue_centers, labels)):
        cy = y + 0.048
        ax.add_patch(Circle((cx, cy), 0.013, facecolor=STAGE3_BG, edgecolor=STAGE3, linewidth=0.7, zorder=6))
        if index == 0:
            ax.add_patch(Circle((cx, cy), 0.008, facecolor="#F4C96C", edgecolor="none", alpha=0.65, zorder=7))
        elif index == 1:
            ax.plot([cx - 0.007, cx + 0.007], [cy, cy], color=STEM, linewidth=1.2, zorder=7)
            ax.add_patch(Circle((cx - 0.007, cy), 0.0028, facecolor=TRPL, edgecolor="none", zorder=8))
            ax.add_patch(Circle((cx + 0.007, cy), 0.0028, facecolor=TRPL, edgecolor="none", zorder=8))
        elif index == 2:
            ax.plot([cx - 0.007, cx + 0.004], [cy + 0.004, cy + 0.004], color=MODEL, linewidth=1.2, zorder=7)
            ax.plot([cx - 0.004, cx + 0.007], [cy - 0.004, cy - 0.004], color=TCPM, linewidth=1.2, zorder=7)
        else:
            curved_band(ax, cx, cy, 0.018, 0.010, STEM, 2.0, zorder=7)
        small_label(ax, cx, y + 0.017, label, color=INK, size=6.4)

    risk_x = x + width - 0.025
    ax.add_patch(Circle((risk_x, y + 0.049), 0.017, facecolor=STAGE3, edgecolor="none", zorder=7))
    ax.text(risk_x, y + 0.049, r"$R$", ha="center", va="center", color=WHITE, fontsize=8.2, fontweight="bold", zorder=8)
    arrow(ax, (cue_centers[-1] + 0.017, y + 0.049), (risk_x - 0.019, y + 0.049), color=STAGE3, lw=0.9)


def draw_fusion(ax, labels, x, y, width, height):
    rounded_box(ax, x, y, width, height, face=WHITE, edge="none", radius=0.009, zorder=4)
    ax.text(x + 0.011, y + height - 0.021, "GATED FUSION", ha="left", va="center", color=STAGE3, fontsize=6.9, fontweight="bold", zorder=7)
    cy = y + 0.052
    for cx, label, color in ((x + 0.026, "G", MODEL), (x + 0.061, "L", TCPM)):
        ax.add_patch(Circle((cx, cy), 0.013, facecolor=WHITE, edgecolor=color, linewidth=0.9, zorder=6))
        ax.text(cx, cy, label, ha="center", va="center", color=color, fontsize=7.0, fontweight="bold", zorder=7)
    gate_x = x + 0.101
    ax.add_patch(Circle((gate_x, cy), 0.016, facecolor=STAGE3_BG, edgecolor=STAGE3, linewidth=0.9, zorder=6))
    ax.text(gate_x, cy, r"$g$", ha="center", va="center", color=STAGE3, fontsize=7.5, fontstyle="italic", zorder=7)
    arrow(ax, (x + 0.074, cy), (gate_x - 0.018, cy), color=STAGE3, lw=0.8)

    mask_x = x + width - 0.060
    image_tile(ax, color_mask(labels), mask_x, y + 0.019, 0.049, height - 0.045, edge=STAGE3, lw=0.8, radius=0.006, zorder=6)
    arrow(ax, (gate_x + 0.018, cy), (mask_x - 0.004, cy), color=STAGE3, lw=0.9)


def draw_stage_two(ax, train_rgb, train_labels, unlabeled_rgb):
    image_tile(ax, overlay_labels(train_rgb, train_labels), 0.238, 0.675, 0.055, 0.126, radius=0.007)
    image_tile(ax, unlabeled_rgb, 0.304, 0.731, 0.048, 0.070, radius=0.006)
    image_tile(ax, strong_color_view(unlabeled_rgb), 0.304, 0.653, 0.048, 0.070, radius=0.006)
    for yy, text, color in ((0.786, "W", TEACHER), (0.708, "S", MODEL)):
        ax.add_patch(Circle((0.345, yy), 0.0072, facecolor=WHITE, edgecolor=color, linewidth=0.7, zorder=8))
        ax.text(0.345, yy, text, ha="center", va="center", color=color, fontsize=5.3, fontweight="bold", zorder=9)
    small_label(ax, 0.2655, 0.655, r"labeled  $\mathcal{D}_l$", color=STAGE2, size=6.0, weight="bold")
    small_label(ax, 0.328, 0.634, r"unlabeled  $\mathcal{D}_u$", color=STAGE2, size=6.0, weight="bold")

    draw_teacher_student(ax, 0.374, 0.575, 0.128, 0.226)
    arrow(ax, (0.352, 0.766), (0.374, 0.724), color=TEACHER, lw=0.9)
    arrow(ax, (0.352, 0.688), (0.374, 0.626), color=MODEL, lw=0.9)
    arrow(ax, (0.224, 0.606), (0.374, 0.606), color=STAGE1, lw=1.2)

    draw_trpl(ax, 0.518, 0.590, 0.208, 0.211)
    arrow(ax, (0.502, 0.718), (0.518, 0.718), color=TRPL, lw=1.0)
    arrow(ax, (0.518, 0.620), (0.502, 0.620), color=TRPL, lw=1.0)

    draw_tcpm(ax, 0.518, 0.335, 0.208, 0.210)
    arrow(ax, (0.502, 0.600), (0.518, 0.465), color=TCPM, lw=1.0, connection="arc3,rad=-0.08")

    draw_joint_objective(ax, 0.238, 0.142, 0.488, 0.137)
    arrow(ax, (0.622, 0.335), (0.622, 0.279), color=TCPM, lw=0.9)
    arrow(ax, (0.438, 0.279), (0.438, 0.575), color=STAGE2, lw=0.9, dashed=True)


def draw_stage_three(ax, inference_rgb, inference_labels):
    image_tile(ax, inference_rgb, 0.775, 0.684, 0.073, 0.128, radius=0.007)
    small_label(ax, 0.8115, 0.665, r"image  $x$", color=STAGE3, size=6.2, weight="bold")
    draw_segmenter(ax, 0.873, 0.688, 0.091, 0.120, accent=STAGE3, compact=True)
    arrow(ax, (0.848, 0.748), (0.873, 0.748), color=STAGE3, lw=1.0)

    draw_risk_cues(ax, 0.773, 0.527, 0.193, 0.118)
    arrow(ax, (0.918, 0.688), (0.918, 0.645), color=STAGE3, lw=1.0)

    ax.text(0.773, 0.500, "SELECTIVE ZOOM", ha="left", va="center", color=STAGE3, fontsize=6.9, fontweight="bold", zorder=7)
    windows = draw_window_image(ax, inference_rgb, inference_labels, 0.775, 0.324, 0.083, 0.158)
    arrow(ax, (0.869, 0.527), (0.869, 0.490), color=STAGE3, lw=0.85)
    crop = crop_from_grid(inference_rgb, windows[0][0], windows[0][1])
    image_tile(ax, crop, 0.887, 0.345, 0.068, 0.119, edge=MODEL, lw=0.8, radius=0.007, zorder=6)
    arrow(ax, (0.858, 0.403), (0.887, 0.403), color=MODEL, lw=0.9)
    small_label(ax, 0.8725, 0.433, "Top-K", color=STAGE3, size=5.6)
    small_label(ax, 0.8725, 0.374, "NMS", color=MUTED, size=5.6)

    draw_fusion(ax, inference_labels, 0.773, 0.142, 0.193, 0.130)
    arrow(ax, (0.921, 0.324), (0.921, 0.272), color=STAGE3, lw=0.9)


def module_panel(ax, x, y, width, height, number, acronym, full_name, color, background):
    del background
    ax.plot([x, x + width], [y + height, y + height], color=color, linewidth=1.25, solid_capstyle="butt", zorder=3)
    ax.text(x, y + height - 0.027, str(number), ha="left", va="center", color=color, fontsize=7.2, fontweight="bold", zorder=6)
    ax.text(x + 0.015, y + height - 0.027, acronym, ha="left", va="center", color=color, fontsize=8.0, fontweight="bold", zorder=6)
    ax.text(x + 0.072, y + height - 0.027, full_name, ha="left", va="center", color=INK, fontsize=5.8, zorder=6)


def operator_box(ax, x, y, width, height, title, lines, color):
    rounded_box(ax, x, y, width, height, face=WHITE, edge=color, radius=0.007, lw=0.65, zorder=5)
    ax.text(x + width / 2, y + height - 0.022, title, ha="center", va="center", color=color, fontsize=7.0, fontweight="bold", zorder=7)
    if isinstance(lines, str):
        lines = [lines]
    ax.text(
        x + width / 2,
        y + height * 0.40,
        "\n".join(lines),
        ha="center",
        va="center",
        color=INK,
        fontsize=5.3,
        linespacing=1.05,
        zorder=7,
    )


def draw_micro_model(ax, x, y, width, height, color, label=r"$f_\theta$"):
    ax.add_patch(Rectangle((x, y), width, height, facecolor=WHITE, edgecolor=color, linewidth=0.75, zorder=5))
    layers = ((0.72, MODEL), (0.88, STAGE2)) if height < 0.095 else ((0.72, MODEL), (0.88, STAGE2), (0.60, STAGE3))
    for index, (fraction, block_color) in enumerate(layers):
        yy = y + height - 0.024 - index * 0.023
        ax.plot(
            [x + width * (1 - fraction) / 2, x + width * (1 + fraction) / 2],
            [yy, yy],
            color=block_color,
            linewidth=3.2,
            solid_capstyle="round",
            zorder=6,
        )
    ax.text(x + width / 2, y + 0.014, label, ha="center", va="center", color=color, fontsize=6.2, fontweight="bold", zorder=7)


def draw_compact_teacher_student(ax, x, y, width, height):
    ax.add_patch(Rectangle((x, y), width, height, facecolor=WHITE, edgecolor=LINE, linewidth=0.55, zorder=4))
    teacher_y = y + height - 0.042
    student_y = y + 0.038
    for yy, color, label in ((teacher_y, TEACHER, "teacher"), (student_y, MODEL, "student")):
        ax.plot([x + 0.015, x + width - 0.012], [yy, yy], color=color, linewidth=4.6, solid_capstyle="round", zorder=5)
        for index in range(4):
            xx = x + 0.024 + index * (width - 0.048) / 3
            ax.add_patch(Circle((xx, yy), 0.0053, facecolor=WHITE, edgecolor=color, linewidth=0.6, zorder=6))
        ax.text(x + 0.008, yy + (0.018 if label == "teacher" else -0.020), label, ha="left", va="center", color=color, fontsize=5.8, fontweight="bold", zorder=7)
    arrow(ax, (x + width * 0.53, student_y + 0.012), (x + width * 0.53, teacher_y - 0.012), color=TEACHER, lw=0.8, dashed=True)
    ax.text(x + width * 0.70, y + height / 2, "EMA", ha="center", va="center", color=TEACHER, fontsize=5.5, zorder=7)


def draw_overview_stages(ax, train_rgb, train_labels, unlabeled_rgb, inference_rgb, inference_labels):
    stage_panel(ax, 0.012, 0.692, 0.195, 0.270, "I", "Supervised warm-up", STAGE1, STAGE1_BG)
    stage_panel(ax, 0.218, 0.692, 0.505, 0.270, "II", "Semi-supervised training", STAGE2, STAGE2_BG)
    stage_panel(ax, 0.734, 0.692, 0.254, 0.270, "III", "Frozen inference", STAGE3, STAGE3_BG)

    # Stage A: labeled pair, shared model, supervised objective, and warm-start weights.
    image_tile(ax, train_rgb, 0.027, 0.776, 0.043, 0.083, radius=0.006)
    image_tile(ax, overlay_labels(train_rgb, train_labels), 0.076, 0.776, 0.043, 0.083, radius=0.006)
    small_label(ax, 0.0485, 0.760, r"$x^l$", size=5.7)
    small_label(ax, 0.0975, 0.760, r"$y^l$", size=5.7)
    draw_micro_model(ax, 0.132, 0.766, 0.050, 0.102, STAGE1)
    arrow(ax, (0.119, 0.818), (0.132, 0.818), color=STAGE1, lw=0.85)
    ax.text(0.1015, 0.7255, r"$\mathcal{L}_{sup}=\mathcal{L}_{cls}+\mathcal{L}_{mask}+\mathcal{L}_{topology}$", ha="center", va="center", color=STAGE1, fontsize=5.3, zorder=7)
    ax.add_patch(Circle((0.200, 0.818), 0.017, facecolor=WHITE, edgecolor=STAGE1, linewidth=1.0, zorder=8))
    ax.text(0.200, 0.820, r"$\theta_0$", ha="center", va="center", color=STAGE1, fontsize=8.5, zorder=9)

    # Stage B: real labeled/unlabeled streams, EMA pair, two innovations, and joint loss.
    image_tile(ax, overlay_labels(train_rgb, train_labels), 0.234, 0.773, 0.040, 0.079, radius=0.005)
    image_tile(ax, unlabeled_rgb, 0.282, 0.813, 0.034, 0.039, radius=0.004)
    image_tile(ax, strong_color_view(unlabeled_rgb), 0.282, 0.770, 0.034, 0.039, radius=0.004)
    ax.text(0.268, 0.753, r"$\mathcal{D}_l$", ha="center", va="center", color=STAGE2, fontsize=5.7, fontweight="bold", zorder=7)
    ax.text(0.299, 0.753, r"$\mathcal{D}_u$: W/S", ha="center", va="center", color=STAGE2, fontsize=5.5, fontweight="bold", zorder=7)
    draw_compact_teacher_student(ax, 0.338, 0.748, 0.112, 0.119)
    arrow(ax, (0.316, 0.833), (0.338, 0.825), color=TEACHER, lw=0.8)
    arrow(ax, (0.316, 0.790), (0.338, 0.786), color=MODEL, lw=0.8)
    arrow(ax, (0.217, 0.818), (0.338, 0.786), color=STAGE1, lw=0.9, connection="arc3,rad=0.08")

    for yy, number, text, color in ((0.814, "1", "TRPL", TRPL), (0.756, "2", "TCPM", TCPM)):
        ax.add_patch(Rectangle((0.474, yy - 0.020), 0.079, 0.040, facecolor=WHITE, edgecolor=color, linewidth=0.65, zorder=5))
        ax.text(0.486, yy, number, ha="center", va="center", color=color, fontsize=5.7, fontweight="bold", zorder=7)
        ax.text(0.521, yy, text, ha="center", va="center", color=color, fontsize=6.1, fontweight="bold", zorder=7)
    arrow(ax, (0.450, 0.825), (0.474, 0.814), color=TRPL, lw=0.8)
    arrow(ax, (0.450, 0.786), (0.474, 0.756), color=TCPM, lw=0.8)

    ax.add_patch(Rectangle((0.574, 0.752), 0.104, 0.094, facecolor=WHITE, edgecolor=STAGE2, linewidth=0.65, zorder=5))
    ax.text(0.626, 0.826, "Joint objective", ha="center", va="center", color=STAGE2, fontsize=5.9, fontweight="bold", zorder=7)
    ax.text(0.626, 0.789, r"$\mathcal{L}_{sup}+\lambda_u\mathcal{L}_{reg}$", ha="center", va="center", color=INK, fontsize=5.1, zorder=7)
    ax.text(0.626, 0.768, r"$+\lambda_t\mathcal{L}_{topo}+\lambda_p\mathcal{L}_{proto}$", ha="center", va="center", color=INK, fontsize=4.9, zorder=7)
    arrow(ax, (0.553, 0.785), (0.574, 0.799), color=STAGE2, lw=0.8)
    ax.add_patch(Circle((0.705, 0.806), 0.017, facecolor=WHITE, edgecolor=STAGE2, linewidth=1.0, zorder=8))
    ax.text(0.705, 0.808, r"$\theta_t^*$", ha="center", va="center", color=STAGE2, fontsize=8.3, zorder=9)
    arrow(ax, (0.678, 0.806), (0.688, 0.806), color=STAGE2, lw=0.8)

    # Stage C: the converged teacher invokes the detailed BAZR route below.
    image_tile(ax, inference_rgb, 0.750, 0.772, 0.046, 0.088, radius=0.006)
    draw_micro_model(ax, 0.817, 0.766, 0.053, 0.102, STAGE3, label=r"$f_{\theta_t^*}$")
    ax.add_patch(Rectangle((0.885, 0.782), 0.057, 0.063, facecolor=WHITE, edgecolor=STAGE3, linewidth=0.7, zorder=5))
    ax.text(0.897, 0.813, "3", ha="center", va="center", color=STAGE3, fontsize=5.7, fontweight="bold", zorder=7)
    ax.text(0.921, 0.813, "BAZR", ha="center", va="center", color=STAGE3, fontsize=5.8, fontweight="bold", zorder=7)
    image_tile(ax, color_mask(inference_labels), 0.954, 0.782, 0.022, 0.063, edge=STAGE3, lw=0.7, radius=0.004, zorder=6)
    arrow(ax, (0.796, 0.816), (0.817, 0.816), color=STAGE3, lw=0.8)
    arrow(ax, (0.870, 0.816), (0.885, 0.816), color=STAGE3, lw=0.8)
    arrow(ax, (0.942, 0.816), (0.954, 0.816), color=STAGE3, lw=0.8)
    orthogonal_arrow(ax, [(0.722, 0.806), (0.731, 0.806), (0.731, 0.884), (0.844, 0.884), (0.844, 0.868)], color=INK, lw=0.8)
    ax.text(0.783, 0.893, "frozen EMA teacher", ha="center", va="center", color=MUTED, fontsize=5.5, zorder=7)


def draw_trpl_mechanism(ax, labels):
    x, y, width, height = 0.012, 0.048, 0.326, 0.604
    module_panel(ax, x, y, width, height, 1, "TRPL", "Topology-Reliable Pseudo-Labeling", TRPL, TRPL_BG)

    window = select_stem_windows(labels, count=1)[0]
    crop_labels = crop_from_grid(labels, window[0], window[1])
    maps = mechanism_maps(crop_labels)

    phase_label(ax, x + 0.014, y + 0.526, "VIEWS", TRPL, 0.080)
    phase_label(ax, x + 0.112, y + 0.526, "RELIABILITY", TRPL, 0.128)
    phase_label(ax, x + 0.259, y + 0.526, "TARGETS", TRPL, 0.052)

    # Multiple teacher predictions remain spatially registered to the same real stem geometry.
    stack_x, stack_y = x + 0.016, y + 0.338
    for index, prediction in enumerate(maps["variants"]):
        image_tile(
            ax,
            heatmap_rgb(prediction, cmap="viridis"),
            stack_x + index * 0.006,
            stack_y + index * 0.009,
            0.060,
            0.146,
            edge=WHITE,
            lw=0.7,
            radius=0.005,
            zorder=5 + index,
        )
    map_caption(ax, stack_x, y + 0.318, 0.072, r"$\mathcal{T}^{-1}_m(P_m)$", color=TRPL, size=5.5)
    ax.text(stack_x + 0.036, y + 0.292, r"$\bar P=M^{-1}\sum_mP_m$", ha="center", va="center", color=INK, fontsize=5.0, zorder=9)

    uncertainty_x = x + 0.112
    persistence_x = x + 0.183
    image_tile(ax, heatmap_rgb(maps["uncertainty"], cmap="magma"), uncertainty_x, stack_y, 0.057, 0.146, edge=TRPL, lw=0.7, radius=0.005, zorder=6)
    image_tile(ax, heatmap_rgb(maps["persistence"], cmap="Blues"), persistence_x, stack_y, 0.057, 0.146, edge=MODEL, lw=0.7, radius=0.005, zorder=6)
    map_caption(ax, uncertainty_x, y + 0.318, 0.057, r"uncertainty $U$", color=TRPL, size=5.0)
    map_caption(ax, persistence_x, y + 0.318, 0.057, r"persistence $A$", color=MODEL, size=5.0)
    ax.text(uncertainty_x + 0.0285, y + 0.285, r"$H(\bar P)+\beta\,\mathrm{JS}$", ha="center", va="center", color=INK, fontsize=4.8, zorder=9)
    ax.text(persistence_x + 0.0285, y + 0.285, r"$M^{-1}\sum_m\mathbf{1}[\mathcal{S}(P_m)]$", ha="center", va="center", color=INK, fontsize=4.5, zorder=9)
    arrow(ax, (stack_x + 0.074, stack_y + 0.098), (uncertainty_x - 0.004, stack_y + 0.102), color=TRPL, lw=0.75)
    arrow(ax, (stack_x + 0.074, stack_y + 0.056), (persistence_x - 0.004, stack_y + 0.064), color=MODEL, lw=0.75, connection="arc3,rad=0.08")

    target_x = x + 0.259
    image_tile(ax, maps["targets"], target_x, stack_y, 0.052, 0.146, edge=INK, lw=0.65, radius=0.005, zorder=6)
    map_caption(ax, target_x, y + 0.318, 0.052, "decoupled map", color=INK, size=5.0)
    arrow(ax, (uncertainty_x + 0.057, stack_y + 0.100), (target_x - 0.005, stack_y + 0.100), color=TRPL, lw=0.75)
    arrow(ax, (persistence_x + 0.057, stack_y + 0.055), (target_x - 0.005, stack_y + 0.055), color=MODEL, lw=0.75)

    legend_y = y + 0.246
    legend = (
        (x + 0.020, TCPM, r"$R^+$ region"),
        (x + 0.123, MODEL, r"$K^+$ centerline"),
        (x + 0.235, "#AAB4B8", r"$B^{?}$ ignore width"),
    )
    for xx, color, label in legend:
        ax.add_patch(Rectangle((xx, legend_y - 0.006), 0.009, 0.012, facecolor=color, edgecolor="none", zorder=7))
        ax.text(xx + 0.013, legend_y, label, ha="left", va="center", color=INK, fontsize=4.7, zorder=8)

    # Each target type has a distinct optimization consequence.
    output_y = y + 0.072
    outputs = (
        (x + 0.016, 0.092, TCPM, r"$w=e^{-U/\tau_u}$", r"$\mathcal{L}_{region}$"),
        (x + 0.117, 0.096, MODEL, r"visible topology", r"$\mathcal{L}_{topology}$"),
        (x + 0.222, 0.088, MUTED, r"no hard label", "ignore mask"),
    )
    for xx, box_w, color, upper, lower in outputs:
        ax.plot([xx, xx + box_w], [output_y + 0.095, output_y + 0.095], color=color, linewidth=0.85, zorder=5)
        ax.text(xx + box_w / 2, output_y + 0.061, upper, ha="center", va="center", color=INK, fontsize=4.9, zorder=7)
        ax.text(xx + box_w / 2, output_y + 0.023, lower, ha="center", va="center", color=color, fontsize=5.6, fontweight="bold", zorder=7)
    for target_center, color in (
        (x + 0.062, TCPM),
        (x + 0.165, MODEL),
        (x + 0.266, MUTED),
    ):
        arrow(ax, (target_center, legend_y - 0.014), (target_center, output_y + 0.099), color=color, lw=0.65, dashed=color == MUTED, zorder=6)


def draw_tcpm_mechanism(ax, labels):
    x, y, width, height = 0.346, 0.048, 0.330, 0.604
    module_panel(ax, x, y, width, height, 2, "TCPM", "Topology-Core Prototype Memory", TCPM, TCPM_BG)
    colors = (BACKGROUND, HEAD, STEM, LEAF)

    window = select_stem_windows(labels, count=1)[0]
    crop_labels = crop_from_grid(labels, window[0], window[1])
    core_visual = core_sampling_visual(crop_labels)

    phase_label(ax, x + 0.014, y + 0.526, "CORE ANCHORS", TCPM, 0.113)
    phase_label(ax, x + 0.141, y + 0.526, "DUAL MEMORY", TCPM, 0.096)
    phase_label(ax, x + 0.251, y + 0.526, "ANCHORS", TCPM, 0.064)

    # Real organ geometry determines which pixels are allowed to update prototypes.
    core_x, core_y = x + 0.016, y + 0.344
    image_tile(ax, core_visual, core_x, core_y, 0.073, 0.151, edge=TCPM, lw=0.7, radius=0.005, zorder=6)
    map_caption(ax, core_x, y + 0.324, 0.073, r"core set $\Omega_{i,c}$", color=TCPM, size=5.2)
    rounded_box(ax, x + 0.096, y + 0.416, 0.034, 0.066, face=WHITE, edge=TCPM, radius=0.007, lw=0.7, zorder=5)
    ax.text(x + 0.113, y + 0.454, r"$\sum$", ha="center", va="center", color=TCPM, fontsize=9.0, fontweight="bold", zorder=7)
    ax.text(x + 0.113, y + 0.429, r"$wz$", ha="center", va="center", color=INK, fontsize=5.0, zorder=7)
    arrow(ax, (core_x + 0.073, core_y + 0.082), (x + 0.096, y + 0.449), color=TCPM, lw=0.75)
    ax.text(x + 0.051, y + 0.286, "eroded interiors", ha="center", va="center", color=HEAD, fontsize=4.8, zorder=8)
    ax.text(x + 0.051, y + 0.268, "skeleton tube for stem", ha="center", va="center", color=MODEL, fontsize=4.8, zorder=8)

    # Labelled and pseudo-labelled evidence use separate domain-class banks.
    memory_x, memory_y = x + 0.143, y + 0.326
    ax.add_patch(Rectangle((memory_x, memory_y), 0.094, 0.181, facecolor=WHITE, edgecolor=TCPM, linewidth=0.7, zorder=5))
    ax.text(memory_x + 0.047, memory_y + 0.158, r"labelled $q^L_{c,d}$, $\mu_L=.95$", ha="center", va="center", color=TCPM, fontsize=4.5, zorder=7)
    ax.text(memory_x + 0.047, memory_y + 0.081, r"pseudo $q^P_{c,d}$, $\mu_P=.995$", ha="center", va="center", color=MODEL, fontsize=4.5, zorder=7)
    ax.plot([memory_x + 0.006, memory_x + 0.088], [memory_y + 0.096, memory_y + 0.096], color=LINE, linewidth=0.45, zorder=6)
    for bank_offset in (0.0, -0.077):
        for row, domain in enumerate((r"$d_1$", r"$d_2$")):
            yy = memory_y + 0.132 + bank_offset - row * 0.026
            ax.text(memory_x + 0.012, yy, domain, ha="center", va="center", color=INK, fontsize=4.4, zorder=8)
            for column, color in enumerate(colors):
                ax.add_patch(Circle((memory_x + 0.036 + column * 0.014, yy), 0.0043, facecolor=color, edgecolor=WHITE, linewidth=0.3, zorder=8))
    arrow(ax, (x + 0.130, y + 0.449), (memory_x - 0.004, memory_y + 0.101), color=TCPM, lw=0.75)

    anchor_x, anchor_y = x + 0.283, y + 0.420
    ax.add_patch(Circle((anchor_x, anchor_y), 0.035, facecolor=WHITE, edgecolor=TCPM, linewidth=0.8, zorder=6))
    anchor_positions = []
    for index, color in enumerate(colors):
        angle = np.pi / 4 + index * np.pi / 2
        point = (anchor_x + 0.020 * np.cos(angle), anchor_y + 0.020 * np.sin(angle))
        anchor_positions.append(point)
        ax.add_patch(Circle(point, 0.0060, facecolor=color, edgecolor=WHITE, linewidth=0.35, zorder=8))
    ax.text(anchor_x, y + 0.367, r"$\bar q_c^{(-d_i)}=(1-\eta)q_c^L+\eta q_c^P$", ha="center", va="center", color=TCPM, fontsize=3.6, zorder=8)
    arrow(ax, (memory_x + 0.094, memory_y + 0.103), (anchor_x - 0.038, anchor_y), color=TCPM, lw=0.75)
    # Explicitly encode the difficult stem/leaf separation at the anchor level.
    stem_point, leaf_point = anchor_positions[2], anchor_positions[3]
    ax.annotate("", xy=stem_point, xytext=leaf_point, arrowprops=dict(arrowstyle="<->", color=TRPL, lw=0.65), zorder=9)
    ax.text(anchor_x, y + 0.337, "stem $\leftrightarrow$ leaf", ha="center", va="center", color=TRPL, fontsize=4.5, zorder=8)

    # Clean cores write anchors; difficult reliable pixels receive gradients.
    episode_y = y + 0.164
    ax.plot([x + 0.016, x + 0.314], [episode_y + 0.092, episode_y + 0.092], color=TCPM, linewidth=0.75, zorder=5)
    ax.plot([x + 0.016, x + 0.314], [episode_y, episode_y], color=LINE, linewidth=0.45, zorder=5)
    ax.text(x + 0.028, episode_y + 0.073, "LODO + HARD QUERIES", ha="left", va="center", color=TCPM, fontsize=5.0, fontweight="bold", zorder=7)
    support_x = x + 0.058
    query_x = x + 0.254
    for index, color in enumerate(colors):
        ax.add_patch(Circle((support_x + index * 0.011, episode_y + 0.035), 0.0044, facecolor=color, edgecolor=WHITE, linewidth=0.25, zorder=8))
        query_edge = TRPL if index >= 2 else WHITE
        query_width = 0.9 if index >= 2 else 0.25
        ax.add_patch(Circle((query_x + index * 0.011, episode_y + 0.035), 0.0044, facecolor=color, edgecolor=query_edge, linewidth=query_width, zorder=8))
    ax.text(support_x + 0.017, episode_y + 0.013, r"memory $d\ne d_i$", ha="center", va="center", color=MUTED, fontsize=4.7, zorder=8)
    ax.add_patch(Circle((x + 0.164, episode_y + 0.035), 0.017, facecolor=WHITE, edgecolor=TCPM, linewidth=0.75, zorder=7))
    ax.text(x + 0.164, episode_y + 0.035, r"$\bar q_c^{(-d_i)}$", ha="center", va="center", color=TCPM, fontsize=4.8, fontweight="bold", zorder=8)
    ax.add_patch(Rectangle((query_x - 0.010, episode_y + 0.021), 0.054, 0.034, fill=False, edgecolor=TRPL, linewidth=0.65, linestyle=(0, (2, 2)), zorder=7))
    ax.text(query_x + 0.017, episode_y + 0.010, "top-25% hard, conf. >= .6", ha="center", va="center", color=TRPL, fontsize=3.9, zorder=8)
    arrow(ax, (support_x + 0.058, episode_y + 0.035), (x + 0.144, episode_y + 0.035), color=TCPM, lw=0.7)
    arrow(ax, (x + 0.182, episode_y + 0.035), (query_x - 0.012, episode_y + 0.035), color=TRPL, lw=0.7, dashed=True)
    ax.text(x + 0.302, episode_y + 0.073, "0 $q^L$ | 5k loss | 15k $q^P$ | 25k $\eta=.2$", ha="right", va="center", color=MUTED, fontsize=3.35, zorder=8)

    output_y = y + 0.070
    outputs = (
        (x + 0.016, 0.087, TCPM, r"hard query $\leftrightarrow \bar q_c^{(-d)}$", r"$\mathcal{L}_{con}$"),
        (x + 0.111, 0.091, TCPM, r"tolerant alignment", r"$\mathcal{L}_{dom}$"),
        (x + 0.210, 0.104, TRPL, "hard negative", r"stem $\leftrightarrow$ leaf"),
    )
    for xx, box_w, color, upper, lower in outputs:
        ax.plot([xx, xx + box_w], [output_y + 0.058, output_y + 0.058], color=color, linewidth=0.8, zorder=5)
        ax.text(xx + box_w / 2, output_y + 0.043, upper, ha="center", va="center", color=MUTED, fontsize=4.6, zorder=7)
        ax.text(xx + box_w / 2, output_y + 0.020, lower, ha="center", va="center", color=color, fontsize=5.2, fontweight="bold", zorder=7)


def draw_risk_tile(ax, x, y, width, height, kind, label):
    mini_frame(ax, x, y, width, height, face=WHITE)
    cx, cy = x + width / 2, y + height * 0.58
    if kind == "entropy":
        for radius, color, alpha in ((0.014, "#F7D98F", 0.45), (0.010, "#E8AA37", 0.50), (0.006, TRPL, 0.58)):
            ax.add_patch(Circle((cx, cy), radius, facecolor=color, edgecolor="none", alpha=alpha, zorder=7))
    elif kind == "endpoints":
        ax.plot([cx - 0.010, cx + 0.010], [cy - 0.005, cy + 0.006], color=STEM, linewidth=1.3, zorder=7)
        ax.add_patch(Circle((cx - 0.010, cy - 0.005), 0.0032, facecolor=TRPL, edgecolor="none", zorder=8))
        ax.add_patch(Circle((cx + 0.010, cy + 0.006), 0.0032, facecolor=TRPL, edgecolor="none", zorder=8))
    elif kind == "disagreement":
        ax.plot([cx - 0.011, cx + 0.005], [cy + 0.006, cy + 0.006], color=MODEL, linewidth=1.3, zorder=7)
        ax.plot([cx - 0.005, cx + 0.011], [cy - 0.006, cy - 0.006], color=TCPM, linewidth=1.3, zorder=7)
    else:
        curved_band(ax, cx, cy, 0.039, 0.018, STEM, 4.0, alpha=0.30)
        curved_band(ax, cx, cy, 0.039, 0.018, MODEL, 1.1)
    ax.text(cx, y + 0.011, label, ha="center", va="center", color=INK, fontsize=5.3, zorder=8)


def draw_bazr_mechanism(ax, inference_rgb, inference_labels):
    x, y, width, height = 0.684, 0.048, 0.304, 0.604
    module_panel(ax, x, y, width, height, 3, "BAZR", "Break-Aware Zoom Refinement", STAGE3, STAGE3_BG)
    maps = mechanism_maps(inference_labels)

    phase_label(ax, x + 0.014, y + 0.526, "GLOBAL PASS", STAGE3, 0.102)
    phase_label(ax, x + 0.132, y + 0.526, "RISK CUES", STAGE3, 0.101)
    phase_label(ax, x + 0.246, y + 0.526, "RISK MAP", STAGE3, 0.043)

    # The four routing cues are aligned maps, not detached scalar icons.
    image_tile(ax, inference_rgb, x + 0.014, y + 0.367, 0.053, 0.133, radius=0.005)
    image_tile(ax, color_mask(inference_labels), x + 0.076, y + 0.367, 0.045, 0.133, edge=MODEL, lw=0.7, radius=0.005, zorder=6)
    map_caption(ax, x + 0.014, y + 0.347, 0.053, "global RGB", size=5.0)
    map_caption(ax, x + 0.076, y + 0.347, 0.045, r"$P^g$", color=MODEL, size=5.5)
    arrow(ax, (x + 0.067, y + 0.434), (x + 0.076, y + 0.434), color=STAGE3, lw=0.72)

    cue_specs = (
        (x + 0.133, y + 0.434, maps["routing_uncertainty"], "magma", r"$\bar U_k$"),
        (x + 0.176, y + 0.434, maps["endpoints"], "inferno", r"$E_k$"),
        (x + 0.133, y + 0.367, maps["disagreement"], "cividis", r"$D_k$"),
        (x + 0.176, y + 0.367, maps["stem_prior"], "Blues", r"$\bar P^s_k$"),
    )
    for xx, yy, values, cmap, label in cue_specs:
        image_tile(ax, heatmap_rgb(values, cmap=cmap), xx, yy, 0.036, 0.058, edge=WHITE, lw=0.55, radius=0.004, zorder=6)
        ax.text(xx + 0.018, yy + 0.008, label, ha="center", va="center", color=WHITE, fontsize=4.7, fontweight="bold", zorder=8)
    arrow(ax, (x + 0.121, y + 0.434), (x + 0.133, y + 0.434), color=STAGE3, lw=0.72)

    risk_x = x + 0.246
    image_tile(ax, heatmap_rgb(maps["risk"], cmap="magma", background=inference_rgb, alpha=0.82), risk_x, y + 0.367, 0.043, 0.133, edge=STAGE3, lw=0.8, radius=0.005, zorder=6)
    map_caption(ax, risk_x, y + 0.347, 0.043, r"risk $r_k$", color=STAGE3, size=5.2)
    arrow(ax, (x + 0.216, y + 0.434), (risk_x - 0.004, y + 0.434), color=STAGE3, lw=0.75)
    ax.text(x + 0.211, y + 0.317, r"$r_k=\mathbf{w}^{\top}[\bar U_k,E_k,D_k,\bar P^s_k]$", ha="center", va="center", color=INK, fontsize=4.7, zorder=8)

    # Only risk-ranked windows are revisited at high resolution.
    windows = draw_window_image(ax, inference_rgb, inference_labels, x + 0.014, y + 0.159, 0.073, 0.137)
    crop_rgb = crop_from_grid(inference_rgb, windows[0][0], windows[0][1])
    crop_labels = crop_from_grid(inference_labels, windows[0][0], windows[0][1])
    ax.add_patch(Rectangle((x + 0.099, y + 0.188), 0.047, 0.078, facecolor=WHITE, edgecolor=STAGE3, linewidth=0.7, zorder=5))
    ax.text(x + 0.1225, y + 0.243, "Top-K", ha="center", va="center", color=STAGE3, fontsize=5.5, fontweight="bold", zorder=7)
    ax.text(x + 0.1225, y + 0.214, "+ NMS", ha="center", va="center", color=INK, fontsize=5.1, zorder=7)
    image_tile(ax, crop_rgb, x + 0.158, y + 0.173, 0.052, 0.108, edge=MODEL, lw=0.75, radius=0.005, zorder=6)
    draw_micro_model(ax, x + 0.220, y + 0.183, 0.037, 0.088, MODEL, label="local")
    image_tile(ax, color_mask(crop_labels), x + 0.267, y + 0.183, 0.024, 0.088, edge=TCPM, lw=0.75, radius=0.004, zorder=6)
    arrow(ax, (x + 0.087, y + 0.228), (x + 0.099, y + 0.228), color=STAGE3, lw=0.72)
    arrow(ax, (x + 0.146, y + 0.228), (x + 0.158, y + 0.228), color=MODEL, lw=0.72)
    arrow(ax, (x + 0.210, y + 0.228), (x + 0.220, y + 0.228), color=MODEL, lw=0.72)
    arrow(ax, (x + 0.257, y + 0.228), (x + 0.267, y + 0.228), color=TCPM, lw=0.72)
    orthogonal_arrow(ax, [(risk_x + 0.0215, y + 0.363), (risk_x + 0.0215, y + 0.306), (x + 0.1225, y + 0.306), (x + 0.1225, y + 0.268)], color=STAGE3, lw=0.72)

    # The gate preserves global context and changes only selected pixels.
    fusion_y = y + 0.064
    ax.plot([x + 0.014, x + 0.291], [fusion_y + 0.063, fusion_y + 0.063], color=STAGE3, linewidth=0.85, zorder=5)
    for cx, label, color in ((x + 0.038, r"$P^g$", MODEL), (x + 0.077, r"$P_k^z$", TCPM)):
        ax.add_patch(Circle((cx, fusion_y + 0.032), 0.013, facecolor=WHITE, edgecolor=color, linewidth=0.8, zorder=6))
        ax.text(cx, fusion_y + 0.032, label, ha="center", va="center", color=color, fontsize=5.6, fontweight="bold", zorder=7)
    ax.add_patch(Circle((x + 0.116, fusion_y + 0.032), 0.014, facecolor=STAGE3_BG, edgecolor=STAGE3, linewidth=0.8, zorder=6))
    ax.text(x + 0.116, fusion_y + 0.032, r"$g_k$", ha="center", va="center", color=STAGE3, fontsize=5.8, zorder=7)
    ax.text(x + 0.190, fusion_y + 0.034, r"$P^{final}=(1-g_k)P^g+g_kP_k^z$", ha="center", va="center", color=INK, fontsize=4.7, zorder=7)
    image_tile(ax, color_mask(inference_labels), x + 0.259, fusion_y + 0.008, 0.025, 0.047, edge=STAGE3, lw=0.7, radius=0.004, zorder=7)
    arrow(ax, (x + 0.090, fusion_y + 0.032), (x + 0.100, fusion_y + 0.032), color=STAGE3, lw=0.7)
    arrow(ax, (x + 0.132, fusion_y + 0.032), (x + 0.144, fusion_y + 0.032), color=STAGE3, lw=0.7)
    arrow(ax, (x + 0.231, fusion_y + 0.032), (x + 0.259, fusion_y + 0.032), color=STAGE3, lw=0.7)
    ax.text(x + 0.237, y + 0.138, r"compute: $1+K$ passes", ha="center", va="center", color=STAGE3, fontsize=4.9, fontweight="bold", zorder=8)


def draw_framework(repo_root, output_pdf, output_png):
    configure_style()
    train_rgb, train_labels, unlabeled_rgb, inference_rgb, inference_labels, rows = validate_and_load_assets(repo_root)
    manifest_path = write_manifest(repo_root, rows)

    figure, ax = plt.subplots(figsize=(6.40, 3.53))
    figure.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_overview_stages(ax, train_rgb, train_labels, unlabeled_rgb, inference_rgb, inference_labels)
    ax.plot([0.012, 0.988], [0.674, 0.674], color=LINE, linewidth=0.55, zorder=1)
    for xx in (0.2125, 0.7285):
        ax.plot([xx, xx], [0.702, 0.962], color=LINE, linewidth=0.45, zorder=1)
    for xx in (0.342, 0.680):
        ax.plot([xx, xx], [0.058, 0.652], color=LINE, linewidth=0.45, zorder=1)
    draw_trpl_mechanism(ax, inference_labels)
    draw_tcpm_mechanism(ax, inference_labels)
    draw_bazr_mechanism(ax, inference_rgb, inference_labels)

    figure.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(output_png, dpi=450, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return manifest_path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    output_pdf = repo_root / "Paper_Template" / "figures" / "framework.pdf"
    output_png = repo_root / "Paper_Template" / "figures" / "framework.png"
    manifest = draw_framework(repo_root, output_pdf, output_png)
    print("Figure PDF: {}".format(output_pdf))
    print("Figure PNG: {}".format(output_png))
    print("Asset manifest: {}".format(manifest))


if __name__ == "__main__":
    main()
