#!/usr/bin/env python3
"""Draw the TRPL mechanism figure with optional aligned teacher stem probabilities."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "topowheat-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

from plot_method_framework import (
    INK,
    LINE,
    MODEL,
    MUTED,
    TCPM,
    TEACHER,
    TRPL,
    WHITE,
    arrow,
    binary_skeleton,
    configure_style,
    heatmap_rgb,
    image_tile,
    load_mask,
    load_rgb,
    mechanism_maps,
    normalize_map,
    select_stem_windows,
    sha256,
    strong_color_view,
)


DEFAULT_IMAGE = "GWFSS/gwfss_competition_val/images/domain7_00021.png"
DEFAULT_MASK = "GWFSS/gwfss_competition_val/class_id/domain7_00021.png"
DISPLAY_SIZE = 512


def section_header(ax, x, marker, title, color):
    """Use compact journal-style panel labels instead of presentation headers."""
    ax.text(x, 0.950, "({})".format(marker), ha="left", va="center", color=color, fontsize=7.1, fontweight="bold")
    ax.text(x + 0.030, 0.950, title, ha="left", va="center", color=INK, fontsize=6.8, fontweight="bold")
    ax.plot([x, x + 0.020], [0.918, 0.918], color=color, linewidth=1.15, solid_capstyle="butt")


def label(ax, x, y, text, color=INK, size=5.4, weight="normal"):
    ax.text(x, y, text, ha="center", va="center", color=color, fontsize=size, fontweight=weight, zorder=10)


def high_resolution_crop(array, column, row, grid=4, size=DISPLAY_SIZE):
    """Crop once from the native image and retain print-resolution detail."""
    height, width = array.shape[:2]
    cell_w, cell_h = width / grid, height / grid
    margin = 0.28
    x0 = int(max(0, (column - margin) * cell_w))
    x1 = int(min(width, (column + 1 + margin) * cell_w))
    y0 = int(max(0, (row - margin) * cell_h))
    y1 = int(min(height, (row + 1 + margin) * cell_h))
    crop = np.asarray(array[y0:y1, x0:x1])
    if array.ndim == 2:
        return np.asarray(Image.fromarray(crop).resize((size, size), Image.Resampling.NEAREST))
    image = Image.fromarray(crop).resize((size, size), Image.Resampling.LANCZOS)
    image = image.filter(ImageFilter.UnsharpMask(radius=0.8, percent=70, threshold=2))
    return np.asarray(image, dtype=np.uint8)


def probability_crop(array, column, row, grid=4, size=DISPLAY_SIZE):
    height, width = array.shape
    cell_w, cell_h = width / grid, height / grid
    margin = 0.28
    x0 = int(max(0, (column - margin) * cell_w))
    x1 = int(min(width, (column + 1 + margin) * cell_w))
    y0 = int(max(0, (row - margin) * cell_h))
    y1 = int(min(height, (row + 1 + margin) * cell_h))
    image = Image.fromarray(np.asarray(array[y0:y1, x0:x1], dtype=np.float32), mode="F")
    return np.asarray(image.resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)


def resize_probability_stack(probabilities, size=DISPLAY_SIZE):
    resized = []
    for probability in probabilities:
        image = Image.fromarray(np.asarray(probability, dtype=np.float32), mode="F")
        resized.append(np.asarray(image.resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32))
    return np.stack(resized, axis=0)


def weak_views(rgb):
    base = np.asarray(rgb, dtype=np.uint8)
    bright = np.clip(base.astype(np.float32) * np.array([0.98, 1.04, 0.96]) + 4.0, 0, 255).astype(np.uint8)
    return (base, bright, strong_color_view(base))


def maps_from_aligned_probabilities(probabilities):
    stack = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError("aligned_probs must have shape [M, H, W] with M >= 2")
    mean = stack.mean(axis=0)
    entropy = -(mean * np.log(mean) + (1.0 - mean) * np.log(1.0 - mean)) / np.log(2.0)
    kl = stack * np.log(stack / mean[None, ...]) + (1.0 - stack) * np.log((1.0 - stack) / (1.0 - mean[None, ...]))
    uncertainty = normalize_map(entropy + 0.65 * kl.mean(axis=0))
    skeletons = np.stack([binary_skeleton(item > 0.45) for item in stack], axis=0)
    persistence = skeletons.mean(axis=0)
    reliable_region = (mean > 0.52) & (uncertainty < 0.62)
    stable_core = persistence >= (2.0 / 3.0)
    uncertain_boundary = (mean > 0.16) & ~reliable_region & ~stable_core

    targets = np.full((*mean.shape, 3), 247, dtype=np.uint8)
    targets[uncertain_boundary] = np.array([172, 181, 185], dtype=np.uint8)
    targets[reliable_region] = np.array([21, 139, 128], dtype=np.uint8)
    targets[stable_core] = np.array([51, 120, 185], dtype=np.uint8)
    return {
        "variants": list(stack),
        "mean": mean,
        "uncertainty": uncertainty,
        "persistence": persistence,
        "targets": targets,
        "region": reliable_region,
        "core": stable_core,
        "boundary": uncertain_boundary,
        "u_threshold": 0.62,
        "persistence_threshold": 2.0 / 3.0,
    }


def load_probability_maps(npz_path, crop_column, crop_row, full_shape):
    if npz_path is None:
        return None
    data = np.load(npz_path)
    if "aligned_probs" not in data:
        raise KeyError("prediction archive must contain 'aligned_probs' with shape [M, H, W]")
    probabilities = np.asarray(data["aligned_probs"], dtype=np.float32)
    if probabilities.ndim != 3:
        raise ValueError("aligned_probs must have shape [M, H, W]")
    if probabilities.shape[1:] == full_shape:
        probabilities = np.stack(
            [probability_crop(item, crop_column, crop_row) for item in probabilities],
            axis=0,
        )
    elif probabilities.shape[1:] != (DISPLAY_SIZE, DISPLAY_SIZE):
        if probabilities.shape[1] != probabilities.shape[2]:
            raise ValueError("cropped aligned_probs must have square spatial dimensions")
        probabilities = resize_probability_stack(probabilities)
    return maps_from_aligned_probabilities(probabilities)


def draw_teacher(ax, x, y, width, height):
    ax.add_patch(Rectangle((x, y), width, height, facecolor=WHITE, edgecolor=TEACHER, linewidth=0.75, zorder=5))
    ax.text(x + width / 2, y + height - 0.035, "EMA", ha="center", va="center", color=TEACHER, fontsize=5.6, fontweight="bold", zorder=7)
    for index, color in enumerate(("#8A9AA7", "#71879A", "#597486")):
        yy = y + 0.095 - index * 0.026
        ax.plot([x + 0.012, x + width - 0.012], [yy, yy], color=color, linewidth=3.0, solid_capstyle="round", zorder=6)
    ax.text(x + width / 2, y + 0.020, r"$f_{\theta_t}$", ha="center", va="center", color=TEACHER, fontsize=6.0, zorder=7)


def contour_envelope(rgb, maps):
    canvas = np.asarray(rgb, dtype=np.float32).copy()
    colors = (
        np.array([206, 81, 75], dtype=np.float32),
        np.array([51, 120, 185], dtype=np.float32),
        np.array([210, 138, 10], dtype=np.float32),
    )
    for probability, color in zip(maps["variants"][:3], colors):
        selected = probability > 0.34
        edge = selected & ~ndi.binary_erosion(selected)
        edge = ndi.binary_dilation(edge, iterations=max(1, rgb.shape[0] // 256))
        canvas[edge] = 0.18 * canvas[edge] + 0.82 * color
    stable = maps["core"]
    canvas[stable] = 0.08 * canvas[stable] + 0.92 * np.array([244, 248, 250], dtype=np.float32)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def target_overlay(rgb, maps):
    canvas = 0.72 * np.asarray(rgb, dtype=np.float32) + 0.28 * 248.0
    for selected, color, alpha in (
        (maps["boundary"], np.array([145, 155, 160], dtype=np.float32), 0.60),
        (maps["region"], np.array([21, 139, 128], dtype=np.float32), 0.72),
        (maps["core"], np.array([51, 120, 185], dtype=np.float32), 0.95),
    ):
        canvas[selected] = (1.0 - alpha) * canvas[selected] + alpha * color
    return np.clip(canvas, 0, 255).astype(np.uint8)


def target_layer(rgb, mask, color):
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32)
    canvas = np.repeat((0.18 * gray + 0.82 * 250.0)[..., None], 3, axis=2)
    canvas[mask] = 0.10 * canvas[mask] + 0.90 * np.asarray(color, dtype=np.float32)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def probability_overlay(rgb, probability, color):
    """Render probability mass and its decision contour on the same real crop."""
    probability = normalize_map(probability)
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32)
    canvas = np.repeat((0.42 * gray + 0.58 * 246.0)[..., None], 3, axis=2)
    tint = np.asarray(color, dtype=np.float32)
    alpha = (0.12 + 0.72 * probability)[..., None]
    canvas = (1.0 - alpha) * canvas + alpha * tint
    selected = probability > 0.34
    edge = selected & ~ndi.binary_erosion(selected)
    edge = ndi.binary_dilation(edge, iterations=max(1, rgb.shape[0] // 256))
    canvas[edge] = 0.05 * canvas[edge] + 0.95 * tint
    return np.clip(canvas, 0, 255).astype(np.uint8)


def skeleton_ensemble(rgb, maps):
    """Overlay view-specific skeletons to expose structural agreement."""
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32)
    canvas = np.repeat((0.25 * gray + 0.75 * 247.0)[..., None], 3, axis=2)
    colors = (
        np.array([206, 81, 75], dtype=np.float32),
        np.array([51, 120, 185], dtype=np.float32),
        np.array([210, 138, 10], dtype=np.float32),
    )
    for probability, color in zip(maps["variants"][:3], colors):
        skeleton = binary_skeleton(np.asarray(probability) > 0.34)
        skeleton = ndi.binary_dilation(skeleton, iterations=max(1, rgb.shape[0] // 256))
        canvas[skeleton] = 0.08 * canvas[skeleton] + 0.92 * color
    return np.clip(canvas, 0, 255).astype(np.uint8)


def draw_phase_plane(ax, x, y, width, height, maps):
    phase = ax.inset_axes([x, y, width, height])
    uncertainty = np.asarray(maps["uncertainty"], dtype=np.float32)
    persistence = np.asarray(maps["persistence"], dtype=np.float32)
    candidate = np.asarray(maps["mean"] > 0.10, dtype=bool)
    u_stem = float(maps.get("u_threshold", 0.62))
    rho = float(maps.get("persistence_threshold", 2.0 / 3.0))

    # The four quadrants make the decoupling explicit; density contours show
    # where candidate stem pixels actually fall for the displayed crop.
    phase.add_patch(Rectangle((0.0, 0.0), u_stem, rho, facecolor="#E4F4EF", edgecolor="none", zorder=0))
    phase.add_patch(Rectangle((0.0, rho), u_stem, 1.0 - rho, facecolor="#DCEDEF", edgecolor="none", zorder=0))
    phase.add_patch(Rectangle((u_stem, rho), 1.0 - u_stem, 1.0 - rho, facecolor="#E5EFF8", edgecolor="none", zorder=0))
    phase.add_patch(Rectangle((u_stem, 0.0), 1.0 - u_stem, rho, facecolor="#F1F3F3", edgecolor="none", zorder=0))

    xx = uncertainty[candidate]
    yy = persistence[candidate]
    hist, xedges, yedges = np.histogram2d(xx, yy, bins=(54, 42), range=((0.0, 1.0), (0.0, 1.0)))
    density = ndi.gaussian_filter(hist.T, sigma=(1.15, 1.25))
    if density.max() > 0:
        density /= density.max()
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        phase.contourf(xcenters, ycenters, density, levels=(0.08, 0.20, 0.38, 0.60, 1.01), cmap="Greys", alpha=0.34, zorder=1)
        phase.contour(xcenters, ycenters, density, levels=(0.16, 0.36, 0.64), colors=INK, linewidths=(0.30, 0.42, 0.58), alpha=0.72, zorder=2)

    phase.axvline(u_stem, color=TRPL, linewidth=0.75, linestyle=(0, (3, 2)), zorder=3)
    phase.axhline(rho, color=MODEL, linewidth=0.75, linestyle=(0, (3, 2)), zorder=3)
    phase.text(0.30, 0.29, r"$R^+$", ha="center", va="center", color=TCPM, fontsize=6.1, fontweight="bold")
    phase.text(0.30, 0.84, r"$R^+\cap K^+$", ha="center", va="center", color="#216C83", fontsize=5.6, fontweight="bold")
    phase.text(0.81, 0.84, r"$K^+$", ha="center", va="center", color=MODEL, fontsize=6.1, fontweight="bold")
    phase.text(0.81, 0.29, r"$B^{?}$", ha="center", va="center", color=MUTED, fontsize=6.1, fontweight="bold")
    phase.text(u_stem - 0.018, 0.035, r"$u_{\mathrm{stem}}$", ha="right", va="bottom", color=TRPL, fontsize=5.3)
    phase.text(0.975, rho + 0.025, r"$\rho$", ha="right", va="bottom", color=MODEL, fontsize=5.3)
    phase.set_xlim(0.0, 1.0)
    phase.set_ylim(0.0, 1.0)
    phase.set_xticks((0.0, 0.5, 1.0))
    phase.set_yticks((0.0, 0.5, 1.0))
    phase.tick_params(axis="both", labelsize=5.3, length=1.8, pad=1.0, colors=MUTED)
    phase.set_xlabel(r"distributional uncertainty $U(p)$", fontsize=5.5, color=INK, labelpad=0.7)
    phase.set_ylabel(r"skeleton persistence $A(p)$", fontsize=5.5, color=INK, labelpad=0.7)
    for side in ("top", "right"):
        phase.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        phase.spines[side].set_color(LINE)
        phase.spines[side].set_linewidth(0.6)


def draw_figure(rgb_crop, maps, output_pdf, output_png):
    configure_style()
    figure, ax = plt.subplots(figsize=(5.85, 3.10))
    figure.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    section_header(ax, 0.018, "a", "View-aligned evidence", TRPL)
    section_header(ax, 0.405, "b", "Dual reliability fields", MODEL)
    section_header(ax, 0.805, "c", "Factorized supervision", TCPM)

    # (a) A single real crop branches into weak views and returns as registered
    # probability fields. Colored contours expose boundary disagreement directly.
    views = weak_views(rgb_crop)
    view_y = (0.675, 0.435, 0.195)
    contour_colors = (TRPL, MODEL, "#D28A0A")
    contour_rgb = ((206, 81, 75), (51, 120, 185), (210, 138, 10))
    image_tile(ax, rgb_crop, 0.010, 0.385, 0.095, 0.185, edge=INK, lw=0.55, radius=0.002, zorder=6)
    label(ax, 0.0575, 0.350, r"unlabeled $x^u$", size=5.3)
    ax.plot([0.114, 0.114], [0.245, 0.775], color=LINE, linewidth=0.65, zorder=2)
    arrow(ax, (0.106, 0.478), (0.112, 0.478), color=TEACHER, lw=0.65)
    for index, (view, yy, color, rgb_color) in enumerate(zip(views, view_y, contour_colors, contour_rgb), start=1):
        center_y = yy + 0.055
        ax.plot([0.114, 0.124], [center_y, center_y], color=LINE, linewidth=0.65, zorder=2)
        image_tile(ax, view, 0.126, yy, 0.060, 0.112, edge=color, lw=0.48, radius=0.002, zorder=6)
        ax.text(0.120, center_y, r"$\mathcal{T}_{%d}$" % index, ha="right", va="center", color=color, fontsize=5.2)
        probability = maps["variants"][index - 1]
        image_tile(ax, probability_overlay(rgb_crop, probability, rgb_color), 0.216, yy, 0.060, 0.112, edge=color, lw=0.55, radius=0.002, zorder=6)
        arrow(ax, (0.189, center_y), (0.213, center_y), color=TEACHER, lw=0.55)
        ax.text(0.201, center_y + 0.035, r"$f_{\theta_t}$", ha="center", va="center", color=TEACHER, fontsize=5.1)
        ax.text(0.246, yy - 0.020, r"$P_%d$" % index, ha="center", va="center", color=color, fontsize=5.4)

    envelope = contour_envelope(rgb_crop, maps)
    image_tile(ax, envelope, 0.293, 0.385, 0.095, 0.185, edge=INK, lw=0.6, radius=0.002, zorder=6)
    for yy, color in zip(view_y, contour_colors):
        arrow(ax, (0.279, yy + 0.055), (0.290, 0.478), color=color, lw=0.45, connection="arc3,rad=0.08")
    label(ax, 0.3405, 0.350, "common coordinates", color=INK, size=5.2)
    for xx, color, text in ((0.293, TRPL, r"$P_1$"), (0.327, MODEL, r"$P_2$"), (0.361, "#D28A0A", r"$P_3$")):
        ax.plot([xx, xx + 0.012], [0.322, 0.322], color=color, linewidth=1.15)
        ax.text(xx + 0.014, 0.322, text, ha="left", va="center", color=color, fontsize=5.2)
    ax.text(0.202, 0.090, r"$P_m=\mathcal{T}^{-1}_m\!\left[f_{\theta_t}(\mathcal{T}_m(x^u))\right]$", ha="center", va="center", color=INK, fontsize=5.4)

    # (b) The registered stack is read through two complementary operators.
    ax.add_patch(Rectangle((0.405, 0.535), 0.205, 0.345, facecolor="#FFF6F4", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.405, 0.105), 0.205, 0.345, facecolor="#F2F7FB", edgecolor="none", zorder=0))
    ax.text(0.417, 0.850, "DISTRIBUTIONAL", ha="left", va="center", color=TRPL, fontsize=5.3, fontweight="bold")
    ax.text(0.417, 0.420, "STRUCTURAL", ha="left", va="center", color=MODEL, fontsize=5.3, fontweight="bold")

    mean_overlay = heatmap_rgb(maps["mean"], cmap="YlGn", background=rgb_crop, alpha=0.84)
    uncertainty_overlay = heatmap_rgb(maps["uncertainty"], cmap="magma", background=rgb_crop, alpha=0.92)
    skeleton_overlay = skeleton_ensemble(rgb_crop, maps)
    persistence_overlay = heatmap_rgb(maps["persistence"], cmap="Blues", background=rgb_crop, alpha=0.98)
    branch_specs = (
        (0.650, mean_overlay, uncertainty_overlay, TCPM, TRPL, r"$\bar P$", r"$U(p)$", r"$U=H(\bar P)/\log C+\beta\,\mathrm{JS}$" + "\n" + r"$w=e^{-U/\tau_u}$"),
        (0.220, skeleton_overlay, persistence_overlay, "#D28A0A", MODEL, r"$\{\mathcal{S}(P_m)\}$", r"$A(p)$", r"$M^{-1}\!\sum_m\mathbf{1}[\mathcal{S}(P_m)>\tau_k]$"),
    )
    for yy, left_image, right_image, left_color, right_color, left_text, right_text, formula in branch_specs:
        image_tile(ax, left_image, 0.412, yy, 0.075, 0.142, edge=left_color, lw=0.55, radius=0.002, zorder=6)
        image_tile(ax, right_image, 0.525, yy, 0.075, 0.142, edge=right_color, lw=0.6, radius=0.002, zorder=6)
        arrow(ax, (0.490, yy + 0.071), (0.522, yy + 0.071), color=right_color, lw=0.65)
        ax.text(0.4495, yy - 0.022, left_text, ha="center", va="center", color=left_color, fontsize=5.4)
        ax.text(0.5625, yy - 0.022, right_text, ha="center", va="center", color=right_color, fontsize=5.6, fontweight="bold")
        ax.text(0.506, yy - 0.065, formula, ha="center", va="center", color=right_color, fontsize=5.2)
    arrow(ax, (0.390, 0.490), (0.409, 0.721), color=TRPL, lw=0.65, connection="arc3,rad=-0.12")
    arrow(ax, (0.390, 0.470), (0.409, 0.291), color=MODEL, lw=0.65, connection="arc3,rad=0.12")

    # The phase portrait is the visual centre: colored decision regions encode
    # the gate, while isolines retain the empirical distribution of this crop.
    ax.text(0.700, 0.850, "JOINT RELIABILITY GATE", ha="center", va="center", color=INK, fontsize=5.5, fontweight="bold")
    draw_phase_plane(ax, 0.620, 0.305, 0.172, 0.325, maps)
    arrow(ax, (0.603, 0.720), (0.700, 0.635), color=TRPL, lw=0.65, connection="arc3,rad=0.08")
    arrow(ax, (0.603, 0.345), (0.617, 0.430), color=MODEL, lw=0.65, connection="arc3,rad=-0.15")
    ax.text(0.705, 0.190, r"$R^+=[\hat c=\mathrm{stem}]\cap[w\geq\tau_{\mathrm{stem}}]$", ha="center", va="center", color=TCPM, fontsize=5.3)
    ax.text(0.705, 0.125, r"$K^+=[A\geq\rho]$", ha="center", va="center", color=MODEL, fontsize=5.3)

    # (c) The same spatial support is factorized into region, centreline, and
    # ignored-boundary channels instead of being collapsed into one hard mask.
    composite = target_overlay(rgb_crop, maps)
    image_tile(ax, composite, 0.812, 0.635, 0.096, 0.180, edge=INK, lw=0.6, radius=0.002, zorder=6)
    label(ax, 0.860, 0.845, r"joint target $\hat y$", size=5.5)
    arrow(ax, (0.795, 0.475), (0.809, 0.705), color=TCPM, lw=0.72, connection="arc3,rad=-0.12")

    layer_specs = (
        (0.812, maps["region"], (21, 139, 128), TCPM, r"$wR^+$", r"$\mathcal{L}_{region}$"),
        (0.873, maps["core"], (51, 120, 185), MODEL, r"$K^+$", r"$\mathcal{L}_{topology}$"),
        (0.934, maps["boundary"], (145, 155, 160), MUTED, r"$B^{?}$", "ignore"),
    )
    branch_centers = tuple(xx + 0.0275 for xx, *_ in layer_specs)
    arrow(ax, (0.860, 0.625), (0.860, 0.535), color=INK, lw=0.58)
    ax.plot([branch_centers[0], branch_centers[-1]], [0.510, 0.510], color=LINE, linewidth=0.65, zorder=2)
    ax.plot([0.860, 0.860], [0.535, 0.510], color=LINE, linewidth=0.65, zorder=2)
    ax.text(0.930, 0.542, "target factorization", ha="center", va="center", color=MUTED, fontsize=5.2)
    for xx, selected, rgb_color, color, source, target in layer_specs:
        center = xx + 0.0235
        arrow(ax, (center, 0.510), (center, 0.455), color=color, lw=0.52)
        image_tile(ax, target_layer(rgb_crop, selected, rgb_color), xx, 0.340, 0.055, 0.106, edge=color, lw=0.55, radius=0.002, zorder=6)
        ax.text(center, 0.306, source, ha="center", va="center", color=color, fontsize=5.4, fontweight="bold")
        arrow(ax, (center, 0.280), (center, 0.215), color=color, lw=0.58, dashed=color == MUTED)
        ax.text(center, 0.173, target, ha="center", va="center", color=color, fontsize=5.2, fontweight="bold")
    ax.text(0.901, 0.085, r"$\mathcal{L}_{u}=\mathcal{L}_{region}+\lambda_t\mathcal{L}_{topology}$", ha="center", va="center", color=INK, fontsize=5.5)

    figure.subplots_adjust(left=0.005, right=0.995, top=0.988, bottom=0.018)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, dpi=600, bbox_inches="tight", pad_inches=0.018)
    figure.savefig(output_png, dpi=600, bbox_inches="tight", pad_inches=0.018)
    plt.close(figure)


def write_manifest(path, image_path, mask_path, prediction_path, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["role", "path", "sha256", "mode"]
    rows = [
        {"role": "rgb", "path": str(image_path), "sha256": sha256(image_path), "mode": mode},
        {"role": "annotation", "path": str(mask_path), "sha256": sha256(mask_path), "mode": mode},
    ]
    if prediction_path is not None:
        rows.append({"role": "aligned_teacher_probabilities", "path": str(prediction_path), "sha256": sha256(prediction_path), "mode": mode})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Source RGB image; defaults to the aligned validation example")
    parser.add_argument("--mask", type=Path, help="Source class-id mask; defaults to the aligned validation example")
    parser.add_argument(
        "--prediction-npz",
        type=Path,
        help="Optional NPZ containing aligned stem probabilities as aligned_probs[M,H,W]; real predictions replace geometry-derived schematic maps",
    )
    parser.add_argument("--output-pdf", type=Path)
    parser.add_argument("--output-png", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    image_path = (args.image or (repo_root / DEFAULT_IMAGE)).resolve()
    mask_path = (args.mask or (repo_root / DEFAULT_MASK)).resolve()
    prediction_path = args.prediction_npz.resolve() if args.prediction_npz else None
    output_pdf = (args.output_pdf or (repo_root / "Paper_Template/figures/topology_pseudo_label.pdf")).resolve()
    output_png = (args.output_png or (repo_root / "Paper_Template/figures/topology_pseudo_label.png")).resolve()

    rgb = load_rgb(image_path)
    labels = load_mask(mask_path)
    column, row = select_stem_windows(labels, count=1)[0]
    rgb_crop = high_resolution_crop(rgb, column, row)
    labels_crop = high_resolution_crop(labels, column, row)
    maps = load_probability_maps(prediction_path, column, row, labels.shape)
    mode = "model_predictions" if maps is not None else "geometry_schematic"
    if maps is None:
        maps = mechanism_maps(labels_crop)
        maps.setdefault("u_threshold", 0.68)
        maps.setdefault("persistence_threshold", 2.0 / 3.0)

    draw_figure(rgb_crop, maps, output_pdf, output_png)
    manifest = repo_root / "analysis_outputs/trpl/figure_assets_manifest.csv"
    write_manifest(manifest, image_path, mask_path, prediction_path, mode)
    print("Figure PDF: {}".format(output_pdf))
    print("Figure PNG: {}".format(output_png))
    print("Mode: {}".format(mode))
    print("Asset manifest: {}".format(manifest))


if __name__ == "__main__":
    main()
