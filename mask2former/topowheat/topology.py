import torch
import torch.nn.functional as F


_TRPL_SKELETON_ITERATIONS = 20
_TRPL_SPATIAL_TOLERANCE = 1


def query_semantic_probabilities(pred_logits, pred_masks, size=None, eps=1e-6):
    """Compose Mask2Former query predictions into normalized class probabilities."""
    class_prob = F.softmax(pred_logits.float(), dim=-1)[..., :-1]
    mask_prob = pred_masks.float().sigmoid()
    semantic = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)
    if size is not None and tuple(semantic.shape[-2:]) != tuple(size):
        semantic = F.interpolate(
            semantic,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
    return semantic / semantic.sum(dim=1, keepdim=True).clamp_min(eps)


def soft_erode(mask):
    if mask.ndim != 4:
        raise ValueError("soft_erode expects an NCHW tensor")
    eroded_h = -F.max_pool2d(-mask, (3, 1), stride=1, padding=(1, 0))
    eroded_w = -F.max_pool2d(-mask, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(eroded_h, eroded_w)


def soft_dilate(mask):
    if mask.ndim != 4:
        raise ValueError("soft_dilate expects an NCHW tensor")
    return F.max_pool2d(mask, 3, stride=1, padding=1)


def soft_open(mask):
    return soft_dilate(soft_erode(mask))


def soft_skeletonize(mask, iterations=20):
    """Differentiable morphological skeletonization used by soft-clDice."""
    mask = mask.float().clamp(0.0, 1.0)
    opened = soft_open(mask)
    skeleton = F.relu(mask - opened)
    for _ in range(max(int(iterations), 0)):
        mask = soft_erode(mask)
        opened = soft_open(mask)
        delta = F.relu(mask - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0.0, 1.0)


def hard_skeletonize(mask, iterations=20, threshold=0.5):
    return soft_skeletonize(mask.float(), iterations).ge(threshold)


def endpoint_map(mask, skeleton_iterations=20, dilation=5):
    """Return a dense heatmap around endpoints of a binary structure."""
    skeleton = hard_skeletonize(mask, skeleton_iterations).float()
    kernel = torch.ones(
        (1, 1, 3, 3),
        device=mask.device,
        dtype=skeleton.dtype,
    )
    neighbours = F.conv2d(skeleton, kernel, padding=1) - skeleton
    endpoints = skeleton * neighbours.le(1.0).to(skeleton)
    if dilation > 1:
        endpoints = F.max_pool2d(
            endpoints,
            kernel_size=dilation,
            stride=1,
            padding=dilation // 2,
        )
    return endpoints


def _binary_erode(mask, iterations):
    result = mask.float()
    for _ in range(max(int(iterations), 0)):
        result = soft_erode(result)
    return result.gt(0.5)


def _binary_dilate(mask, iterations):
    result = mask.float()
    for _ in range(max(int(iterations), 0)):
        result = soft_dilate(result)
    return result.gt(0.5)


def build_core_mask(
    labels,
    reliable,
    num_classes,
    stem_class=2,
    stable_skeleton=None,
    strategy="topology",
    erode_iterations=2,
    stem_radius=1,
    skeleton_iterations=20,
):
    """Select topology-aware organ interiors for prototype construction."""
    if labels.ndim != 3 or reliable.ndim != 3:
        raise ValueError("labels and reliable must have shape [N, H, W]")
    if strategy not in {"reliable", "eroded", "topology"}:
        raise ValueError("Unknown prototype core strategy: {}".format(strategy))
    if strategy == "reliable":
        return reliable.bool()

    core = torch.zeros_like(reliable, dtype=torch.bool)
    for class_id in range(num_classes):
        class_mask = labels.eq(class_id).unsqueeze(1)
        if class_id == stem_class and strategy == "topology":
            if stable_skeleton is None:
                class_core = hard_skeletonize(
                    class_mask,
                    iterations=skeleton_iterations,
                )
                class_core = _binary_dilate(class_core, stem_radius)
                class_core &= class_mask
                class_core &= reliable.unsqueeze(1)
            else:
                class_core = stable_skeleton.unsqueeze(1).bool()
                class_core = _binary_dilate(class_core, stem_radius)
                class_core &= class_mask
        else:
            class_core = _binary_erode(class_mask, erode_iterations)
            class_core &= reliable.unsqueeze(1)
        core |= class_core.squeeze(1)
    return core


@torch.no_grad()
def build_trpl_targets(
    view_probabilities,
    reliability_threshold=0.75,
    stem_class=2,
):
    """Build reliable regions and stable centrelines from aligned teacher views."""
    if len(view_probabilities) < 2:
        raise ValueError("TRPL requires at least two aligned teacher views")
    shape = view_probabilities[0].shape
    if len(shape) != 4 or shape[1] < 2:
        raise ValueError("TRPL views must have shape [N, C, H, W] with C > 1")
    if any(prob.shape != shape for prob in view_probabilities):
        raise ValueError("All TRPL views must be aligned and have identical shapes")
    if not 0 <= int(stem_class) < shape[1]:
        raise ValueError("TRPL stem class is out of range")
    if not 0.0 < float(reliability_threshold) < 1.0:
        raise ValueError("TRPL reliability threshold must be in (0, 1)")

    eps = 1e-6
    probabilities = []
    for probability in view_probabilities:
        probability = probability.float().clamp_min(eps)
        probabilities.append(
            probability
            / probability.sum(dim=1, keepdim=True).clamp_min(eps)
        )
    mean_prob = torch.stack(probabilities, dim=0).mean(dim=0)
    mean_prob = mean_prob / mean_prob.sum(dim=1, keepdim=True).clamp_min(eps)

    confidence, labels = mean_prob.max(dim=1)
    view_disagreement = torch.stack(
        [
            0.5 * (probability - mean_prob).abs().sum(dim=1)
            for probability in probabilities
        ],
        dim=0,
    ).mean(dim=0)
    reliability = (
        confidence * (1.0 - view_disagreement)
    ).clamp(0.0, 1.0)
    reliable = reliability.ge(float(reliability_threshold))

    view_skeletons = [
        hard_skeletonize(
            probability.argmax(dim=1).eq(stem_class).unsqueeze(1),
            iterations=_TRPL_SKELETON_ITERATIONS,
        )
        for probability in probabilities
    ]
    matched_skeletons = [
        _binary_dilate(skeleton, _TRPL_SPATIAL_TOLERANCE)
        for skeleton in view_skeletons
    ]
    strict_consensus = torch.stack(matched_skeletons, dim=0).all(dim=0)
    mean_skeleton = hard_skeletonize(
        labels.eq(stem_class).unsqueeze(1),
        iterations=_TRPL_SKELETON_ITERATIONS,
    )
    stable_skeleton = (mean_skeleton & strict_consensus).squeeze(1)

    reliable_stem = reliable & labels.eq(stem_class)
    structural_stem = reliable_stem | stable_skeleton
    stem_neighbourhood = _binary_dilate(
        structural_stem.unsqueeze(1),
        _TRPL_SPATIAL_TOLERANCE,
    ).squeeze(1)
    uncertain_boundary = (
        labels.eq(stem_class)
        & stem_neighbourhood
        & ~reliable
        & ~stable_skeleton
    )

    return {
        "labels": labels,
        "probabilities": mean_prob,
        "weights": reliability,
        "reliable": reliable,
        "reliability": reliability,
        "confidence": confidence,
        "view_disagreement": view_disagreement,
        "stable_skeleton": stable_skeleton,
        "uncertain_boundary": uncertain_boundary,
    }


def masked_nll_loss(
    probabilities,
    labels,
    weights,
    valid,
    eps=1e-6,
    class_balanced=False,
    num_classes=None,
    min_class_pixels=1,
):
    log_probabilities = probabilities.clamp_min(eps).log()
    per_pixel = F.nll_loss(log_probabilities, labels, reduction="none")
    pixel_weights = weights.float() * valid.float()
    if class_balanced:
        if num_classes is None:
            num_classes = probabilities.shape[1]
        class_ids = torch.arange(
            int(num_classes),
            device=labels.device,
        ).view(1, -1, 1, 1)
        class_valid = (
            labels.unsqueeze(1).eq(class_ids)
            & valid.unsqueeze(1).bool()
        )
        counts = class_valid.sum(dim=(0, 2, 3)).float()
        weighted_losses = (
            per_pixel.unsqueeze(1)
            * pixel_weights.unsqueeze(1)
            * class_valid.float()
        ).sum(dim=(0, 2, 3))
        present = counts.ge(max(int(min_class_pixels), 1))
        if present.any():
            return (weighted_losses[present] / counts[present]).mean()
        return probabilities.sum() * 0.0
    # Normalize by accepted pixels, not by confidence mass. Consequently a
    # uniformly low-confidence pseudo-label contributes less than a uniformly
    # confident one instead of having the scale cancel out.
    return (per_pixel * pixel_weights).sum() / valid.float().sum().clamp_min(1.0)


def class_balanced_consistency_loss(
    student_probabilities,
    target_probabilities,
    labels,
    weights,
    valid,
    eps=1e-6,
):
    """Class-average teacher-student KL over accepted pseudo regions."""
    if student_probabilities.shape != target_probabilities.shape:
        raise ValueError("TRPL teacher and student probabilities must match")
    if labels.shape != valid.shape or labels.shape != weights.shape:
        raise ValueError("TRPL labels, weights and valid mask must match")

    student = student_probabilities.float().clamp_min(eps)
    teacher = target_probabilities.float().clamp_min(eps)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(eps)
    per_pixel = (
        teacher * (teacher.log() - student.log())
    ).sum(dim=1).clamp_min(0.0)

    num_classes = student.shape[1]
    class_ids = torch.arange(
        num_classes,
        device=labels.device,
    ).view(1, -1, 1, 1)
    class_valid = labels.unsqueeze(1).eq(class_ids) & valid.unsqueeze(1).bool()
    counts = class_valid.sum(dim=(0, 2, 3)).float()
    weighted_losses = (
        per_pixel.unsqueeze(1)
        * weights.float().unsqueeze(1)
        * class_valid.float()
    ).sum(dim=(0, 2, 3))
    present = counts.gt(0)
    if present.any():
        return (weighted_losses[present] / counts[present]).mean()
    return student_probabilities.sum() * 0.0


def stable_centerline_loss(
    prediction,
    target,
    eps=1e-6,
):
    """Maximize stem probability around stable multi-view centreline evidence."""
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("prediction and target must have shape [N, 1, H, W]")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    prediction = prediction.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    prediction = F.max_pool2d(
        prediction,
        kernel_size=2 * _TRPL_SPATIAL_TOLERANCE + 1,
        stride=1,
        padding=_TRPL_SPATIAL_TOLERANCE,
    )
    target_mass = target.sum(dim=(1, 2, 3))
    losses = -(
        prediction.clamp_min(eps).log() * target
    ).sum(dim=(1, 2, 3)) / target_mass.clamp_min(1.0)
    present = target_mass.gt(0.0)
    if bool(present.any()):
        return losses[present].mean()
    return prediction.sum() * 0.0
