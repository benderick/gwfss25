import math

import torch
import torch.nn.functional as F


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
            else:
                class_core = stable_skeleton.unsqueeze(1).bool()
            class_core = _binary_dilate(class_core, stem_radius)
            class_core &= class_mask
        else:
            class_core = _binary_erode(class_mask, erode_iterations)
        core |= class_core.squeeze(1) & reliable
    return core


@torch.no_grad()
def build_trpl_targets(
    view_probabilities,
    class_thresholds,
    uncertainty_temperature=0.5,
    uncertainty_weight=1.0,
    max_uncertainty=0.75,
    stem_class=2,
    skeleton_threshold=0.35,
    persistence=0.5,
    skeleton_iterations=20,
    boundary_radius=2,
    core_strategy="topology",
    core_erode_iterations=2,
    core_stem_radius=1,
):
    """Build topology-reliable semantic targets from aligned teacher views."""
    if not view_probabilities:
        raise ValueError("TRPL requires at least one teacher view")
    shape = view_probabilities[0].shape
    if any(prob.shape != shape for prob in view_probabilities):
        raise ValueError("All TRPL views must be aligned and have identical shapes")

    eps = 1e-6
    probabilities = [prob.float().clamp_min(eps) for prob in view_probabilities]
    mean_prob = torch.stack(probabilities, dim=0).mean(dim=0)
    mean_prob = mean_prob / mean_prob.sum(dim=1, keepdim=True).clamp_min(eps)

    entropy = -(mean_prob * mean_prob.log()).sum(dim=1) / math.log(shape[1])
    js_terms = []
    for probability in probabilities:
        midpoint = 0.5 * (probability + mean_prob)
        js = 0.5 * (
            (probability * (probability.log() - midpoint.log())).sum(dim=1)
            + (mean_prob * (mean_prob.log() - midpoint.log())).sum(dim=1)
        )
        js_terms.append(js / math.log(2.0))
    view_disagreement = torch.stack(js_terms, dim=0).mean(dim=0)
    uncertainty = entropy + uncertainty_weight * view_disagreement
    weights = torch.exp(
        -uncertainty / max(float(uncertainty_temperature), eps)
    ).clamp(0.0, 1.0)

    confidence, labels = mean_prob.max(dim=1)
    thresholds = torch.as_tensor(
        class_thresholds,
        device=labels.device,
        dtype=confidence.dtype,
    )
    if thresholds.numel() != shape[1]:
        raise ValueError(
            "Expected {} TRPL class thresholds, got {}".format(
                shape[1], thresholds.numel()
            )
        )
    reliable = (
        confidence.ge(thresholds[labels])
        & uncertainty.le(float(max_uncertainty))
    )

    skeletons = []
    for probability in probabilities:
        stem_probability = probability[:, stem_class : stem_class + 1]
        skeletons.append(
            soft_skeletonize(stem_probability, skeleton_iterations)
            .ge(skeleton_threshold)
            .float()
        )
    persistence_map = torch.stack(skeletons, dim=0).mean(dim=0)
    stable_skeleton = persistence_map.ge(persistence).squeeze(1)
    stable_skeleton &= labels.eq(stem_class)

    reliable_stem = reliable & labels.eq(stem_class)
    stem_neighbourhood = _binary_dilate(
        reliable_stem.unsqueeze(1),
        boundary_radius,
    ).squeeze(1)
    uncertain_boundary = (
        labels.eq(stem_class)
        & ~reliable_stem
        & stem_neighbourhood
    )
    reliable &= ~uncertain_boundary

    core_mask = build_core_mask(
        labels,
        reliable,
        num_classes=shape[1],
        stem_class=stem_class,
        stable_skeleton=stable_skeleton,
        strategy=core_strategy,
        erode_iterations=core_erode_iterations,
        stem_radius=core_stem_radius,
        skeleton_iterations=skeleton_iterations,
    )

    return {
        "labels": labels,
        "probabilities": mean_prob,
        "weights": weights,
        "reliable": reliable,
        "uncertainty": uncertainty,
        "stable_skeleton": stable_skeleton,
        "uncertain_boundary": uncertain_boundary,
        "core_mask": core_mask,
    }


def masked_nll_loss(probabilities, labels, weights, valid, eps=1e-6):
    log_probabilities = probabilities.clamp_min(eps).log()
    per_pixel = F.nll_loss(log_probabilities, labels, reduction="none")
    pixel_weights = weights.float() * valid.float()
    return (per_pixel * pixel_weights).sum() / pixel_weights.sum().clamp_min(1.0)


def masked_dice_loss(
    probabilities,
    labels,
    weights,
    valid,
    num_classes,
    eps=1.0,
):
    one_hot = F.one_hot(
        labels.clamp(0, num_classes - 1),
        num_classes=num_classes,
    ).permute(0, 3, 1, 2).to(probabilities)
    pixel_weights = (weights * valid.float()).unsqueeze(1)
    intersection = (probabilities * one_hot * pixel_weights).sum(dim=(0, 2, 3))
    denominator = (
        (probabilities + one_hot) * pixel_weights
    ).sum(dim=(0, 2, 3))
    present = (one_hot * pixel_weights).sum(dim=(0, 2, 3)).gt(0)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    if present.any():
        return 1.0 - dice[present].mean()
    return probabilities.sum() * 0.0


def soft_cldice_loss(
    prediction,
    target,
    valid=None,
    iterations=20,
    eps=1.0,
):
    prediction = prediction.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    if valid is not None:
        valid = valid.float()
        prediction = prediction * valid
        target = target * valid

    prediction_skeleton = soft_skeletonize(prediction, iterations)
    target_skeleton = soft_skeletonize(target, iterations)
    topology_precision = (
        (prediction_skeleton * target).sum(dim=(1, 2, 3)) + eps
    ) / (prediction_skeleton.sum(dim=(1, 2, 3)) + eps)
    topology_sensitivity = (
        (target_skeleton * prediction).sum(dim=(1, 2, 3)) + eps
    ) / (target_skeleton.sum(dim=(1, 2, 3)) + eps)
    cldice = (
        2.0
        * topology_precision
        * topology_sensitivity
        / (topology_precision + topology_sensitivity).clamp_min(1e-6)
    )
    return 1.0 - cldice.mean()
