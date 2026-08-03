"""Configuration-aligned feature-statistic intervention for CARE-Wheat."""

import torch


def interpolate_feature_statistics(
    features,
    donor_mean,
    donor_std,
    compatibility_weight,
    eps=1.0e-6,
    return_statistics=False,
):
    """Preserve spatial activations while interpolating channel statistics.

    Args:
        features: Tensor with shape ``B x C x H x W``.
        donor_mean: Frozen-teacher donor means with shape ``B x C``.
        donor_std: Frozen-teacher donor standard deviations with shape ``B x C``.
        compatibility_weight: Per-image interpolation weights in ``[0, 1]``.
    """
    if features.ndim != 4:
        raise ValueError("CARE features must have shape BxCxHxW")

    batch_size, channels = features.shape[:2]
    donor_mean = donor_mean.reshape(batch_size, channels, 1, 1)
    donor_std = donor_std.reshape(batch_size, channels, 1, 1)
    compatibility_weight = compatibility_weight.reshape(batch_size, 1, 1, 1)

    if not torch.isfinite(donor_mean).all():
        raise ValueError("CARE donor means contain non-finite values")
    if not torch.isfinite(donor_std).all() or (donor_std < 0).any():
        raise ValueError("CARE donor standard deviations are invalid")
    if (
        not torch.isfinite(compatibility_weight).all()
        or (compatibility_weight < 0).any()
        or (compatibility_weight > 1).any()
    ):
        raise ValueError("CARE compatibility weights must be in [0, 1]")

    original_dtype = features.dtype
    features_float = features.float()
    donor_mean = donor_mean.to(device=features.device, dtype=torch.float32)
    donor_std = donor_std.to(device=features.device, dtype=torch.float32)
    weight = compatibility_weight.to(device=features.device, dtype=torch.float32)

    source_mean = features_float.mean(dim=(2, 3), keepdim=True)
    source_variance = (
        (features_float - source_mean).square().mean(dim=(2, 3), keepdim=True)
    )
    source_std = (source_variance + eps).sqrt()

    target_mean = source_mean.lerp(donor_mean, weight)
    target_std = source_std.lerp(donor_std.clamp_min(eps), weight)
    normalized = (features_float - source_mean) / source_std
    mixed = normalized * target_std + target_mean

    # Unsupported anchors carry weight zero. Keep their activations bitwise
    # unchanged instead of relying on normalize-denormalize roundoff.
    mixed = torch.where(weight > 0, mixed, features_float)
    mixed = mixed.to(dtype=original_dtype)
    if not return_statistics:
        return mixed

    statistics = {
        "normalized_mean_shift": (
            (target_mean - source_mean).abs() / source_std
        ).flatten(1),
        "target_to_source_std_ratio": (target_std / source_std).flatten(1),
    }
    return mixed, statistics
