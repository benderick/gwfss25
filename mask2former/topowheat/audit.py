"""Streaming metrics used to audit TRPL pseudo-label quality."""

from collections import OrderedDict

import torch
import torch.nn.functional as F


def _ratio(numerator, denominator):
    denominator = float(denominator)
    if denominator <= 0.0:
        return None
    return float(numerator) / denominator


def _harmonic_mean(first, second):
    if first is None or second is None or first + second <= 0.0:
        return None
    return 2.0 * first * second / (first + second)


def matched_topk_mask(scores, reference_mask, valid=None):
    """Select exactly as many valid pixels as ``reference_mask`` selects."""
    if scores.shape != reference_mask.shape:
        raise ValueError("scores and reference_mask must have identical shapes")
    if valid is None:
        valid = torch.ones_like(reference_mask, dtype=torch.bool)
    if valid.shape != scores.shape:
        raise ValueError("valid must have the same shape as scores")

    valid = valid.bool()
    reference_mask = reference_mask.bool() & valid
    selected_count = int(reference_mask.sum().item())
    result = torch.zeros_like(valid)
    if selected_count <= 0:
        return result

    valid_indices = valid.flatten().nonzero(as_tuple=False).flatten()
    if selected_count >= int(valid_indices.numel()):
        return valid.clone()
    valid_scores = scores.flatten()[valid_indices]
    chosen = torch.topk(
        valid_scores,
        k=selected_count,
        largest=True,
        sorted=False,
    ).indices
    result.flatten()[valid_indices[chosen]] = True
    return result


def matched_topk_mask_by_class(
    scores,
    labels,
    reference_mask,
    num_classes,
    valid=None,
):
    """Match reference coverage independently inside every predicted class."""
    if not (scores.shape == labels.shape == reference_mask.shape):
        raise ValueError("scores, labels and reference_mask must match")
    if valid is None:
        valid = torch.ones_like(reference_mask, dtype=torch.bool)
    if valid.shape != scores.shape:
        raise ValueError("valid must have the same shape as scores")
    result = torch.zeros_like(reference_mask, dtype=torch.bool)
    for class_id in range(int(num_classes)):
        class_valid = valid.bool() & labels.eq(class_id)
        result |= matched_topk_mask(
            scores,
            reference_mask.bool() & class_valid,
            class_valid,
        )
    return result


def binary_dilate(mask, radius=1):
    """Dilate a binary mask without requiring OpenCV or SciPy."""
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    original_shape = mask.shape
    if mask.ndim == 2:
        mask_4d = mask[None, None]
    elif mask.ndim == 3:
        mask_4d = mask[:, None]
    elif mask.ndim == 4 and mask.shape[1] == 1:
        mask_4d = mask
    else:
        raise ValueError("mask must have shape [H, W], [N, H, W], or [N, 1, H, W]")
    if radius == 0:
        return mask.bool().clone()
    dilated = F.max_pool2d(
        mask_4d.float(),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    ).gt(0.5)
    return dilated.reshape(original_shape)


class SelectiveSegmentationAccumulator:
    """Accumulate segmentation quality when only selected pixels are trusted."""

    def __init__(self, num_classes, class_names=None, ignore_label=255):
        self.num_classes = int(num_classes)
        self.class_names = list(class_names or range(self.num_classes))
        if len(self.class_names) != self.num_classes:
            raise ValueError("class_names must contain one name per class")
        self.ignore_label = int(ignore_label)
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
        )
        self.gt_count = torch.zeros(self.num_classes, dtype=torch.int64)
        self.valid_count = 0
        self.accepted_count = 0

    def update(self, prediction, target, selected=None):
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have identical shapes")
        prediction = prediction.detach().to(device="cpu", dtype=torch.long)
        target = target.detach().to(device="cpu", dtype=torch.long)
        valid = (
            target.ne(self.ignore_label)
            & target.ge(0)
            & target.lt(self.num_classes)
        )
        if selected is None:
            selected = valid
        else:
            if selected.shape != target.shape:
                raise ValueError("selected and target must have identical shapes")
            selected = selected.detach().to(device="cpu", dtype=torch.bool) & valid

        valid_prediction = prediction[valid]
        if valid_prediction.numel() and (
            valid_prediction.lt(0).any()
            or valid_prediction.ge(self.num_classes).any()
        ):
            raise ValueError("prediction contains an out-of-range class id")

        self.valid_count += int(valid.sum().item())
        self.accepted_count += int(selected.sum().item())
        self.gt_count += torch.bincount(
            target[valid],
            minlength=self.num_classes,
        )
        if selected.any():
            encoded = (
                target[selected] * self.num_classes
                + prediction[selected]
            )
            self.confusion += torch.bincount(
                encoded,
                minlength=self.num_classes * self.num_classes,
            ).reshape(self.num_classes, self.num_classes)

    def summary(self):
        true_positive = self.confusion.diag()
        predicted_count = self.confusion.sum(dim=0)
        per_class = OrderedDict()
        ious = []
        for class_id, class_name in enumerate(self.class_names):
            tp = int(true_positive[class_id].item())
            predicted = int(predicted_count[class_id].item())
            target = int(self.gt_count[class_id].item())
            union = predicted + target - tp
            precision = _ratio(tp, predicted)
            recall = _ratio(tp, target)
            iou = _ratio(tp, union)
            if iou is not None:
                ious.append(iou)
            per_class[str(class_name)] = {
                "class_id": class_id,
                "true_positive": tp,
                "selected_prediction_count": predicted,
                "target_count": target,
                "precision": precision,
                "recall": recall,
                "iou": iou,
            }

        correct = int(true_positive.sum().item())
        return {
            "valid_pixels": self.valid_count,
            "accepted_pixels": self.accepted_count,
            "coverage": _ratio(self.accepted_count, self.valid_count),
            "accepted_accuracy": _ratio(correct, self.accepted_count),
            "mean_iou": sum(ious) / len(ious) if ious else None,
            "per_class": per_class,
        }


class CalibrationAccumulator:
    """Streaming expected calibration error and Brier score."""

    def __init__(self, num_bins=15):
        self.num_bins = int(num_bins)
        if self.num_bins <= 0:
            raise ValueError("num_bins must be positive")
        self.counts = torch.zeros(self.num_bins, dtype=torch.int64)
        self.score_sums = torch.zeros(self.num_bins, dtype=torch.float64)
        self.correct_sums = torch.zeros(self.num_bins, dtype=torch.float64)
        self.squared_error_sum = 0.0

    def update(self, scores, correct, valid=None):
        if scores.shape != correct.shape:
            raise ValueError("scores and correct must have identical shapes")
        scores = scores.detach().to(device="cpu", dtype=torch.float64)
        correct = correct.detach().to(device="cpu", dtype=torch.bool)
        if valid is None:
            valid = torch.ones_like(correct)
        else:
            if valid.shape != scores.shape:
                raise ValueError("valid and scores must have identical shapes")
            valid = valid.detach().to(device="cpu", dtype=torch.bool)
        scores = scores[valid].clamp(0.0, 1.0)
        correct_float = correct[valid].to(torch.float64)
        if not scores.numel():
            return

        indices = torch.floor(scores * self.num_bins).long().clamp_max(
            self.num_bins - 1
        )
        self.counts += torch.bincount(indices, minlength=self.num_bins)
        self.score_sums.scatter_add_(0, indices, scores)
        self.correct_sums.scatter_add_(0, indices, correct_float)
        self.squared_error_sum += float(
            ((scores - correct_float) ** 2).sum().item()
        )

    def summary(self):
        total = int(self.counts.sum().item())
        ece = 0.0
        bins = []
        for index in range(self.num_bins):
            count = int(self.counts[index].item())
            mean_score = _ratio(self.score_sums[index].item(), count)
            accuracy = _ratio(self.correct_sums[index].item(), count)
            if count and total:
                ece += (count / float(total)) * abs(accuracy - mean_score)
            bins.append(
                {
                    "lower": index / float(self.num_bins),
                    "upper": (index + 1) / float(self.num_bins),
                    "count": count,
                    "mean_score": mean_score,
                    "accuracy": accuracy,
                }
            )
        return {
            "pixels": total,
            "mean_score": _ratio(self.score_sums.sum().item(), total),
            "accuracy": _ratio(self.correct_sums.sum().item(), total),
            "ece": ece if total else None,
            "brier": _ratio(self.squared_error_sum, total),
            "bins": bins,
        }


class TopologyAlignmentAccumulator:
    """Measure precision and sensitivity of a candidate stem centreline."""

    def __init__(self, tolerance=1):
        self.tolerance = int(tolerance)
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.candidate_pixels = 0
        self.candidate_hits = 0
        self.target_skeleton_pixels = 0
        self.target_hits = 0

    def update(self, candidate_skeleton, target_stem, target_skeleton):
        if not (
            candidate_skeleton.shape
            == target_stem.shape
            == target_skeleton.shape
        ):
            raise ValueError("all topology masks must have identical shapes")
        candidate = candidate_skeleton.detach().bool()
        target_stem = target_stem.detach().bool()
        target_skeleton = target_skeleton.detach().bool()
        candidate_support = binary_dilate(candidate, self.tolerance)
        target_support = binary_dilate(target_stem, self.tolerance)

        self.candidate_pixels += int(candidate.sum().item())
        self.candidate_hits += int((candidate & target_support).sum().item())
        self.target_skeleton_pixels += int(target_skeleton.sum().item())
        self.target_hits += int(
            (target_skeleton & candidate_support).sum().item()
        )

    def summary(self):
        precision = _ratio(self.candidate_hits, self.candidate_pixels)
        sensitivity = _ratio(self.target_hits, self.target_skeleton_pixels)
        return {
            "candidate_pixels": self.candidate_pixels,
            "candidate_hits": self.candidate_hits,
            "target_skeleton_pixels": self.target_skeleton_pixels,
            "target_hits": self.target_hits,
            "precision": precision,
            "sensitivity": sensitivity,
            "cldice": _harmonic_mean(precision, sensitivity),
        }
