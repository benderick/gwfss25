import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.data.detection_utils import read_image

from .topology import endpoint_map


def _normalized_probabilities(scores, eps=1e-6):
    scores = scores.float().clamp_min(0.0)
    return scores / scores.sum(dim=0, keepdim=True).clamp_min(eps)


def fusion_gate_features(global_probabilities, local_probabilities):
    """Build scale-independent evidence for learned global/local fusion."""
    if global_probabilities.shape != local_probabilities.shape:
        raise ValueError("Global and local probabilities must have equal shapes")
    if global_probabilities.ndim != 4:
        raise ValueError("Fusion probabilities must have shape [N, C, H, W]")
    global_entropy = -(
        global_probabilities.clamp_min(1e-6)
        * global_probabilities.clamp_min(1e-6).log()
    ).sum(dim=1, keepdim=True)
    local_entropy = -(
        local_probabilities.clamp_min(1e-6)
        * local_probabilities.clamp_min(1e-6).log()
    ).sum(dim=1, keepdim=True)
    return torch.cat(
        [
            global_probabilities,
            local_probabilities,
            (local_probabilities - global_probabilities).abs(),
            global_entropy,
            local_entropy,
        ],
        dim=1,
    )


class BAZRFusionGate(nn.Module):
    """Predict where a local high-resolution result should replace the global one."""

    def __init__(self, num_classes, hidden_dim=16):
        super().__init__()
        input_dim = 3 * int(num_classes) + 2
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, int(hidden_dim), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(hidden_dim), 1, kernel_size=1),
        )
        nn.init.constant_(self.layers[-1].bias, -1.0)

    def forward(self, global_probabilities, local_probabilities):
        features = fusion_gate_features(
            global_probabilities,
            local_probabilities,
        )
        return torch.sigmoid(self.layers(features))


def _window_iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1)


class BrokenAwareZoomRefiner(nn.Module):
    """Allocate high-resolution inference to windows with structural risk."""

    def __init__(self, cfg, model):
        super().__init__()
        self.model = model
        bazr = cfg.MODEL.TOPOWHEAT.BAZR
        self.stem_class = int(cfg.MODEL.TOPOWHEAT.STEM_CLASS)
        self.topk = int(bazr.TOPK)
        self.window_size = int(bazr.WINDOW_SIZE)
        self.window_stride = int(bazr.WINDOW_STRIDE)
        self.zoom_size = int(bazr.ZOOM_SIZE)
        self.nms_threshold = float(bazr.NMS_THRESHOLD)
        self.entropy_weight = float(bazr.ENTROPY_WEIGHT)
        self.endpoint_weight = float(bazr.ENDPOINT_WEIGHT)
        self.disagreement_weight = float(bazr.DISAGREEMENT_WEIGHT)
        self.stem_weight = float(bazr.STEM_WEIGHT)
        self.gate_slope = float(bazr.GATE_SLOPE)
        self.gate_margin = float(bazr.GATE_MARGIN)
        self.skeleton_iterations = int(bazr.SKELETON_ITERATIONS)
        self.return_diagnostics = bool(bazr.RETURN_DIAGNOSTICS)

    @property
    def device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _ensure_image(self, sample):
        if "image" in sample:
            return sample
        sample = copy.copy(sample)
        image = read_image(sample["file_name"], "RGB")
        sample["image"] = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        sample.setdefault("height", image.shape[0])
        sample.setdefault("width", image.shape[1])
        return sample

    def _risk_maps(self, result):
        global_scores = result["sem_seg"]
        probabilities = _normalized_probabilities(global_scores)
        entropy = -(
            probabilities.clamp_min(1e-6)
            * probabilities.clamp_min(1e-6).log()
        ).sum(dim=0)
        entropy /= np.log(probabilities.shape[0])

        stem_mask = probabilities.argmax(dim=0).eq(self.stem_class)
        endpoints = endpoint_map(
            stem_mask.unsqueeze(0).unsqueeze(0),
            skeleton_iterations=self.skeleton_iterations,
            dilation=7,
        ).squeeze(0).squeeze(0)

        high = result.pop("_bazr_aux_high", None)
        low = result.pop("_bazr_aux_low", None)
        if high is not None and low is not None:
            high_prob = F.softmax(high.float(), dim=0)[self.stem_class]
            low_prob = F.softmax(low.float(), dim=0)[self.stem_class]
            disagreement = (high_prob - low_prob).abs()
        else:
            stem = probabilities[self.stem_class].unsqueeze(0).unsqueeze(0)
            pooled = F.avg_pool2d(stem, kernel_size=5, stride=1, padding=2)
            disagreement = (stem - pooled).abs().squeeze(0).squeeze(0)
        return probabilities, entropy, endpoints, disagreement

    def _select_windows(self, maps):
        probabilities, entropy, endpoints, disagreement = maps
        height, width = entropy.shape
        window_height = min(self.window_size, height)
        window_width = min(self.window_size, width)
        stride = max(self.window_stride, 1)
        candidates = []
        y_positions = list(range(0, max(height - window_height, 0) + 1, stride))
        x_positions = list(range(0, max(width - window_width, 0) + 1, stride))
        if not y_positions or y_positions[-1] != height - window_height:
            y_positions.append(height - window_height)
        if not x_positions or x_positions[-1] != width - window_width:
            x_positions.append(width - window_width)

        for y1 in y_positions:
            for x1 in x_positions:
                y2 = y1 + window_height
                x2 = x1 + window_width
                score = (
                    self.entropy_weight * entropy[y1:y2, x1:x2].mean()
                    + self.endpoint_weight * endpoints[y1:y2, x1:x2].mean()
                    + self.disagreement_weight
                    * disagreement[y1:y2, x1:x2].mean()
                    + self.stem_weight
                    * probabilities[self.stem_class, y1:y2, x1:x2].mean()
                )
                candidates.append((float(score.detach().cpu()), (x1, y1, x2, y2)))

        selected = []
        for score, window in sorted(candidates, key=lambda item: item[0], reverse=True):
            if all(_window_iou(window, kept[1]) <= self.nms_threshold for kept in selected):
                selected.append((score, window))
            if len(selected) >= self.topk:
                break
        return selected

    def _local_prediction(self, sample, window, output_shape):
        image = sample["image"]
        output_height, output_width = output_shape
        image_height, image_width = image.shape[-2:]
        x1, y1, x2, y2 = window
        image_x1 = int(round(x1 * image_width / output_width))
        image_x2 = int(round(x2 * image_width / output_width))
        image_y1 = int(round(y1 * image_height / output_height))
        image_y2 = int(round(y2 * image_height / output_height))
        crop = image[:, image_y1:image_y2, image_x1:image_x2]
        crop = F.interpolate(
            crop.unsqueeze(0).float(),
            size=(self.zoom_size, self.zoom_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        if not image.dtype.is_floating_point:
            crop = crop.round().clamp(0, 255).to(image.dtype)
        local_input = {
            "image": crop,
            "height": y2 - y1,
            "width": x2 - x1,
        }
        return self.model([local_input])[0]["sem_seg"]

    def _fuse(self, global_scores, local_scores, window):
        x1, y1, x2, y2 = window
        local_scores = F.interpolate(
            local_scores.unsqueeze(0),
            size=(y2 - y1, x2 - x1),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        global_window = global_scores[:, y1:y2, x1:x2]
        local_prob = _normalized_probabilities(local_scores)
        global_prob = _normalized_probabilities(global_window)
        gate_owner = getattr(self.model, "module", self.model)
        learned_gate = getattr(gate_owner, "bazr_gate", None)
        if learned_gate is not None:
            confidence_gate = learned_gate(
                global_prob.unsqueeze(0),
                local_prob.unsqueeze(0),
            ).squeeze(0).squeeze(0)
        else:
            confidence_gain = (
                local_prob.max(dim=0).values
                - global_prob.max(dim=0).values
                - self.gate_margin
            )
            confidence_gate = torch.sigmoid(
                self.gate_slope * confidence_gain
            )

        height, width = confidence_gate.shape
        window_y = torch.hann_window(
            height,
            periodic=False,
            device=confidence_gate.device,
            dtype=confidence_gate.dtype,
        )
        window_x = torch.hann_window(
            width,
            periodic=False,
            device=confidence_gate.device,
            dtype=confidence_gate.dtype,
        )
        spatial_gate = torch.outer(window_y, window_x)
        gate = (confidence_gate * spatial_gate).unsqueeze(0)
        global_scores[:, y1:y2, x1:x2] = (
            gate * local_prob + (1.0 - gate) * global_prob
        )

    def _refine_one(self, sample):
        sample = self._ensure_image(sample)
        result = self.model([sample])[0]
        global_scores = _normalized_probabilities(
            result["sem_seg"]
        ).clone()
        maps = self._risk_maps(result)
        selected = self._select_windows(maps)
        for _, window in selected:
            local_scores = self._local_prediction(
                sample,
                window,
                global_scores.shape[-2:],
            )
            self._fuse(global_scores, local_scores, window)
        result["sem_seg"] = global_scores
        if self.return_diagnostics:
            result["bazr_windows"] = [window for _, window in selected]
            result["bazr_scores"] = [score for score, _ in selected]
        return result

    def forward(self, batched_inputs):
        return [self._refine_one(sample) for sample in batched_inputs]
