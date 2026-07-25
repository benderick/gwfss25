import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class TopologyCorePrototypeMemory(nn.Module):
    """Domain-aware class prototypes updated only from reliable organ cores."""

    def __init__(
        self,
        num_domains,
        num_classes,
        feature_dim,
        momentum=0.99,
        temperature=0.1,
        max_samples_per_class=256,
        hard_negative_margin=0.2,
        stem_class=2,
        leaf_class=3,
    ):
        super().__init__()
        self.num_domains = int(num_domains)
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.max_samples_per_class = int(max_samples_per_class)
        self.hard_negative_margin = float(hard_negative_margin)
        self.stem_class = int(stem_class)
        self.leaf_class = int(leaf_class)

        self.register_buffer(
            "prototypes",
            torch.zeros(num_domains, num_classes, feature_dim),
        )
        self.register_buffer(
            "initialized",
            torch.zeros(num_domains, num_classes, dtype=torch.bool),
        )

    def _global_prototypes(self, heldout_domain):
        valid = self.initialized.clone()
        if heldout_domain is not None:
            valid[int(heldout_domain) % self.num_domains] = False
        weights = valid.float().unsqueeze(-1)
        summed = (self.prototypes * weights).sum(dim=0)
        counts = weights.sum(dim=0)
        global_prototypes = summed / counts.clamp_min(1.0)
        global_prototypes = F.normalize(global_prototypes, dim=-1)
        return global_prototypes, counts.squeeze(-1).gt(0)

    @torch.no_grad()
    def _update(self, sums, counts, heldout_domain):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums)
            dist.all_reduce(counts)

        heldout = None
        if heldout_domain is not None:
            heldout = int(heldout_domain) % self.num_domains
        for domain_id in range(self.num_domains):
            if domain_id == heldout:
                continue
            for class_id in range(self.num_classes):
                count = counts[domain_id, class_id]
                if count <= 0:
                    continue
                prototype = sums[domain_id, class_id] / count
                prototype = F.normalize(prototype, dim=0)
                if self.initialized[domain_id, class_id]:
                    updated = (
                        self.momentum * self.prototypes[domain_id, class_id]
                        + (1.0 - self.momentum) * prototype
                    )
                    self.prototypes[domain_id, class_id].copy_(
                        F.normalize(updated, dim=0)
                    )
                else:
                    self.prototypes[domain_id, class_id].copy_(prototype)
                    self.initialized[domain_id, class_id] = True

    def forward(
        self,
        features,
        labels,
        weights,
        core_mask,
        domain_ids,
        heldout_domain=None,
        update_memory=True,
    ):
        if features.ndim != 4:
            raise ValueError("TCPM features must have shape [N, C, H, W]")
        feature_size = features.shape[-2:]
        labels = F.interpolate(
            labels.unsqueeze(1).float(),
            size=feature_size,
            mode="nearest",
        ).squeeze(1).long()
        weights = F.interpolate(
            weights.unsqueeze(1).float(),
            size=feature_size,
            mode="nearest",
        ).squeeze(1)
        core_mask = F.interpolate(
            core_mask.unsqueeze(1).float(),
            size=feature_size,
            mode="nearest",
        ).squeeze(1).bool()
        features = F.normalize(features.float(), dim=1)

        sums = features.new_zeros(
            self.num_domains,
            self.num_classes,
            self.feature_dim,
        )
        counts = features.new_zeros(self.num_domains, self.num_classes)
        selected_features = []
        selected_labels = []
        batch_prototypes = []

        for batch_index in range(features.shape[0]):
            domain_id = int(domain_ids[batch_index]) % self.num_domains
            feature_map = features[batch_index].permute(1, 2, 0)
            for class_id in range(self.num_classes):
                selected = core_mask[batch_index] & labels[batch_index].eq(class_id)
                indices = selected.nonzero(as_tuple=False)
                if indices.numel() == 0:
                    continue
                if indices.shape[0] > self.max_samples_per_class:
                    indices = indices[: self.max_samples_per_class]
                class_features = feature_map[indices[:, 0], indices[:, 1]]
                class_weights = weights[batch_index, indices[:, 0], indices[:, 1]]
                class_weights = class_weights.clamp_min(1e-6)
                weighted_sum = (class_features * class_weights.unsqueeze(1)).sum(dim=0)
                weight_sum = class_weights.sum()
                sums[domain_id, class_id] += weighted_sum.detach()
                counts[domain_id, class_id] += weight_sum.detach()
                selected_features.append(class_features)
                selected_labels.append(
                    torch.full(
                        (class_features.shape[0],),
                        class_id,
                        device=features.device,
                        dtype=torch.long,
                    )
                )
                batch_prototypes.append(
                    (class_id, weighted_sum / weight_sum.clamp_min(1e-6))
                )

        if update_memory:
            self._update(sums, counts, heldout_domain)

        global_prototypes, available = self._global_prototypes(heldout_domain)
        zero = features.sum() * 0.0
        if not selected_features or available.sum() < 2:
            return {
                "contrastive": zero,
                "domain_compact": zero,
                "hard_negative": zero,
            }

        available_classes = available.nonzero(as_tuple=False).flatten()
        class_remap = labels.new_full((self.num_classes,), -1)
        class_remap[available_classes] = torch.arange(
            available_classes.numel(),
            device=labels.device,
        )
        feature_samples = torch.cat(selected_features, dim=0)
        label_samples = torch.cat(selected_labels, dim=0)
        keep = class_remap[label_samples].ge(0)
        if keep.any():
            logits = (
                feature_samples[keep]
                @ global_prototypes[available_classes].t()
                / self.temperature
            )
            contrastive = F.cross_entropy(
                logits,
                class_remap[label_samples[keep]],
            )
        else:
            contrastive = zero

        compact_terms = []
        for class_id, batch_prototype in batch_prototypes:
            if available[class_id]:
                compact_terms.append(
                    1.0
                    - F.cosine_similarity(
                        batch_prototype.unsqueeze(0),
                        global_prototypes[class_id].unsqueeze(0),
                    ).mean()
                )
        domain_compact = (
            torch.stack(compact_terms).mean() if compact_terms else zero
        )

        hard_negative_terms = []
        if available[self.stem_class] and available[self.leaf_class]:
            stem_global = global_prototypes[self.stem_class]
            leaf_global = global_prototypes[self.leaf_class]
            for class_id, batch_prototype in batch_prototypes:
                if class_id == self.stem_class:
                    positive = F.cosine_similarity(
                        batch_prototype.unsqueeze(0),
                        stem_global.unsqueeze(0),
                    )
                    negative = F.cosine_similarity(
                        batch_prototype.unsqueeze(0),
                        leaf_global.unsqueeze(0),
                    )
                elif class_id == self.leaf_class:
                    positive = F.cosine_similarity(
                        batch_prototype.unsqueeze(0),
                        leaf_global.unsqueeze(0),
                    )
                    negative = F.cosine_similarity(
                        batch_prototype.unsqueeze(0),
                        stem_global.unsqueeze(0),
                    )
                else:
                    continue
                hard_negative_terms.append(
                    F.relu(negative - positive + self.hard_negative_margin).mean()
                )
        hard_negative = (
            torch.stack(hard_negative_terms).mean()
            if hard_negative_terms
            else zero
        )
        return {
            "contrastive": contrastive,
            "domain_compact": domain_compact,
            "hard_negative": hard_negative,
        }
