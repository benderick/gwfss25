import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def tcpm_curriculum_state(
    current_iter,
    start_iter,
    memory_warmup_iters,
    loss_ramp_iters,
    pseudo_update_start,
    pseudo_ramp_iters,
    pseudo_blend_max,
):
    current_iter = int(current_iter)
    start_iter = int(start_iter)
    loss_start = start_iter + max(int(memory_warmup_iters), 0)
    loss_ramp = max(int(loss_ramp_iters), 1)
    loss_scale = min(max((current_iter - loss_start) / loss_ramp, 0.0), 1.0)

    pseudo_start = int(pseudo_update_start)
    if pseudo_start < 0:
        pseudo_start = loss_start + loss_ramp
    pseudo_ramp = max(int(pseudo_ramp_iters), 1)
    pseudo_scale = min(max((current_iter - pseudo_start) / pseudo_ramp, 0.0), 1.0)
    active = current_iter >= start_iter
    return {
        "active": active,
        "loss_scale": loss_scale,
        "pseudo_update": active and current_iter >= pseudo_start,
        "pseudo_blend": pseudo_scale * float(pseudo_blend_max),
    }


class TopologyCorePrototypeMemory(nn.Module):
    """Curriculum-stabilized domain prototypes built from topology cores."""

    def __init__(
        self,
        num_domains,
        num_classes,
        feature_dim,
        labeled_momentum=0.95,
        pseudo_momentum=0.995,
        temperature=0.1,
        max_samples_per_class=256,
        min_core_pixels=8,
        hard_negative_margin=0.1,
        stem_class=2,
        leaf_class=3,
    ):
        super().__init__()
        self.num_domains = int(num_domains)
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.labeled_momentum = float(labeled_momentum)
        self.pseudo_momentum = float(pseudo_momentum)
        self.temperature = float(temperature)
        self.max_samples_per_class = int(max_samples_per_class)
        self.min_core_pixels = int(min_core_pixels)
        self.hard_negative_margin = float(hard_negative_margin)
        self.stem_class = int(stem_class)
        self.leaf_class = int(leaf_class)

        shape = (self.num_domains, self.num_classes, self.feature_dim)
        flags = (self.num_domains, self.num_classes)
        self.register_buffer("labeled_prototypes", torch.zeros(shape))
        self.register_buffer("labeled_initialized", torch.zeros(flags, dtype=torch.bool))
        self.register_buffer("pseudo_prototypes", torch.zeros(shape))
        self.register_buffer("pseudo_initialized", torch.zeros(flags, dtype=torch.bool))
        self.register_buffer("memory_started", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("last_labeled_drift", torch.tensor(0.0))
        self.register_buffer("last_pseudo_drift", torch.tensor(0.0))

    @torch.no_grad()
    def reset_memory(self):
        self.labeled_prototypes.zero_()
        self.labeled_initialized.zero_()
        self.pseudo_prototypes.zero_()
        self.pseudo_initialized.zero_()
        self.last_labeled_drift.zero_()
        self.last_pseudo_drift.zero_()
        self.memory_started.fill_(True)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Old experimental TCPM checkpoints used one bank. Treat it as the
        # labelled bank so a resumed run does not silently discard it.
        old_prototypes = state_dict.pop(prefix + "prototypes", None)
        old_initialized = state_dict.pop(prefix + "initialized", None)
        if old_prototypes is not None:
            state_dict.setdefault(prefix + "labeled_prototypes", old_prototypes)
        if old_initialized is not None:
            state_dict.setdefault(prefix + "labeled_initialized", old_initialized)
        defaults = {
            "labeled_prototypes": self.labeled_prototypes,
            "labeled_initialized": self.labeled_initialized,
            "pseudo_prototypes": self.pseudo_prototypes,
            "pseudo_initialized": self.pseudo_initialized,
            "memory_started": self.memory_started,
            "last_labeled_drift": self.last_labeled_drift,
            "last_pseudo_drift": self.last_pseudo_drift,
        }
        for name, value in defaults.items():
            state_dict.setdefault(prefix + name, value.detach().clone())
        if old_initialized is not None and prefix + "memory_started" in state_dict:
            state_dict[prefix + "memory_started"] = old_initialized.any().to(
                dtype=torch.bool
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _aggregate_bank(self, prototypes, initialized, excluded_domain=None):
        valid = initialized.clone()
        if excluded_domain is not None:
            valid[int(excluded_domain) % self.num_domains] = False
        weights = valid.to(dtype=prototypes.dtype).unsqueeze(-1)
        counts = weights.sum(dim=0)
        aggregated = (prototypes * weights).sum(dim=0) / counts.clamp_min(1.0)
        aggregated = F.normalize(aggregated, dim=-1)
        return aggregated, counts.squeeze(-1).gt(0)

    def _reference_prototypes(self, excluded_domain=None, pseudo_blend=0.0):
        labeled, labeled_available = self._aggregate_bank(
            self.labeled_prototypes,
            self.labeled_initialized,
            excluded_domain,
        )
        pseudo, pseudo_available = self._aggregate_bank(
            self.pseudo_prototypes,
            self.pseudo_initialized,
            excluded_domain,
        )
        blend = min(max(float(pseudo_blend), 0.0), 1.0)
        references = labeled.clone()
        both = labeled_available & pseudo_available
        if both.any() and blend > 0.0:
            references[both] = F.normalize(
                (1.0 - blend) * labeled[both] + blend * pseudo[both],
                dim=-1,
            )
        return references, labeled_available

    @torch.no_grad()
    def _update_bank(self, sums, counts, source):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums)
            dist.all_reduce(counts)

        if source == "labeled":
            bank = self.labeled_prototypes
            initialized = self.labeled_initialized
            momentum = self.labeled_momentum
        elif source == "pseudo":
            bank = self.pseudo_prototypes
            initialized = self.pseudo_initialized
            momentum = self.pseudo_momentum
        else:
            raise ValueError("TCPM source must be 'labeled' or 'pseudo'")

        drift_terms = []
        for domain_id in range(self.num_domains):
            for class_id in range(self.num_classes):
                if counts[domain_id, class_id] <= 0:
                    continue
                prototype = F.normalize(
                    sums[domain_id, class_id] / counts[domain_id, class_id],
                    dim=0,
                )
                if initialized[domain_id, class_id]:
                    previous = bank[domain_id, class_id].clone()
                    prototype = F.normalize(
                        momentum * bank[domain_id, class_id]
                        + (1.0 - momentum) * prototype,
                        dim=0,
                    )
                    drift_terms.append(
                        1.0 - F.cosine_similarity(
                            previous.unsqueeze(0), prototype.unsqueeze(0)
                        ).mean()
                    )
                bank[domain_id, class_id].copy_(prototype)
                initialized[domain_id, class_id] = True
        drift = torch.stack(drift_terms).mean() if drift_terms else bank.new_zeros(())
        if source == "labeled":
            self.last_labeled_drift.copy_(drift)
        else:
            self.last_pseudo_drift.copy_(drift)

    def _spread_indices(self, indices):
        if indices.shape[0] <= self.max_samples_per_class:
            return indices
        positions = torch.div(
            torch.arange(
                self.max_samples_per_class,
                device=indices.device,
            )
            * indices.shape[0],
            self.max_samples_per_class,
            rounding_mode="floor",
        )
        return indices[positions]

    def forward(
        self,
        features,
        labels,
        weights,
        core_mask,
        domain_ids,
        source="labeled",
        update_memory=True,
        pseudo_blend=0.0,
        leave_one_domain_out=True,
    ):
        if features.ndim != 4:
            raise ValueError("TCPM features must have shape [N, C, H, W]")
        if source not in {"labeled", "pseudo"}:
            raise ValueError("TCPM source must be 'labeled' or 'pseudo'")

        feature_size = features.shape[-2:]
        labels = F.interpolate(
            labels.unsqueeze(1).float(), size=feature_size, mode="nearest"
        ).squeeze(1).long()
        weights = F.interpolate(
            weights.unsqueeze(1).float(), size=feature_size, mode="nearest"
        ).squeeze(1)
        core_mask = F.interpolate(
            core_mask.unsqueeze(1).float(), size=feature_size, mode="nearest"
        ).squeeze(1).bool()
        features = F.normalize(features.float(), dim=1)

        sums = features.new_zeros(
            self.num_domains, self.num_classes, self.feature_dim
        )
        counts = features.new_zeros(self.num_domains, self.num_classes)
        centroids = []

        for batch_index in range(features.shape[0]):
            domain_id = int(domain_ids[batch_index]) % self.num_domains
            feature_map = features[batch_index].permute(1, 2, 0)
            for class_id in range(self.num_classes):
                selected = core_mask[batch_index] & labels[batch_index].eq(class_id)
                indices = selected.nonzero(as_tuple=False)
                if indices.shape[0] < self.min_core_pixels:
                    continue
                indices = self._spread_indices(indices)
                class_features = feature_map[indices[:, 0], indices[:, 1]]
                class_weights = weights[batch_index, indices[:, 0], indices[:, 1]]
                class_weights = class_weights.clamp_min(1e-6)
                weighted_sum = (class_features * class_weights.unsqueeze(1)).sum(0)
                weight_sum = class_weights.sum()
                centroid = F.normalize(weighted_sum / weight_sum.clamp_min(1e-6), dim=0)
                sums[domain_id, class_id] += centroid.detach()
                counts[domain_id, class_id] += 1.0
                centroids.append((domain_id, class_id, centroid))

        if update_memory:
            self._update_bank(sums, counts, source)

        zero = features.sum() * 0.0
        contrastive_terms = []
        compact_terms = []
        hard_negative_terms = []
        reference_cache = {}

        for domain_id, class_id, centroid in centroids:
            excluded_domain = domain_id if leave_one_domain_out else None
            cache_key = excluded_domain if excluded_domain is not None else -1
            if cache_key not in reference_cache:
                reference_cache[cache_key] = self._reference_prototypes(
                    excluded_domain=excluded_domain,
                    pseudo_blend=pseudo_blend,
                )
            references, available = reference_cache[cache_key]

            available_classes = available.nonzero(as_tuple=False).flatten()
            if available[class_id] and available_classes.numel() >= 2:
                logits = centroid @ references[available_classes].t()
                target = available_classes.eq(class_id).nonzero(as_tuple=False)
                if target.numel() > 0:
                    contrastive_terms.append(
                        F.cross_entropy(
                            (logits / self.temperature).unsqueeze(0),
                            target.flatten()[:1],
                        )
                    )
            if available[class_id]:
                compact_terms.append(
                    1.0
                    - F.cosine_similarity(
                        centroid.unsqueeze(0),
                        references[class_id].unsqueeze(0),
                    ).mean()
                )

            if available[self.stem_class] and available[self.leaf_class]:
                if class_id == self.stem_class:
                    positive = references[self.stem_class]
                    negative = references[self.leaf_class]
                elif class_id == self.leaf_class:
                    positive = references[self.leaf_class]
                    negative = references[self.stem_class]
                else:
                    continue
                positive_similarity = F.cosine_similarity(
                    centroid.unsqueeze(0), positive.unsqueeze(0)
                )
                negative_similarity = F.cosine_similarity(
                    centroid.unsqueeze(0), negative.unsqueeze(0)
                )
                hard_negative_terms.append(
                    F.relu(
                        negative_similarity
                        - positive_similarity
                        + self.hard_negative_margin
                    ).mean()
                )

        def mean_or_zero(terms):
            return torch.stack(terms).mean() if terms else zero

        return {
            "contrastive": mean_or_zero(contrastive_terms),
            "domain_compact": mean_or_zero(compact_terms),
            "hard_negative": mean_or_zero(hard_negative_terms),
        }
