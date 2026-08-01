# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import cv2
from copy import deepcopy
from collections import OrderedDict
import numpy as np

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom

import detectron2.utils.comm as comm

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .thresh_control import ThreshController
from .topowheat import (
    BAZRFusionGate,
    TopologyCorePrototypeMemory,
    build_core_mask,
    build_trpl_targets,
    centerline_recall_loss,
    masked_dice_loss,
    masked_nll_loss,
    masked_soft_cross_entropy,
    positive_probability_floor_loss,
    query_semantic_probabilities,
    soft_cldice_loss,
    tcpm_curriculum_state,
)
@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        cfg,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # ssl
        ssl_criterion: nn.Module,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.iter = 0
        self.cfg = cfg
        self.ssl_freq = self.cfg.SSL.FREQ
        self.burn_in = self.cfg.SSL.BURNIN_ITER
        configured_ema_start = int(self.cfg.SSL.EMA_UPDATE_START)
        self.ema_update_start = (
            self.burn_in if configured_ema_start < 0 else configured_ema_start
        )
        self.reset_teacher_at_start = bool(
            self.cfg.SSL.RESET_TEACHER_AT_START
        )
        self.ssl_diagnostic_period = int(self.cfg.SSL.DIAGNOSTIC_PERIOD)
        self.max_iter = self.cfg.SOLVER.MAX_ITER
        self.do_ssl = self.cfg.SSL.TRAIN_SSL
        self.ema_decay = self.cfg.SSL.EMA_DECAY
        self.ssl_criterion = ssl_criterion
        self.dropouts = nn.ModuleList([nn.Dropout(p=0.5) for _ in range(4)])
        self.img_size = self.backbone.model.img_size
        self.resize = transforms.Resize([self.img_size, self.img_size])
        self.thresh_controller = None
        self.thresh_class = 0.7
        if hasattr(self.cfg, 'THRESH_CONTROLLER_ON') and self.cfg.THRESH_CONTROLLER_ON:
            self.thresh_controller = ThreshController(momentum=0.999, thresh_init= self.thresh_class)

        topowheat = self.cfg.MODEL.TOPOWHEAT
        self.stem_class = int(topowheat.STEM_CLASS)
        self.leaf_class = int(topowheat.LEAF_CLASS)
        self.trpl_enabled = bool(topowheat.TRPL.ENABLED)
        self.tcpm_enabled = bool(topowheat.TCPM.ENABLED)
        self.bazr_aux_enabled = bool(topowheat.BAZR.AUX_HEADS_ENABLED)
        self.trpl_cfg = topowheat.TRPL
        self.tcpm_cfg = topowheat.TCPM
        self.bazr_cfg = topowheat.BAZR

        if self.tcpm_enabled:
            self.prototype_memory = TopologyCorePrototypeMemory(
                num_domains=self.tcpm_cfg.NUM_DOMAINS,
                num_classes=self.sem_seg_head.num_classes,
                feature_dim=self.cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
                labeled_momentum=self.tcpm_cfg.LABELED_MOMENTUM,
                pseudo_momentum=self.tcpm_cfg.PSEUDO_MOMENTUM,
                temperature=self.tcpm_cfg.TEMPERATURE,
                max_samples_per_class=self.tcpm_cfg.MAX_SAMPLES_PER_CLASS,
                max_query_pixels_per_class=(
                    self.tcpm_cfg.MAX_QUERY_PIXELS_PER_CLASS
                ),
                min_core_pixels=self.tcpm_cfg.MIN_CORE_PIXELS,
                query_mode=self.tcpm_cfg.QUERY_MODE,
                hard_query_fraction=self.tcpm_cfg.HARD_QUERY_FRACTION,
                alignment_tolerance=self.tcpm_cfg.ALIGNMENT_TOLERANCE,
                hard_negative_margin=self.tcpm_cfg.HARD_NEGATIVE_MARGIN,
                stem_class=self.stem_class,
                leaf_class=self.leaf_class,
            )

        if self.bazr_aux_enabled:
            # Keep the global RNG stream identical to the pure TCPM ablation.
            with torch.random.fork_rng(devices=[]):
                auxiliary_seed = max(int(self.cfg.SEED), 0) + 1907
                torch.manual_seed(auxiliary_seed)
                num_classes = self.sem_seg_head.num_classes
                self.bazr_high_head = nn.Conv2d(
                    self.cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
                    num_classes,
                    kernel_size=1,
                )
                self.bazr_low_head = nn.Conv2d(
                    self.cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
                    num_classes,
                    kernel_size=1,
                )
                self.bazr_gate = BAZRFusionGate(
                    num_classes,
                    hidden_dim=self.bazr_cfg.GATE_HIDDEN_DIM,
                )

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        matcher_ssl = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        ssl_criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher_ssl,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "cfg": cfg,
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "ssl_criterion": ssl_criterion,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def init_ema_weights(self, ckpt_dir, strict=True):
        if ckpt_dir != "":
            ckpt = torch.load(ckpt_dir, map_location="cpu")['model']
            self.load_state_dict(ckpt, strict=False)
            
            del ckpt
          
    def init_ema(self, cfg):
        # Initialize EMA model.
        self.ema_backbone = build_backbone(cfg).to(self.device) #deepcopy(self.backbone)
        self.ema_sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape()).to(self.device)
        self.ema_sem_seg_head.load_state_dict(self.sem_seg_head.state_dict())
        self.ema_backbone.load_state_dict(self.backbone.state_dict())

        for param in self.ema_backbone.parameters():
            param.detach_()
        self.ema_backbone.requires_grad_(False)
        self.ema_backbone.eval()

        for param in self.ema_sem_seg_head.parameters():
            param.detach_()
        self.ema_sem_seg_head.requires_grad_(False)
        self.ema_sem_seg_head.eval()

        if cfg.SSL.TEACHER_CKPT != "":
            self.init_ema_weights(cfg.SSL.TEACHER_CKPT)

    def update_ema_module(self, module, ema_module, ema_decay):
        """ unused: replaced by update_ema_module() in train_loop.py """

        # Update parameters.
        module_params = OrderedDict(module.named_parameters())
        ema_module_params = OrderedDict(ema_module.named_parameters())

        assert module_params.keys() == ema_module_params.keys()

        for name, param in module_params.items():
            ema_module_params[name].sub_((1. - ema_decay) * (ema_module_params[name] - param))

        # Update buffers.
        module_buffers = OrderedDict(module.named_buffers())
        ema_module_buffers = OrderedDict(ema_module.named_buffers())

        assert module_buffers.keys() == ema_module_buffers.keys()

        for name, buffer in module_buffers.items():
            if buffer.dtype == torch.float32:
                ema_module_buffers[name].sub_((1. - ema_decay) * (ema_module_buffers[name] - buffer))
            else:
                print(buffer.dtype)
                ema_module_buffers[name] = buffer.clone()

    def update_ema_step(self, ema_decay):
        """ unused """

        assert self.training, "EMA should only be updated during training!"

        self.update_ema_module(self.backbone, self.ema_backbone, ema_decay=ema_decay)
        self.update_ema_module(self.sem_seg_head, self.ema_sem_seg_head, ema_decay=ema_decay)


    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        """ unused """
        
        student_model_dict = self.backbone.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.ema_backbone.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))
        self.ema_backbone.load_state_dict(new_teacher_dict)

        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.sem_seg_head.state_dict().items()
            }
        else:
            student_model_dict = self.sem_seg_head.state_dict()

        student_model_dict = self.sem_seg_head.state_dict()
        new_teacher_dict = OrderedDict()
        for key, value in self.ema_sem_seg_head.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.ema_sem_seg_head.load_state_dict(new_teacher_dict)

    def _run_head_on_images(self, image_tensors, return_features=False):
        normalized = [
            (image.to(self.device) - self.pixel_mean) / self.pixel_std
            for image in image_tensors
        ]
        images = ImageList.from_tensors(normalized, self.size_divisibility)
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(
            features,
            return_features=return_features,
        )
        return images, outputs

    def _semantic_probabilities(self, outputs, size):
        return query_semantic_probabilities(
            outputs["pred_logits"],
            outputs["pred_masks"],
            size=size,
        )

    @torch.no_grad()
    def generate_ssl_targets(self, batched_inputs):
        """Generate legacy or TRPL targets without changing the old forward API."""
        image_tensors = [sample["image"].to(self.device) for sample in batched_inputs]
        base_height = max(image.shape[-2] for image in image_tensors)
        base_width = max(image.shape[-1] for image in image_tensors)
        target_size = (base_height, base_width)

        scales = [float(scale) for scale in self.trpl_cfg.VIEW_SCALES]
        if 1.0 not in scales:
            scales.insert(0, 1.0)
        view_probabilities = []
        primary_outputs = None
        for scale in scales:
            if scale == 1.0:
                scaled_images = image_tensors
            else:
                scaled_images = [
                    F.interpolate(
                        image.unsqueeze(0).float(),
                        scale_factor=scale,
                        mode="bilinear",
                        align_corners=False,
                        recompute_scale_factor=False,
                    ).squeeze(0)
                    for image in image_tensors
                ]
            _, outputs = self._run_head_on_images(scaled_images)
            if scale == 1.0:
                primary_outputs = outputs
            view_probabilities.append(
                self._semantic_probabilities(outputs, target_size)
            )

        legacy_targets = self.prepare_ssl_outputs(primary_outputs)
        if not self.trpl_enabled:
            return {"pseudo_label": legacy_targets}

        trpl_targets = build_trpl_targets(
            view_probabilities,
            class_thresholds=self.trpl_cfg.CLASS_THRESHOLDS,
            uncertainty_temperature=self.trpl_cfg.UNCERTAINTY_TEMPERATURE,
            uncertainty_weight=self.trpl_cfg.UNCERTAINTY_WEIGHT,
            max_uncertainty=self.trpl_cfg.MAX_UNCERTAINTY,
            stem_class=self.stem_class,
            skeleton_threshold=self.trpl_cfg.SKELETON_THRESHOLD,
            persistence=self.trpl_cfg.PERSISTENCE,
            support_persistence=self.trpl_cfg.SUPPORT_PERSISTENCE,
            skeleton_iterations=self.trpl_cfg.SKELETON_ITERATIONS,
            boundary_radius=self.trpl_cfg.BOUNDARY_RADIUS,
            boundary_min_stem_probability=(
                self.trpl_cfg.BOUNDARY_MIN_STEM_PROBABILITY
            ),
            skeleton_match_radius=self.trpl_cfg.SKELETON_MATCH_RADIUS,
            stem_support_radius=self.trpl_cfg.STEM_SUPPORT_RADIUS,
            stem_support_min_probability=(
                self.trpl_cfg.STEM_SUPPORT_MIN_PROBABILITY
            ),
            core_strategy=self.tcpm_cfg.CORE_STRATEGY,
            core_erode_iterations=self.trpl_cfg.CORE_ERODE_ITERATIONS,
            core_stem_radius=self.trpl_cfg.CORE_STEM_RADIUS,
        )
        return {
            "pseudo_label": legacy_targets,
            "trpl": trpl_targets,
        }

    def _padded_semantic_targets(self, batched_inputs, size):
        labels = torch.full(
            (len(batched_inputs), size[0], size[1]),
            255,
            device=self.device,
            dtype=torch.long,
        )
        for index, sample in enumerate(batched_inputs):
            semantic = sample["sem_seg"].to(self.device)
            height = min(semantic.shape[-2], size[0])
            width = min(semantic.shape[-1], size[1])
            labels[index, :height, :width] = semantic[:height, :width]
        return labels

    def _domain_ids(self, batched_inputs):
        num_domains = int(self.tcpm_cfg.NUM_DOMAINS)
        domain_ids = []
        for sample in batched_inputs:
            if "domain_id" not in sample:
                raise ValueError("TCPM requires domain_id in every dataset record")
            domain_id = int(sample["domain_id"])
            if not 0 <= domain_id < num_domains:
                raise ValueError(
                    "domain_id {} is outside configured range [0, {})".format(
                        domain_id,
                        num_domains,
                    )
                )
            domain_ids.append(domain_id)
        return domain_ids

    def _tcpm_start_iter(self):
        configured = int(self.tcpm_cfg.START_ITER)
        return self.burn_in if configured < 0 else configured

    def _trpl_loss_scale(self):
        start_iter = int(self.trpl_cfg.START_ITER)
        ramp_iters = int(self.trpl_cfg.RAMP_ITERS)
        if self.iter < start_iter:
            return 0.0
        if ramp_iters <= 0:
            return 1.0
        return min(
            max((self.iter - start_iter) / float(ramp_iters), 0.0),
            1.0,
        )

    def _tcpm_state(self):
        return tcpm_curriculum_state(
            current_iter=self.iter,
            start_iter=self._tcpm_start_iter(),
            memory_warmup_iters=self.tcpm_cfg.MEMORY_WARMUP_ITERS,
            loss_ramp_iters=self.tcpm_cfg.LOSS_RAMP_ITERS,
            pseudo_update_start=self.tcpm_cfg.PSEUDO_UPDATE_START,
            pseudo_ramp_iters=self.tcpm_cfg.PSEUDO_RAMP_ITERS,
            pseudo_blend_max=self.tcpm_cfg.PSEUDO_BLEND_MAX,
        )

    def _ensure_tcpm_started(self):
        if not bool(self.prototype_memory.memory_started.item()):
            self.prototype_memory.reset_memory()

    def _prototype_losses(
        self,
        outputs,
        labels,
        weights,
        confidence,
        core_mask,
        query_mask,
        batched_inputs,
        source,
    ):
        state = self._tcpm_state()
        if not state["active"]:
            return {}
        self._ensure_tcpm_started()
        if source == "pseudo":
            raw_core_mask = core_mask
            raw_query_mask = query_mask
            core_mask = raw_core_mask & weights.ge(
                float(self.tcpm_cfg.PSEUDO_MIN_WEIGHT)
            )
            if confidence is not None:
                core_mask &= confidence.ge(
                    float(self.tcpm_cfg.PSEUDO_MIN_CONFIDENCE)
                )
            query_mask = raw_query_mask & weights.ge(
                float(self.tcpm_cfg.QUERY_MIN_WEIGHT)
            )
            update_memory = state["pseudo_update"]
        else:
            raw_core_mask = core_mask
            raw_query_mask = query_mask
            update_memory = True
        losses = self.prototype_memory(
            outputs["mask_features"],
            labels,
            weights,
            core_mask,
            self._domain_ids(batched_inputs),
            query_mask=query_mask,
            source=source,
            update_memory=update_memory,
            pseudo_blend=state["pseudo_blend"],
            leave_one_domain_out=bool(
                self.tcpm_cfg.LEAVE_ONE_DOMAIN_OUT
            ),
        )
        try:
            storage = get_event_storage()
            storage.put_scalar("tcpm/loss_scale", state["loss_scale"])
            storage.put_scalar("tcpm/pseudo_blend", state["pseudo_blend"])
            storage.put_scalar(
                "tcpm/{}_core_pixel_acceptance".format(source),
                float(core_mask.sum().item())
                / max(float(raw_core_mask.sum().item()), 1.0),
            )
            storage.put_scalar(
                "tcpm/{}_query_pixel_acceptance".format(source),
                float(query_mask.sum().item())
                / max(float(raw_query_mask.sum().item()), 1.0),
            )
            drift = (
                self.prototype_memory.last_pseudo_drift
                if source == "pseudo"
                else self.prototype_memory.last_labeled_drift
            )
            storage.put_scalar(
                "tcpm/{}_drift".format(source),
                float(drift.item()),
            )
            for name in (
                "contrastive",
                "domain_compact",
                "hard_negative",
                "query_distance",
                "anchor_coverage",
                "hard_negative_rate",
                "query_records",
            ):
                storage.put_scalar(
                    "tcpm/{}_raw_{}".format(source, name),
                    float(losses[name].detach().item()),
                )
            if source == "pseudo":
                initialized = self.prototype_memory.pseudo_initialized
                candidates = (
                    self.prototype_memory.last_pseudo_candidate_counts
                )
                cumulative = self.prototype_memory.pseudo_update_counts
            else:
                initialized = self.prototype_memory.labeled_initialized
                candidates = (
                    self.prototype_memory.last_labeled_candidate_counts
                )
                cumulative = self.prototype_memory.labeled_update_counts
            storage.put_scalar(
                "tcpm/{}_memory_candidates".format(source),
                float(candidates.sum().item()),
            )
            storage.put_scalar(
                "tcpm/{}_memory_initialized".format(source),
                float(initialized.sum().item()),
            )
            for class_id in range(self.sem_seg_head.num_classes):
                prefix = "tcpm/{}_class_{}".format(source, class_id)
                storage.put_scalar(
                    prefix + "_initialized_domains",
                    float(initialized[:, class_id].sum().item()),
                )
                storage.put_scalar(
                    prefix + "_candidates",
                    float(candidates[:, class_id].sum().item()),
                )
                storage.put_scalar(
                    prefix + "_cumulative_updates",
                    float(cumulative[:, class_id].sum().item()),
                )
        except AssertionError:
            # Direct model calls used by diagnostics may not own EventStorage.
            pass
        loss_scale = state["loss_scale"]
        return {
            "loss_tcpm_contrastive": (
                losses["contrastive"]
                * float(self.tcpm_cfg.CONTRASTIVE_WEIGHT)
                * loss_scale
            ),
            "loss_tcpm_domain": (
                losses["domain_compact"]
                * float(self.tcpm_cfg.DOMAIN_COMPACT_WEIGHT)
                * loss_scale
            ),
            "loss_tcpm_hard_negative": (
                losses["hard_negative"]
                * float(self.tcpm_cfg.HARD_NEGATIVE_WEIGHT)
                * loss_scale
            ),
        }

    def _bazr_logits(self, outputs, size):
        if not self.bazr_aux_enabled:
            return None
        high_features = outputs["mask_features"]
        low_features = outputs["multi_scale_features"][0]
        if self.training:
            high_features = high_features.detach()
            low_features = low_features.detach()
        high = self.bazr_high_head(high_features)
        low = self.bazr_low_head(low_features)
        high = F.interpolate(
            high,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        low = F.interpolate(
            low,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        return high, low

    def _bazr_fused_probabilities(self, auxiliary):
        high, low = auxiliary
        high_probabilities = F.softmax(high.float(), dim=1)
        low_probabilities = F.softmax(low.float(), dim=1)
        gate = self.bazr_gate(
            low_probabilities,
            high_probabilities,
        )
        return (
            gate * high_probabilities
            + (1.0 - gate) * low_probabilities
        )

    def _supervised_topowheat_losses(
        self,
        outputs,
        batched_inputs,
        target_size,
    ):
        losses = {}
        labels = self._padded_semantic_targets(batched_inputs, target_size)
        valid = labels.ne(255)
        safe_labels = labels.masked_fill(~valid, 0)
        probabilities = self._semantic_probabilities(outputs, target_size)

        if self.trpl_enabled:
            region_nll = masked_nll_loss(
                probabilities,
                safe_labels,
                valid.float(),
                valid,
                class_balanced=bool(
                    self.trpl_cfg.CLASS_BALANCED_NLL
                ),
                num_classes=self.sem_seg_head.num_classes,
                min_class_pixels=self.trpl_cfg.MIN_CLASS_PIXELS,
            )
            region_dice = masked_dice_loss(
                probabilities,
                safe_labels,
                valid.float(),
                valid,
                num_classes=self.sem_seg_head.num_classes,
            )
            losses["loss_trpl_region_sup"] = (
                region_nll
                + float(self.trpl_cfg.SUPERVISED_DICE_WEIGHT)
                * region_dice
            ) * float(self.trpl_cfg.SUPERVISED_REGION_WEIGHT)
            stem_probability = probabilities[
                :, self.stem_class : self.stem_class + 1
            ]
            stem_target = safe_labels.eq(self.stem_class).unsqueeze(1)
            losses["loss_trpl_topology_sup"] = (
                soft_cldice_loss(
                    stem_probability,
                    stem_target,
                    valid=valid.unsqueeze(1),
                    iterations=self.trpl_cfg.SKELETON_ITERATIONS,
                    skip_empty_target=True,
                )
                * float(self.trpl_cfg.SUPERVISED_TOPOLOGY_WEIGHT)
            )

        if self.tcpm_enabled and self._tcpm_state()["active"]:
            core_mask = build_core_mask(
                safe_labels,
                valid,
                num_classes=self.sem_seg_head.num_classes,
                stem_class=self.stem_class,
                strategy=self.tcpm_cfg.CORE_STRATEGY,
                erode_iterations=self.trpl_cfg.CORE_ERODE_ITERATIONS,
                stem_radius=self.trpl_cfg.CORE_STEM_RADIUS,
                skeleton_iterations=self.trpl_cfg.SKELETON_ITERATIONS,
            )
            losses.update(
                self._prototype_losses(
                    outputs,
                    safe_labels,
                    valid.float(),
                    None,
                    core_mask,
                    valid,
                    batched_inputs,
                    source="labeled",
                )
            )

        auxiliary = self._bazr_logits(outputs, target_size)
        if auxiliary is not None:
            auxiliary_loss = sum(
                F.cross_entropy(logits, labels, ignore_index=255)
                for logits in auxiliary
            ) / len(auxiliary)
            fused_probabilities = self._bazr_fused_probabilities(auxiliary)
            fusion_loss = masked_nll_loss(
                fused_probabilities,
                safe_labels,
                valid.float(),
                valid,
            )
            losses["loss_bazr_aux_sup"] = (
                (
                    auxiliary_loss
                    + fusion_loss
                    * float(self.bazr_cfg.FUSION_LOSS_WEIGHT)
                )
                * float(self.bazr_cfg.AUX_LOSS_WEIGHT)
            )
        return losses

    def _log_trpl_diagnostics(
        self,
        probabilities,
        labels,
        reliable,
        target,
    ):
        period = max(int(self.ssl_diagnostic_period), 0)
        if not period or self.iter % period != 0:
            return
        try:
            storage = get_event_storage()
        except AssertionError:
            return

        student_labels = probabilities.detach().argmax(dim=1)
        total_pixels = max(float(labels.numel()), 1.0)
        storage.put_scalar("trpl/loss_scale", self._trpl_loss_scale())
        storage.put_scalar(
            "trpl/legacy_query_weight",
            float(self.trpl_cfg.LEGACY_QUERY_LOSS_WEIGHT)
            * self._trpl_loss_scale(),
        )
        storage.put_scalar(
            "trpl/reliable_fraction",
            float(reliable.sum().item()) / total_pixels,
        )
        for class_id in range(self.sem_seg_head.num_classes):
            teacher_class = labels.eq(class_id)
            storage.put_scalar(
                "trpl/teacher_class_{}_fraction".format(class_id),
                float(teacher_class.sum().item()) / total_pixels,
            )
            storage.put_scalar(
                "trpl/reliable_class_{}_fraction".format(class_id),
                float((teacher_class & reliable).sum().item()) / total_pixels,
            )
            storage.put_scalar(
                "trpl/student_class_{}_fraction".format(class_id),
                float(student_labels.eq(class_id).sum().item()) / total_pixels,
            )

        stable_skeleton = target["stable_skeleton"].to(self.device)
        support_skeleton = target["support_skeleton"].to(self.device)
        stem_support = target["stem_support"].to(self.device)
        uncertain_boundary = target["uncertain_boundary"].to(self.device)
        for name, mask in (
            ("stable_skeleton", stable_skeleton),
            ("support_skeleton", support_skeleton),
            ("stem_support", stem_support),
            ("uncertain_boundary", uncertain_boundary),
        ):
            storage.put_scalar(
                "trpl/{}_fraction".format(name),
                float(mask.sum().item()) / total_pixels,
            )

        teacher_stem_pixels = labels.eq(self.stem_class).sum()
        student_stem_pixels = student_labels.eq(self.stem_class).sum()
        storage.put_scalar(
            "trpl/student_teacher_stem_area_ratio",
            float(student_stem_pixels.item())
            / max(float(teacher_stem_pixels.item()), 1.0),
        )
        support_pixels = max(float(stem_support.sum().item()), 1.0)
        student_stem = probabilities.detach()[:, self.stem_class]
        storage.put_scalar(
            "trpl/stem_support_floor_coverage",
            float(
                (
                    student_stem.ge(
                        float(
                            self.trpl_cfg.STEM_SUPPORT_PROBABILITY_FLOOR
                        )
                    )
                    & stem_support
                ).sum().item()
            )
            / support_pixels,
        )

    def _semi_supervised_topowheat_losses(
        self,
        outputs,
        batched_inputs,
        target,
    ):
        labels = target["labels"].to(self.device)
        weights = target["weights"].to(self.device)
        confidence = target["confidence"].to(self.device)
        reliable = target["reliable"].to(self.device)
        teacher_probabilities = target["probabilities"].to(self.device)
        stable_skeleton = target["stable_skeleton"].to(self.device)
        stem_support = target["stem_support"].to(self.device)
        uncertain_boundary = target["uncertain_boundary"].to(self.device)
        core_mask = target["core_mask"].to(self.device)
        target_size = labels.shape[-2:]
        probabilities = self._semantic_probabilities(outputs, target_size)
        trpl_scale = self._trpl_loss_scale()
        self._log_trpl_diagnostics(
            probabilities,
            labels,
            reliable,
            target,
        )

        losses = {}
        region_nll = masked_nll_loss(
            probabilities,
            labels,
            weights,
            reliable,
            class_balanced=bool(self.trpl_cfg.CLASS_BALANCED_NLL),
            num_classes=self.sem_seg_head.num_classes,
            min_class_pixels=self.trpl_cfg.MIN_CLASS_PIXELS,
        )
        region_dice = masked_dice_loss(
            probabilities,
            labels,
            weights,
            reliable,
            num_classes=self.sem_seg_head.num_classes,
        )
        losses["loss_trpl_region_ssl"] = (
            region_nll
            + float(self.trpl_cfg.DICE_LOSS_WEIGHT) * region_dice
        ) * float(self.trpl_cfg.REGION_LOSS_WEIGHT) * trpl_scale

        losses["loss_trpl_boundary_ssl"] = (
            masked_soft_cross_entropy(
                probabilities,
                teacher_probabilities,
                weights,
                uncertain_boundary,
            )
            * float(self.trpl_cfg.BOUNDARY_DISTILLATION_WEIGHT)
            * trpl_scale
        )

        student_stem = probabilities[
            :, self.stem_class : self.stem_class + 1
        ]
        teacher_stem = labels.eq(self.stem_class).unsqueeze(1)
        topology_valid = reliable.unsqueeze(1)
        losses["loss_trpl_topology_ssl"] = (
            soft_cldice_loss(
                student_stem,
                teacher_stem,
                valid=topology_valid,
                iterations=self.trpl_cfg.SKELETON_ITERATIONS,
                skip_empty_target=True,
            )
            * float(self.trpl_cfg.TOPOLOGY_LOSS_WEIGHT)
            * trpl_scale
        )
        skeleton_target = stable_skeleton.unsqueeze(1).float()
        losses["loss_trpl_skeleton_ssl"] = (
            centerline_recall_loss(
                student_stem,
                skeleton_target,
                tolerance_radius=(
                    self.trpl_cfg.CENTERLINE_TOLERANCE_RADIUS
                ),
            )
            * float(self.trpl_cfg.SKELETON_LOSS_WEIGHT)
            * trpl_scale
        )
        teacher_stem_probability = teacher_probabilities[
            :, self.stem_class : self.stem_class + 1
        ]
        support_minimum = float(
            self.trpl_cfg.STEM_SUPPORT_MIN_PROBABILITY
        )
        support_strength = (
            (teacher_stem_probability - support_minimum)
            / max(1.0 - support_minimum, 1e-6)
        ).clamp(0.0, 1.0)
        support_weights = 0.25 + 0.75 * support_strength
        losses["loss_trpl_stem_support_ssl"] = (
            positive_probability_floor_loss(
                student_stem,
                stem_support,
                probability_floor=(
                    self.trpl_cfg.STEM_SUPPORT_PROBABILITY_FLOOR
                ),
                weights=support_weights,
            )
            * float(self.trpl_cfg.STEM_SUPPORT_LOSS_WEIGHT)
            * trpl_scale
        )

        if self.tcpm_enabled and self._tcpm_state()["active"]:
            prototype_losses = self._prototype_losses(
                outputs,
                labels,
                weights,
                confidence,
                core_mask,
                reliable,
                batched_inputs,
                source="pseudo",
            )
            losses.update(
                {
                    key + "_ssl": value
                    for key, value in prototype_losses.items()
                }
            )

        auxiliary = self._bazr_logits(outputs, target_size)
        if auxiliary is not None:
            auxiliary_region = sum(
                masked_nll_loss(
                    F.softmax(logits, dim=1),
                    labels,
                    weights,
                    reliable,
                )
                for logits in auxiliary
            ) / len(auxiliary)
            fused_probabilities = self._bazr_fused_probabilities(auxiliary)
            fusion_loss = masked_nll_loss(
                fused_probabilities,
                labels,
                weights,
                reliable,
            )
            losses["loss_bazr_aux_ssl"] = (
                (
                    auxiliary_region
                    + fusion_loss
                    * float(self.bazr_cfg.FUSION_LOSS_WEIGHT)
                )
                * float(self.bazr_cfg.AUX_LOSS_WEIGHT)
                * trpl_scale
            )
        return losses

    def prepare_ssl_outputs(self, targets, thresh_class = .7, thresh_mask=.95, mask_size = 5):
        new_outputs = []
        with torch.no_grad():
            bs = int(targets['pred_logits'].shape[0])

            if self.thresh_controller is not None:
                self.thresh_controller.thresh_update(targets['pred_logits'].detach())
                self.thresh_class = self.thresh_controller.get_thresh_global()

            for b in range(bs):
                mask_cls = targets['pred_logits'][b]
                mask_pred = targets['pred_masks'][b]

                objects = mask_cls.argmax(dim=1) != mask_cls.shape[1] - 1
                mask_cls = mask_cls[objects]
                mask_pred = mask_pred[objects]

                high_conf = F.softmax(mask_cls, dim=1).max(dim=1).values > self.thresh_class
                mask_cls = mask_cls[high_conf]
                mask_pred = mask_pred[high_conf]
                
                # except stem class
                not_empty = torch.sigmoid(mask_pred).sum(dim=(1,2)) > mask_size
                not_empty[mask_cls.argmax(dim=1)==2] = True    # except stem class
                tar_cls = mask_cls[not_empty].argmax(dim=1)
                tar_mask = torch.sigmoid(mask_pred[not_empty]) > .5

                new_outputs.append({'labels': tar_cls.clone(), 'masks': tar_mask.clone()})

        return new_outputs
    
    def save_images(self, iter, preds, mu, std, grid_size=(2, 2), real=False):
        import os
        from PIL import Image
        import numpy as np
        mu_ = torch.tensor(mu).view(1,3,1,1).cpu()
        std_ = torch.tensor(std).view(1,3,1,1).cpu()

        preds = preds.detach().cpu()*std_ + mu_
        img = np.rint(preds.numpy()).clip(0, 255).astype(np.uint8)
        img = img[:grid_size[0]*grid_size[1]]
        gw, gh = grid_size
        _N, C, H, W = img.shape
        img = img.reshape([gh, gw, C, H, W])
        img = img.transpose(0, 3, 1, 4, 2)
        img = img.reshape([gh * H, gw * W, C])
        
        root=os.path.join(self.cfg.OUTPUT_DIR, "snapshots")
        os.makedirs(root, exist_ok=True)
        
        if real:
            fname = os.path.join(root, f"reals_{iter:07d}.png")
        else:
            fname = os.path.join(root, f"synthesis_{iter:07d}.png")

        assert C in [1, 3]
        if C == 1:
            Image.fromarray(img[:, :, 0], 'L').save(fname)
        if C == 3:
            Image.fromarray(img, 'RGB').save(fname)  

    def forward(self, batched_inputs, branch='supervised', return_preds=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        do_ssl = self.do_ssl and self.iter % self.ssl_freq == 0 and self.training
        assert branch in ['supervised', 'semi-supervised']
        
        losses_all = {}

        if branch == 'supervised': #not do_ssl or self.iter < self.burn_in:
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)
            need_topowheat_features = (
                (
                    self.tcpm_enabled
                    and self._tcpm_state()["active"]
                )
                or self.bazr_aux_enabled
            )
            outputs = self.sem_seg_head(
                features,
                return_features=need_topowheat_features,
            )

            if return_preds:
                return outputs

            if self.training:
                # mask classification target
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, images)
                else:
                    targets = None

                # bipartite matching-based loss
                losses = self.criterion(outputs, targets)

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
                losses_all.update(losses)
                if (
                    self.trpl_enabled
                    or self.tcpm_enabled
                    or self.bazr_aux_enabled
                ):
                    losses_all.update(
                        self._supervised_topowheat_losses(
                            outputs,
                            batched_inputs,
                            images.tensor.shape[-2:],
                        )
                    )
                if not (
                    self.do_ssl
                    and (self.trpl_enabled or self.tcpm_enabled)
                ):
                    self.iter += 1
                return losses_all
            else:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                bazr_aux_results = self._bazr_logits(
                    outputs,
                    images.tensor.shape[-2:],
                )
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

                del outputs

                processed_results = []
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    if self.sem_seg_postprocess_before_inference:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                        mask_cls_result = mask_cls_result.to(mask_pred_result)

                    # semantic segmentation inference
                    if self.semantic_on:
                        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                        if not self.sem_seg_postprocess_before_inference:
                            r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                        processed_results[-1]["sem_seg"] = r

                    if bazr_aux_results is not None:
                        high_aux, low_aux = bazr_aux_results
                        processed_results[-1]["_bazr_aux_high"] = (
                            retry_if_cuda_oom(sem_seg_postprocess)(
                                high_aux[len(processed_results) - 1],
                                image_size,
                                height,
                                width,
                            )
                        )
                        processed_results[-1]["_bazr_aux_low"] = (
                            retry_if_cuda_oom(sem_seg_postprocess)(
                                low_aux[len(processed_results) - 1],
                                image_size,
                                height,
                                width,
                            )
                        )

                    # panoptic segmentation inference
                    if self.panoptic_on:
                        panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["panoptic_seg"] = panoptic_r
                    
                    # instance segmentation inference
                    if self.instance_on:
                        instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                        processed_results[-1]["instances"] = instance_r

                return processed_results
            
        elif branch == 'semi-supervised':  # and self.iter >= self.burn_in:
            images_unl = [x["image_aug"].to(self.device) for x in batched_inputs['data']]
            images_unl = [(x - self.pixel_mean) / self.pixel_std for x in images_unl]
            images_unl = ImageList.from_tensors(images_unl, self.size_divisibility).tensor

            # Student predictions.
            # First perturbation stream.
            features = self.backbone(images_unl)
            need_topowheat_features = (
                (
                    self.tcpm_enabled
                    and self._tcpm_state()["active"]
                )
                or self.bazr_aux_enabled
            )
            outputs = self.sem_seg_head(
                features,
                return_features=need_topowheat_features,
            )

            if self.trpl_enabled and "trpl" in batched_inputs:
                losses_all.update(
                    self._semi_supervised_topowheat_losses(
                        outputs,
                        batched_inputs["data"],
                        batched_inputs["trpl"],
                    )
                )
                legacy_weight = float(
                    self.trpl_cfg.LEGACY_QUERY_LOSS_WEIGHT
                ) * self._trpl_loss_scale()
                if legacy_weight > 0:
                    legacy_losses = self.ssl_criterion(
                        outputs,
                        batched_inputs["pseudo_label"],
                    )
                    for key in list(legacy_losses.keys()):
                        if key in self.ssl_criterion.weight_dict:
                            legacy_losses[key] *= (
                                self.ssl_criterion.weight_dict[key]
                                * legacy_weight
                            )
                        else:
                            legacy_losses.pop(key)
                    losses_all.update(
                        {
                            key + "_legacy_ssl": value
                            for key, value in legacy_losses.items()
                        }
                    )
            else:
                losses = self.ssl_criterion(
                    outputs,
                    batched_inputs["pseudo_label"],
                )

                for k in list(losses.keys()):
                    if k in self.ssl_criterion.weight_dict:
                        losses[k] *= self.ssl_criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)

                losses_ssl = {}
                for k, v in losses.items():
                    losses_ssl[k + "_ssl"] = 2.0*v
                losses_ssl['thresh_class_ssl'] = torch.tensor(
                    self.thresh_class,
                    device=self.device,
                )
                losses_all.update(losses_ssl)
            if self.do_ssl and (self.trpl_enabled or self.tcpm_enabled):
                self.iter += 1
        
        return losses_all

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )

        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
       
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
