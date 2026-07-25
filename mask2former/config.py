# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"
    # The released GWFSS code replaces the final FPN upsampling with SAPA.
    # Keep it configurable so the stock Mask2Former baseline remains runnable.
    cfg.MODEL.SEM_SEG_HEAD.UPSAMPLE_MODE = "sapa"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # TopoWheat extensions. Every module is disabled by default so released
    # configurations and checkpoints retain their original behavior.
    cfg.MODEL.TOPOWHEAT = CN()
    cfg.MODEL.TOPOWHEAT.STEM_CLASS = 2
    cfg.MODEL.TOPOWHEAT.LEAF_CLASS = 3

    cfg.MODEL.TOPOWHEAT.TRPL = CN()
    cfg.MODEL.TOPOWHEAT.TRPL.ENABLED = False
    cfg.MODEL.TOPOWHEAT.TRPL.VIEW_SCALES = [1.0, 0.75]
    cfg.MODEL.TOPOWHEAT.TRPL.CLASS_THRESHOLDS = [0.70, 0.70, 0.55, 0.70]
    cfg.MODEL.TOPOWHEAT.TRPL.UNCERTAINTY_TEMPERATURE = 0.5
    cfg.MODEL.TOPOWHEAT.TRPL.UNCERTAINTY_WEIGHT = 1.0
    cfg.MODEL.TOPOWHEAT.TRPL.MAX_UNCERTAINTY = 0.75
    cfg.MODEL.TOPOWHEAT.TRPL.SKELETON_THRESHOLD = 0.35
    cfg.MODEL.TOPOWHEAT.TRPL.PERSISTENCE = 0.5
    cfg.MODEL.TOPOWHEAT.TRPL.SKELETON_ITERATIONS = 20
    cfg.MODEL.TOPOWHEAT.TRPL.BOUNDARY_RADIUS = 2
    cfg.MODEL.TOPOWHEAT.TRPL.CORE_ERODE_ITERATIONS = 2
    cfg.MODEL.TOPOWHEAT.TRPL.CORE_STEM_RADIUS = 1
    cfg.MODEL.TOPOWHEAT.TRPL.REGION_LOSS_WEIGHT = 2.0
    cfg.MODEL.TOPOWHEAT.TRPL.DICE_LOSS_WEIGHT = 0.5
    cfg.MODEL.TOPOWHEAT.TRPL.TOPOLOGY_LOSS_WEIGHT = 0.5
    cfg.MODEL.TOPOWHEAT.TRPL.SKELETON_LOSS_WEIGHT = 0.25
    cfg.MODEL.TOPOWHEAT.TRPL.SUPERVISED_TOPOLOGY_WEIGHT = 0.2
    cfg.MODEL.TOPOWHEAT.TRPL.LEGACY_QUERY_LOSS_WEIGHT = 0.0

    cfg.MODEL.TOPOWHEAT.TCPM = CN()
    cfg.MODEL.TOPOWHEAT.TCPM.ENABLED = False
    cfg.MODEL.TOPOWHEAT.TCPM.NUM_DOMAINS = 9
    cfg.MODEL.TOPOWHEAT.TCPM.MOMENTUM = 0.99
    cfg.MODEL.TOPOWHEAT.TCPM.TEMPERATURE = 0.1
    cfg.MODEL.TOPOWHEAT.TCPM.MAX_SAMPLES_PER_CLASS = 256
    cfg.MODEL.TOPOWHEAT.TCPM.CORE_STRATEGY = "topology"
    cfg.MODEL.TOPOWHEAT.TCPM.HELDOUT_ENABLED = True
    cfg.MODEL.TOPOWHEAT.TCPM.HOLDOUT_PERIOD = 1000
    cfg.MODEL.TOPOWHEAT.TCPM.CONTRASTIVE_WEIGHT = 0.1
    cfg.MODEL.TOPOWHEAT.TCPM.DOMAIN_COMPACT_WEIGHT = 0.05
    cfg.MODEL.TOPOWHEAT.TCPM.HARD_NEGATIVE_WEIGHT = 0.05
    cfg.MODEL.TOPOWHEAT.TCPM.HARD_NEGATIVE_MARGIN = 0.2

    cfg.MODEL.TOPOWHEAT.BAZR = CN()
    cfg.MODEL.TOPOWHEAT.BAZR.ENABLED = False
    cfg.MODEL.TOPOWHEAT.BAZR.AUX_HEADS_ENABLED = False
    cfg.MODEL.TOPOWHEAT.BAZR.AUX_LOSS_WEIGHT = 0.1
    cfg.MODEL.TOPOWHEAT.BAZR.FUSION_LOSS_WEIGHT = 1.0
    cfg.MODEL.TOPOWHEAT.BAZR.GATE_HIDDEN_DIM = 16
    cfg.MODEL.TOPOWHEAT.BAZR.TOPK = 4
    cfg.MODEL.TOPOWHEAT.BAZR.WINDOW_SIZE = 256
    cfg.MODEL.TOPOWHEAT.BAZR.WINDOW_STRIDE = 128
    cfg.MODEL.TOPOWHEAT.BAZR.ZOOM_SIZE = 512
    cfg.MODEL.TOPOWHEAT.BAZR.NMS_THRESHOLD = 0.3
    cfg.MODEL.TOPOWHEAT.BAZR.ENTROPY_WEIGHT = 1.0
    cfg.MODEL.TOPOWHEAT.BAZR.ENDPOINT_WEIGHT = 2.0
    cfg.MODEL.TOPOWHEAT.BAZR.DISAGREEMENT_WEIGHT = 1.0
    cfg.MODEL.TOPOWHEAT.BAZR.STEM_WEIGHT = 0.5
    cfg.MODEL.TOPOWHEAT.BAZR.GATE_SLOPE = 8.0
    cfg.MODEL.TOPOWHEAT.BAZR.GATE_MARGIN = 0.0
    cfg.MODEL.TOPOWHEAT.BAZR.SKELETON_ITERATIONS = 20
    cfg.MODEL.TOPOWHEAT.BAZR.RETURN_DIAGNOSTICS = False
