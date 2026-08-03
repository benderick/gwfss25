# GWFSS experiment protocol

## Execution workflow

This checkout is used for code changes, static checks, and log analysis only;
it is not expected to contain the training runtime or datasets. Training and
evaluation run on a separate machine. Runtime errors and metrics are copied
back here for diagnosis, then code changes are synchronized to that machine.

The fixed Stage I result for current TopoWheat comparisons is validation mIoU
`73.10942` and competition-test mIoU `69.4389`. Checkpoints and hyperparameters
must be selected on validation; the test result is a locked final comparison.

The former TRPL--TCPM--BAZR stack has failed its matched full-validation audits
and is quarantined. Its configs remain only for reproduction and diagnosis; do
not treat the rows or commands below as a recommendation for another training
run. The replacement decision protocol is in `METHOD_REDESIGN.md`.

## 1. What is the baseline?

The paper's baseline is BEiTv2-L + ViT-Adapter + Mask2Former with standard
bilinear FPN upsampling, trained only on the 99 labelled competition images.
It excludes SAPA, guided distillation, and test-time scaling. Its measured
result in the frozen current protocol is the Stage I result recorded above.

The released `stage1_train.sh` was not this pure baseline: SAPA was hard-coded
inside the pixel decoder. The experiment configs now separate the stages:

| Config | Labelled data | Unlabelled data | Upsampling | TTA |
| --- | ---: | ---: | --- | --- |
| `competition_baseline.yaml` | 99 | 0 | bilinear | off |
| `competition_sapa.yaml` | 99 | 0 | SAPA | off |
| `competition_stage2_stem4500.yaml` | 99 | prior stem-aware 4,500 | SAPA | off |
| `competition_stage2_random4500.yaml` | 99 | random balanced 4,500 | SAPA | off |
| `competition_stage2_all.yaml` | 99 | 64,368 | SAPA | off |
| `competition_topowheat_trpl.yaml` | 99 | prior stem-aware 4,500 | bilinear + TRPL | off |
| `competition_topowheat_trpl_tcpm.yaml` | 99 | prior stem-aware 4,500 | bilinear + TRPL + TCPM | off |
| `competition_topowheat_bazr_train.yaml` | 99 | prior stem-aware 4,500 | full training-time method | off |
| `competition_topowheat_bazr.yaml` | 99 | prior stem-aware 4,500 | bilinear + TRPL + TCPM | BAZR |

TTA must remain disabled in training ablations. Evaluate the zoom-in/multi-scale
stage separately so training improvements are not mixed with inference cost.

The upstream repository omits `maskformer2_mmseg.yaml`. The local replacement
copied from `maskformer2_R50_bs16_90k.yaml` is retained only as the common
Mask2Former semantic-segmentation structure. Stage-specific settings are
declared explicitly in
`beit_adapter/maskformer2_beit_adapter_large_bas8_20k.yaml`, while the
Stage-I competition experiment files explicitly use a global batch of 4, 20,000
iterations for Stage 1, validation-set evaluation, TTA off, and SSL off.
This prevents the copied parent's effective defaults (batch 8, 15,000
iterations, training-set evaluation, and TTA on) from silently controlling
Stage 1.

## 2. Which unlabelled set should be used?

The released 4,500-image list is not an official dataset split. It is an output
of the winning method: Stage 1 pseudo-labels all candidate images, then ranks
them by predicted stem-pixel proportion and retains the top 500 from each of
the nine domains. It may be used to reproduce the winner and as a strong prior
method comparison, but it must be identified as `stem-aware 4500` rather than
as a neutral data split.

Use the same 64,368-image candidate pool and the same 4,500-image budget for
selection experiments:

| Selection | Purpose |
| --- | --- |
| domain-balanced random 500/domain | neutral data-budget control |
| prior stem-aware top 500/domain | winning method reproduction |
| proposed selection top 500/domain | test of the new selection contribution |

Keep the Stage 1 teacher, Stage 2 schedule, augmentations, and random seeds
fixed across these rows. If the proposed method does not change sample
selection, it may train on the released stem-aware 4,500, but the paper must
state that it builds on the winner's selection and must not claim that
selection as a new contribution.

Using all 64,368 images is a data-scaling ablation, not the default experiment.
Run it with the same 90,000 optimizer steps as the 4,500-image setting and
report that compute is fixed. This isolates the effect of data coverage. A
fixed-epoch comparison would give the all-data run much more optimization and
would not be a controlled ablation.

## 3. Dataset splits

### Competition protocol (primary)

| Split | Images | Use |
| --- | ---: | --- |
| train | 99 | training |
| validation | 99 | checkpoint and hyperparameter selection |
| test | 110 | one locked final evaluation |

Do not select thresholds, checkpoints, or modules using the test ground truth,
even though it is locally available.

### Full 1,096-image protocol (secondary)

The dataset paper did define two experimental splits even though the released
files are grouped by institution rather than by `train/val/test` directories.

- Random split: 70% train, 10% validation, 20% test. The paper does not publish
  the random seed or the exact file list, so its reported random-split numbers
  cannot be reproduced exactly from the paper alone.
- Region split: train on Arvalis, CIMMYT, ETHZ, INRAE, NJAU, RRES, and
  ULiege_CRA-W (767 images); validate on UTokyo (109); test on UQ (110).
  USASK's 110 labelled images are not used in this protocol.

The dataset paper obtained its metrics by selecting the best checkpoint on the
corresponding validation split and evaluating it on the test split. Its
SegFormer-B1 results were 73.66 mIoU for the random split and 60.64 mIoU for
the region split.

Use the region split as the secondary result in the improvement paper because
it measures cross-region generalization and is exactly reconstructable. A new
random split can be reported only with a published seed and file manifest; it
must not be described as an exact reproduction of the dataset paper.

## 4. Data preparation

The full-dataset masks are RGB colour masks, while Detectron2 expects one
integer class ID per pixel. Convert them once and validate all local splits:

```bash
python tools/prepare_gwfss_full.py
```

This creates:

```text
GWFSS/GWFSS_v1.0_labelled/class_id/<institution>/<image>.png
```

The class mapping is background=0, head=1, stem=2, leaf=3.

## 5. Commands

Pure supervised baseline:

```bash
CONFIG_FILE=configs/gwfss/experiments/competition_baseline.yaml \
OUTPUT_DIR=outputs/competition_baseline_seed2025 \
bash stage1_train.sh
```

SAPA-enhanced Stage 1:

```bash
bash stage1_train.sh
```

Original Stage 2 with the winner's stem-aware 4,500 images:

```bash
bash stage2_train.sh
```

Neutral same-budget Stage 2 control:

```bash
CONFIG_FILE=configs/gwfss/experiments/competition_stage2_random4500.yaml \
OUTPUT_DIR=outputs/competition_stage2_random4500_seed2025 \
bash stage2_train.sh
```

All-unlabelled data ablation:

```bash
CONFIG_FILE=configs/gwfss/experiments/competition_stage2_all.yaml \
OUTPUT_DIR=outputs/competition_stage2_all_seed2025 \
bash stage2_train.sh
```

Full-dataset region baseline:

```bash
CONFIG_FILE=configs/gwfss/experiments/full_region_baseline.yaml \
OUTPUT_DIR=outputs/full_region_baseline_seed2025 \
bash stage1_train.sh
```

For paper tables, run the final controlled configurations with at least three
fixed seeds and report mean and standard deviation. Keep the validation and
test assignments unchanged across the baseline, original method, and all new
modules.

TopoWheat training and paired global/BAZR evaluation commands are documented
in `TOPOWHEAT.md`.
