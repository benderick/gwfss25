# TRPL pseudo-label audit

This audit treats the validation images as unlabeled while generating teacher
targets. Ground truth is read only after target generation and is used only for
metrics.

It compares four selectors:

- all teacher pixels;
- confidence at the configured numeric threshold;
- current TRPL reliability at the configured threshold;
- confidence with coverage matched independently inside each predicted class.

The class-matched comparison is the primary control. For example, if TRPL
accepts 2,000 predicted stem pixels, the control accepts the 2,000 most
confident predicted stem pixels. A change in class mixture therefore cannot
create a false stem-precision gain.

## Run order

First run a short smoke audit after syncing the code:

```bash
MAX_IMAGES=10 VISUALIZE_WORST=3 \
CHECKPOINT=outputs/competition_baseline_seed2025/model_best.pth \
OUTPUT_DIR=outputs/trpl_audit_stage1_smoke \
bash trpl_audit.sh
```

Then audit all validation images with the exact Stage I bilinear checkpoint
used to initialize TRPL:

```bash
CHECKPOINT=/path/to/stage1_bilinear/model_best.pth \
OUTPUT_DIR=outputs/trpl_audit_stage1 \
bash trpl_audit.sh
```

Stage II ensemble checkpoints are detected automatically and the teacher branch
is selected. Run the same audit on the checkpoint around 5k and the final 30k
checkpoint:

```bash
CHECKPOINT=/path/to/trpl/model_0004999.pth \
OUTPUT_DIR=outputs/trpl_audit_5k \
bash trpl_audit.sh

CHECKPOINT=/path/to/trpl/model_final.pth \
OUTPUT_DIR=outputs/trpl_audit_30k \
bash trpl_audit.sh
```

Use `CHECKPOINT_BRANCH=student` only for a separate student diagnosis. The
teacher branch is the relevant branch for auditing pseudo-label generation.

## Artifacts

Each output directory contains:

- `summary.json`: aggregate selectors, calibration, topology and decision gates;
- `threshold_sweep.csv`: confidence and reliability quality across thresholds;
- `per_class.csv`: primary selector comparison by class;
- `per_image.csv`: image-level failure statistics;
- `worst_stem/`: the images with the most reliable-stem false positives.

The audit uses deterministic full-image validation preprocessing at test
resolution, followed by the exact `1.0` and configured scaled teacher views
used by `build_trpl_targets`. It deliberately excludes random training crops
so checkpoints can be compared on identical pixels. Images are mean-padded to
a size compatible with both view scales and the backbone stride; padded pixels
are cropped before scoring.

The two predeclared mechanism gates in `summary.json` are:

- reliability must improve stem precision by at least 0.03 over the
  class-matched confidence selector;
- stable multi-view skeleton precision must improve by at least 0.03 over the
  same-size subset of the teacher skeleton selected by confidence, must not
  reduce clDice, and must retain at least 0.10 skeleton sensitivity.

These gates diagnose whether the current mechanism adds information beyond
confidence. They are not a replacement for a controlled training ablation.

The audit also reports `semantic_loss_class_balance_pixel_multipliers`. This is
the per-pixel coefficient induced by the current equal-per-pseudo-class KL,
relative to an ordinary accepted-pixel average. A large stem value means that
each accepted stem error receives much more gradient than a typical accepted
pixel. The current centreline loss is positive-only: it raises stem probability
near a stable skeleton but does not penalize false stem regions elsewhere.

## Full-validation decision

The mechanism audit was completed on all 99 validation images using the exact
Stage I warm start, the selected 5k TRPL checkpoint, and the final 30k teacher.
All checkpoints loaded 100 percent of the model state.

| Teacher checkpoint | Mean-view mIoU | Stem prediction pixels | Stem precision | Stem recall | Stem IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage I, iteration 21999 | 0.728239 | 713,994 | 0.651524 | 0.551178 | 0.425684 |
| TRPL, iteration 4999 | 0.729473 | 697,080 | 0.661248 | 0.546153 | 0.426752 |
| TRPL, iteration 29999 | 0.706477 | 1,164,445 | 0.451876 | 0.623456 | 0.354993 |

Both proposed TRPL signals failed their matched controls before model quality
degraded:

| Mechanism result | Stage I | TRPL 5k | TRPL 30k |
| --- | ---: | ---: | ---: |
| Reliable-stem precision gain over class-matched confidence | -0.001967 | -0.001980 | -0.004418 |
| Stable-skeleton precision gain over matched confidence | -0.011399 | -0.011222 | -0.031381 |
| Stable-skeleton clDice gain over matched confidence | 0.001152 | 0.000051 | -0.013345 |
| Stem per-pixel class-balance multiplier | 16.222606 | 16.959244 | 10.781588 |

From 5k to 30k, predicted stem area grew by 67.0 percent. True-positive stem
pixels grew by only 14.2 percent, while false-positive stem pixels grew from
236,137 to 638,260. False stem components grew from 354 to 1,278. The apparent
recall gain is therefore caused by severe stem expansion, not improved thin
structure recovery.

The current TRPL mechanism is rejected, not merely in need of another threshold
search. In particular:

- `confidence * (1 - disagreement)` does not rank pseudo labels better than
  confidence alone;
- strict multi-view skeleton consensus does not add useful evidence beyond a
  same-size confidence-selected skeleton;
- equal averaging over pseudo classes strongly amplifies erroneous stem pixels;
- the positive-only centreline objective reinforces false stable components and
  has no term that suppresses stem predictions elsewhere.

Do not use the current TRPL targets as the foundation for TCPM, and do not run
additional long TRPL hyperparameter sweeps. This result rejects this particular
pseudo-topology construction; it does not reject topology supervision from
ground-truth labels or a future signed, controlled topology objective.
