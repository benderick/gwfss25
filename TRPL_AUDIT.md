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
  ordinary teacher stem skeleton while retaining at least 0.10 skeleton
  sensitivity.

These gates diagnose whether the current mechanism adds information beyond
confidence. They are not a replacement for a controlled training ablation.

The audit also reports `semantic_loss_class_balance_pixel_multipliers`. This is
the per-pixel coefficient induced by the current equal-per-pseudo-class KL,
relative to an ordinary accepted-pixel average. A large stem value means that
each accepted stem error receives much more gradient than a typical accepted
pixel. The current centreline loss is positive-only: it raises stem probability
near a stable skeleton but does not penalize false stem regions elsewhere.
