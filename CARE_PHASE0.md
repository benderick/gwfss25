# CARE-Wheat Phase-0 Audit

This audit is a frozen-model falsification test. It does not train or modify the
Stage-I checkpoint.

## Question

CARE-Wheat requires an unlabeled image to contribute acquisition variation
without changing the labeled anchor's biological configuration. The phase-0
audit asks whether a frozen Stage-I teacher provides enough observable organ
configuration information to make that matching defensible.

The descriptor contains six dimensionless soft spatial moments for each organ
class: mass, horizontal and vertical first moments, and three second moments.
Coordinates are normalized by the current feature-map height and width. The
512-pixel input is therefore the operating condition of the frozen checkpoint,
not a resolution assumption in the method.

## Data separation

- `gwfss_sem_seg_train`: the 99 labeled anchors that would be used by CARE.
- `gwfss_sem_seg_val`: an out-of-sample descriptor validation set. Its masks are
  consulted only after retrieval.
- `gwfss_unlabel_random4500_seed2025`: a deterministic, domain-balanced random
  sample of 500 unlabeled images per source institution.

The old `unlabeled_4500.txt` is deliberately not the default because it was
selected by predicted stem proportion and would confound the new mechanism with
an inherited task-aware sampling rule.

## Pre-registered gates

1. **Inference parity.** Ordinary validation inference must reproduce the locked
   Stage-I mIoU of `73.10942` within `0.10` percentage point.
2. **Out-of-sample signature validity.** On validation images, a cross-domain
   neighbor selected only by teacher signatures must reduce true-mask
   configuration distance by at least `30%` relative to uniform random
   cross-domain retrieval. The paired bootstrap lower bound must be positive.
3. **Cross-domain donor support.** Matching must reduce teacher-signature
   distance by at least `30%`, and at least `80%` of anchors must have compatible
   donors in at least half of the eligible source domains. Compatibility is
   defined from the validation set, not hand-tuned: the donor distance must be
   no larger than the 75th percentile of validation cross-domain nearest-neighbor
   distances.

Only a full run that passes all three gates returns
`care_phase0_supported`. A limited run always returns `smoke_only`.

## Run

Activate the existing environment, then run:

```bash
conda activate gwfss
bash care_phase0.sh
```

The default run processes 99 anchors, 99 validation images, and 4,500 donors.
Descriptor rows are appended to `outputs/care_phase0/cache`; rerunning the same
command resumes an interrupted extraction.

For a quick pipeline smoke test in a fresh output directory:

```bash
MAX_ANCHORS=10 \
MAX_VALIDATION=10 \
MAX_DONORS=90 \
VISUALIZE=2 \
OUTPUT_DIR=outputs/care_phase0_smoke \
bash care_phase0.sh
```

If cache-producing settings change, use a new output directory. To explicitly
discard only the audit cache in the selected output directory, set
`RECOMPUTE_CACHE=1`.

## Artifacts

- `summary.json`: checkpoint identity, all measurements, gates, and verdict.
- `labeled_retrieval.csv`: teacher-selected, random-expected, and GT-oracle
  configuration distances on validation images.
- `donor_matches.csv`: one nearest and one deterministic random donor for every
  eligible anchor-to-domain pair.
- `anchor_support.csv` and `domain_support.csv`: support breadth and domain-level
  failure modes.
- `anchors.csv`, `validation.csv`, and `donors.csv`: reusable descriptor tables.
- `retrievals/*.jpg`: worst-support anchors with matched and same-domain random
  donors for visual inspection.

Return the terminal summary and `summary.json`. If the verdict is negative, do
not start Stage I of CARE training.
