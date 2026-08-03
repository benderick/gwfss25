# CARE-Wheat Phase 1

## Why This Experiment Exists

TRPL asked the teacher for dense semantic and topological supervision on
unlabeled images. The audit showed that its reliability and topology selectors
were no better than confidence-matched controls, while class balancing amplified
stem pixels by more than an order of magnitude. CARE tests a narrower use of the
same unlabeled pool: do not trust its pixel labels; use it only to expose a
labeled image to acquisition statistics observed in another domain.

The evidence ladder is deliberately separated:

1. Phase 0 asks whether the frozen model can identify organ configurations
   across domains. It passed, but this does not establish a training benefit.
2. Bank preflight asks whether the implied feature-statistic intervention is
   numerically controlled. It passed, but this still does not establish a
   training benefit.
3. Phase 1 is the first causal test of whether CARE improves validation
   segmentation. C0 and C1 are both necessary to isolate CARE from ordinary
   continued supervised training.

## Frozen Contract

Phase 1 tests one claim: unlabeled wheat images are useful as acquisition
statistics donors when their predicted organ configuration is compatible with
the labeled anchor. The anchor image supplies every spatial activation and the
exact ground-truth mask. The donor supplies only frozen Stage-I `res2` channel
mean and standard deviation.

The implementation deliberately has no pseudo-label loss, topology loss,
class reweighting, teacher update, inference module, stochastic intervention
strength, or layer search.

Phase 0 fixes all selection behavior:

- Only the 84 anchors with compatible donors in at least half of the other
  acquisition domains are eligible.
- Each eligible anchor samples uniformly from its Phase-0-compatible domains.
- The Phase-0 Gaussian compatibility weight controls the interpolation amount.
- The other 15 anchors remain bitwise identical to ordinary supervised input at
  the intervention point.

## Prepare The Bank

Run this once on the local machine with the `gwfss` environment active:

```bash
bash care_prepare_phase1.sh
```

The expected structural counts are 84 supported anchors, 619 compatible pairs,
495 unique donors, and 1024 `res2` channels. The script is resumable and also
reports the implied target-to-anchor standard-deviation ratios before any
training is authorized.

The portable artifacts needed on the training server are:

```text
outputs/care_phase1_bank/manifest.json
outputs/care_phase1_bank/feature_bank.npz
```

Absolute local image paths are not stored in the runtime mapping.

## Short Controlled Runs

Both runs restart from the same selected Stage-I checkpoint with fresh optimizer
state, learning rate `1e-5`, batch size inherited from Stage I, seed 2025, and
2500 iterations. Do not pass `RESUME=1` on the first launch.

```bash
MODE=c0 bash care_phase1_train.sh
MODE=care bash care_phase1_train.sh
```

C0 is mandatory: it measures the benefit or damage caused by another 2500
supervised updates. C1 differs only by CARE. Validation runs at 1000, 2000, and
the final 2500 iterations.

## Decision Rule

The locked Stage-I references are 73.10942 validation mIoU and 69.4389 test
mIoU. The Phase-0 parity run recovered 73.11132 validation mIoU; the 0.00190
point difference is within its declared inference tolerance.

C0 has no performance cutoff of its own. A lower C0 score can be a legitimate
result of continued supervised fitting and does not invalidate the comparison.
Stop before C1 only for a technical failure such as incorrect checkpoint
loading, non-finite loss, or a clear evaluation/code failure.

CARE passes this short screening stage only if all of the following hold:

1. The selected C1 checkpoint exceeds both the selected C0 checkpoint and the
   frozen Stage-I checkpoint by at least 0.30 mIoU points.
2. At least two of the three matched evaluation points have C1 above C0, so the
   result is not a single selected spike.
3. Per-class IoU deltas against both references are reported explicitly; a
   headline mean gain must not conceal a large organ-class regression.
4. Logs show nonzero `care/applied_fraction`, finite
   `care/mean_compatibility_weight`, and runtime feature-statistic diagnostics
   consistent with the bank preflight.

The 0.30-point threshold is a preregistered minimum practical effect for this
low-cost screen, chosen to avoid pursuing another TRPL-scale marginal change.
It is not a confidence interval or a claim of statistical significance. A
paper-level claim would still require the random-donor mechanism control, the
held-out test set, and a repeat run or another independent dataset/split.

Only after that gate passes is a random cross-domain statistic-donor control
justified. A failed gate stops the method without a learning-rate, layer,
threshold, or interpolation-strength sweep.
