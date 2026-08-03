# TopoWheat implementation

> **Status: rejected experimental implementation.** Full-validation audits show
> that TRPL does not outperform confidence-matched selection, its topology signal
> does not outperform a matched confidence skeleton, and the 30k run degrades
> sharply. TCPM and BAZR depend on unsupported mechanisms. Keep this code only to
> reproduce diagnosed experiments; do not present it as the paper's validated
> method or start new long runs from it. See
> [`METHOD_REDESIGN.md`](METHOD_REDESIGN.md) and
> [`TRPL_AUDIT.md`](TRPL_AUDIT.md).

All extensions are registered under `MODEL.TOPOWHEAT` and are disabled by
default. The released supervised and guided-distillation configurations
therefore keep their original model structure and loss path.

## Modules

### TRPL

`MODEL.TOPOWHEAT.TRPL.ENABLED` enables topology-reliable pseudo-label
learning. The EMA teacher predicts one original-scale weak view and one view at
`VIEW_SCALE`. Query predictions are composed into aligned semantic
probabilities and used to calculate:

- one reliability score: mean confidence times one minus view disagreement;
- a reliable semantic region selected by one global threshold;
- a stable stem centreline shared by both views within one-pixel tolerance;
- uncertain stem boundaries excluded from semantic supervision.

The student receives a class-balanced soft KL loss on the reliable region and
a positive centreline likelihood loss on stable topology evidence. TRPL does
not add losses to labelled images and does not mix in the legacy query
pseudo-label objective. Its only active hyperparameters are `VIEW_SCALE`,
`RELIABILITY_THRESHOLD`, `TOPOLOGY_WEIGHT`, and `RAMP_ITERS`; morphology and
spatial tolerance are fixed implementation choices.

### TCPM

`MODEL.TOPOWHEAT.TCPM.ENABLED` adds a topology-core prototype memory. Stem
prototypes use the stable dual-view skeleton and its narrow neighbourhood; other
classes use eroded reliable interiors. Labelled and pseudo-labelled cores are
stored in separate class-by-domain banks. For each sample, the loss reads an
anchor aggregated from every available domain except that sample's own
domain; all eligible domains continue to update on every iteration.

Only topology cores write to memory. Loss queries come from the wider reliable
region: at most 128 spatially spread pixels are considered per class, and the
25% farthest from their correct leave-one-domain-out anchor receive prototype
contrastive, tolerance-truncated alignment, and stem-versus-leaf ranking
losses. This asymmetric path keeps boundary mixtures out of the anchors while
preventing already-separated core centroids from reducing every TCPM loss to
nearly zero.

Prototype buffers are synchronized across distributed workers and saved in
normal Detectron2 checkpoints. Old single-bank experimental checkpoints are
loaded into the labelled bank for compatibility.

Stage II explicitly warm-starts both branches from the selected supervised
checkpoint and updates the EMA teacher from iteration 0. In this mode the
trainer does not subsequently reload `MODEL.WEIGHTS` into only the student;
the startup log must report a maximum teacher/student parameter difference of
zero. The checkpoint must use the same bilinear upsampling architecture as the
TopoWheat configuration; the SAPA checkpoint belongs to the released-project
baseline and is not a compatible TopoWheat warm start. TCPM writes only the
labelled bank during iterations 0--5k, ramps its losses during 5k--15k, admits
pseudo cores with confidence at least 0.8 at 15k, and ramps their anchor
contribution to 0.2 by 25k. Pseudo queries use separate reliability and
semantic-confidence gates.
`CORE_STRATEGY` selects `reliable`, `eroded`, or `topology` anchor evidence;
`QUERY_MODE` selects the diagnostic centroid path or the default hard-region
path. `LEAVE_ONE_DOMAIN_OUT` controls per-sample domain exclusion.

TensorBoard logs the teacher EMA decay, a chunked full-model teacher--student
parameter RMS every 2,500 iterations, bank drift, raw TCPM losses, hard-query
distance, anchor coverage, and stem--leaf ranking activation. A raw loss near
zero is therefore distinguishable from a batch with no available anchor.

The competition's anonymous domains are aligned with the named unlabelled
folders as follows: domain1--domain9 correspond to CIMMYT, ETHZ, INRAE, NJAU,
RRES, ULiege, UQ, USASK, and UTokyo. The mapping was verified against matching
images in the complete labelled release for the first eight domains; UTokyo is
the sole remaining competition source for domain9. The held-out test domain0
corresponds to Arvalis and uses the distinct internal domain id 9.

### BAZR

`MODEL.TOPOWHEAT.BAZR.AUX_HEADS_ENABLED` adds detached high- and low-resolution
auxiliary heads. Detaching their decoder features ensures that training these
heads does not alter the global segmentation path used by the TRPL+TCPM
ablation.

`MODEL.TOPOWHEAT.BAZR.ENABLED` wraps evaluation with broken-aware selective
zoom. Candidate windows are ranked by entropy, skeleton endpoint density,
internal-scale disagreement, and stem probability. NMS retains `TOPK` windows,
which are enlarged to `ZOOM_SIZE` and inferred with the shared segmentor.
The lightweight gate is trained from high/low auxiliary predictions and is
reused to blend local and global scores at inference. A Hann spatial taper
keeps the learned replacement gate from creating crop-edge seams.

BAZR and the released dense multi-scale TTA are intentionally mutually
exclusive so their compute and gains can be measured separately.

## Configurations

| Configuration | TRPL | TCPM | Auxiliary heads | BAZR inference |
| --- | --- | --- | --- | --- |
| `competition_stage2_stem4500.yaml` | off | off | off | off |
| `competition_topowheat_trpl.yaml` | on | off | off | off |
| `competition_topowheat_trpl_tcpm.yaml` | on | on | off | off |
| `competition_topowheat_bazr_train.yaml` | on | on | on | off |
| `competition_topowheat_bazr.yaml` | on | on | on | on |

## Reference baseline

The selected Stage I supervised baseline has validation mIoU `73.10942` and
competition-test mIoU `69.4389`. TRPL and later modules must be compared
against these fixed values using a validation-selected checkpoint.

## Commands

Create the architecture-matched supervised warm start when it is not already
available:

```bash
CONFIG_FILE=configs/gwfss/experiments/competition_baseline.yaml \
OUTPUT_DIR=outputs/competition_baseline_seed2025 \
bash stage1_train.sh
```

Train the complete training-time method:

```bash
bash topowheat_train.sh
```

The default warm start is
`outputs/competition_baseline_seed2025/model_best.pth`. Override
`TEACHER_CKPT` when the matching supervised run has another output name; the
diagnosed run, for example, used
`outputs/competition_baseline_v0/model_best.pth`.

Start the refactored TRPL, with or without TCPM, in a new `OUTPUT_DIR` without
`--resume`. Use `--resume` only to continue a checkpoint produced by this
four-parameter TRPL implementation. A v2 TRPL/TCPM checkpoint has a different
loss graph and optimizer history even though its model weights remain readable.

Train the TRPL ablation:

```bash
CONFIG_FILE=configs/gwfss/experiments/competition_topowheat_trpl.yaml \
OUTPUT_DIR=outputs/competition_topowheat_trpl_seed2025 \
bash topowheat_train.sh
```

For a remote smoke test, append configuration overrides after the script, for
example `SOLVER.MAX_ITER 20 TEST.EVAL_PERIOD 0 SOLVER.CHECKPOINT_PERIOD 20`.
Use a disposable output directory for that run.

Train the pure TRPL+TCPM ablation without BAZR heads:

```bash
CONFIG_FILE=configs/gwfss/experiments/competition_topowheat_trpl_tcpm.yaml \
OUTPUT_DIR=outputs/competition_topowheat_hardquery_seed2025 \
bash topowheat_train.sh
```

TCPM component ablations have dedicated configs under
`configs/gwfss/experiments/tcpm_ablations/`. They cover the legacy teacher
handoff, reliable/eroded anchors, centroid or all-reliable queries, immediate
activation, labelled-only memory, and removal of per-sample
leave-one-domain-out. The existing `competition_topowheat_trpl.yaml` is the
no-prototype row, and `competition_topowheat_trpl_tcpm.yaml` is the full row.

Evaluate one checkpoint on validation with one global forward:

```bash
SPLIT=val MODE=global \
CHECKPOINT=outputs/competition_topowheat_hardquery_seed2025/model_best.pth \
bash topowheat_eval.sh
```

Evaluate the same checkpoint on validation with BAZR:

```bash
SPLIT=val MODE=bazr \
CHECKPOINT=outputs/competition_topowheat_hardquery_seed2025/model_best.pth \
bash topowheat_eval.sh
```

Use `SPLIT=test` only after validation-based checkpoint and hyperparameter
selection is complete. The full evaluation matrix, including the released
project's dense zoom-in inference, is documented in `EVALUATION.md`.

The old experiment commands remain valid:

```bash
bash stage1_train.sh
bash stage2_train.sh
```
