# TopoWheat implementation

All extensions are registered under `MODEL.TOPOWHEAT` and are disabled by
default. The released supervised and guided-distillation configurations
therefore keep their original model structure and loss path.

## Modules

### TRPL

`MODEL.TOPOWHEAT.TRPL.ENABLED` enables topology-reliable pseudo-label
learning. The EMA teacher predicts aligned weak views at the scales listed in
`VIEW_SCALES`. Query predictions are composed into semantic probabilities and
used to calculate:

- normalized predictive entropy and multi-view Jensen-Shannon disagreement;
- class-adaptive reliable regions and continuous confidence weights;
- persistent stem skeletons across views;
- uncertain stem boundaries excluded from hard region supervision.

The student receives weighted semantic NLL and Dice losses, soft-clDice, and a
stable-skeleton consistency loss. A supervised stem topology loss is also
applied to labelled images. Set `LEGACY_QUERY_LOSS_WEIGHT` above zero only when
testing a hybrid with the released hard query pseudo-label loss.

### TCPM

`MODEL.TOPOWHEAT.TCPM.ENABLED` adds a topology-core prototype memory. Stem
prototypes use the persistent skeleton and its narrow neighbourhood; other
classes use eroded reliable interiors. The memory stores class prototypes per
source domain and derives global prototypes while excluding the rotating
held-out domain.

The module contributes prototype contrastive, domain compactness, and
stem-versus-leaf hard-negative losses. Prototype buffers are synchronized
across distributed workers and saved in normal Detectron2 checkpoints.
`CORE_STRATEGY` selects `reliable`, `eroded`, or `topology` pixels for the
prototype ablation, while `HELDOUT_ENABLED` controls the rotating held-out
domain experiment.

The competition's anonymous domains are aligned with the named unlabelled
folders as follows: domain1--domain9 correspond to CIMMYT, ETHZ, INRAE, NJAU,
RRES, ULiege, UQ, USASK, and UTokyo. The mapping was verified against matching
images in the complete labelled release for the first eight domains; UTokyo is
the sole remaining competition source for domain9.

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

## Commands

Train the complete training-time method:

```bash
bash topowheat_train.sh
```

Train the TRPL ablation:

```bash
CONFIG_FILE=configs/gwfss/experiments/competition_topowheat_trpl.yaml \
OUTPUT_DIR=outputs/competition_topowheat_trpl_seed2025 \
bash topowheat_train.sh
```

Train the pure TRPL+TCPM ablation without BAZR heads:

```bash
CONFIG_FILE=configs/gwfss/experiments/competition_topowheat_trpl_tcpm.yaml \
OUTPUT_DIR=outputs/competition_topowheat_trpl_tcpm_seed2025 \
bash topowheat_train.sh
```

Evaluate one checkpoint on validation with one global forward:

```bash
SPLIT=val MODE=global \
CHECKPOINT=outputs/competition_topowheat_seed2025/model_best.pth \
bash topowheat_eval.sh
```

Evaluate the same checkpoint on validation with BAZR:

```bash
SPLIT=val MODE=bazr \
CHECKPOINT=outputs/competition_topowheat_seed2025/model_best.pth \
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
