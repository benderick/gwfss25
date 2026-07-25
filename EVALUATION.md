# GWFSS checkpoint evaluation

## Which network is evaluated?

Stage 2 checkpoints contain both `modelTeacher.*` and `modelStudent.*`.
Training configuration `SSL.EVAL_WHO: TEACHER` evaluates the EMA teacher, and
`model_best.pth` is selected by the teacher's validation `sem_seg/mIoU`.
Evaluation therefore defaults to `EVAL_WHO=TEACHER`. Setting
`EVAL_WHO=STUDENT` is intended only for a teacher/student ablation.

Stage 1 checkpoints contain the supervised model under `modelTeacher.*`.

## Splits

`SPLIT=val` evaluates the 99-image validation set and `SPLIT=test` evaluates
the 110-image test set. Both use their local `class_id` masks, so Detectron2's
semantic evaluator reports mIoU and per-class IoU directly.

Use validation results to choose checkpoints, thresholds, Top-K, and scaling
settings. Run the test split only after those decisions are fixed.

## Released project

One global forward:

```bash
SPLIT=val MODE=global bash project_eval.sh
SPLIT=test MODE=global bash project_eval.sh
```

The released dense multi-scale zoom-in and sliding-window inference:

```bash
SPLIT=val MODE=zoom bash project_eval.sh
SPLIT=test MODE=zoom bash project_eval.sh
```

The zoom mode uses the six configured test sizes corresponding to scales
1.0--3.5, horizontal flip, and overlapping 512-pixel sliding windows. It is
the project's Stage 3 method, not a single ordinary resize.

## TopoWheat

One global forward and BAZR must use the same checkpoint:

```bash
SPLIT=val MODE=global bash topowheat_eval.sh
SPLIT=val MODE=bazr bash topowheat_eval.sh

SPLIT=test MODE=global bash topowheat_eval.sh
SPLIT=test MODE=bazr bash topowheat_eval.sh
```

Override a checkpoint explicitly when needed:

```bash
SPLIT=val MODE=bazr \
CHECKPOINT=outputs/competition_topowheat_seed2025/model_best.pth \
bash topowheat_eval.sh
```

All output folders are separated by method and split under `outputs/`. The
important console/log entries are `sem_seg/mIoU`, `sem_seg/IoU-background`,
`sem_seg/IoU-head`, `sem_seg/IoU-stem`, and `sem_seg/IoU-leaf`. Dense project
zoom results carry the `_TTA` suffix.
