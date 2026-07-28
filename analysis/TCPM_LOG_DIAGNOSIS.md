# TCPM log diagnosis

Source: `exp_data.md`, 1,138 JSON records covering iterations 49--56,899.
The run is incomplete; it cannot be used as the final 90k result.

## Observed validation trajectory

| Iteration | mIoU | Stem IoU | TCPM state |
| ---: | ---: | ---: | --- |
| 5k | 72.5463 | 42.2318 | inactive |
| 10k | 72.5463 | 42.2318 | inactive |
| 15k | 72.5463 | 42.2318 | inactive |
| 20k | 72.5463 | 42.2318 | inactive |
| 25k | 71.9288 | 41.2874 | labelled memory only; loss scale 0 |
| 30k | 72.2543 | 41.5701 | loss scale about 0.5 |
| 35k | 72.1201 | 40.9122 | loss scale about 1.0 |
| 40k | 70.9404 | 40.5296 | pseudo blend about 0.1 |
| 45k | 71.1644 | 40.5245 | pseudo blend about 0.2 |
| 50k | 72.0778 | 40.5967 | full TCPM |
| 55k | 70.7422 | 40.4436 | full TCPM |

The identical 5k--20k evaluations show that the evaluated EMA teacher was
frozen. The first degradation occurs by 25k while every TCPM loss is still
zero, so that drop cannot be caused by prototype gradients.

## Optimization diagnostics

From 25k onward, the mean weighted TCPM losses are:

- contrastive: approximately `1e-5` for both labelled and pseudo branches;
- domain alignment: approximately `2e-4` to `4e-4`;
- stem--leaf hard negative: exactly zero in all 638 logged records for both
  branches.

The former core-centroid query therefore became an already-solved
self-consistency task. The pseudo-bank drift after opening is around `6e-7`,
which confirms stability but not usefulness. A small loss can mean either a
good representation or a dead objective; the old logging could not distinguish
those cases from missing anchors.

## Code-level causes and corrections

1. The selected supervised checkpoint was loaded into the SSL ensemble, but a
   later `resume_or_load(False)` could reload `MODEL.WEIGHTS` into only the
   student. The revised warm-start path skips this second overwrite.
2. The teacher stayed frozen for 20k and was then replaced with the student
   using EMA decay zero. The revised Stage II synchronizes both branches once
   from the selected checkpoint and uses continuous EMA from iteration 0.
3. Clean core centroids both wrote and queried memory. The revised TCPM keeps
   cores as write-only anchor evidence and mines the 25% most displaced pixels
   from the wider reliable region as loss queries.
4. New logs expose raw losses, query distance, anchor coverage and hard-negative
   activation, in addition to memory drift.

## Restart checks

- Use a new output directory and do not resume the diagnosed checkpoint.
- Warm-start from the matching bilinear supervised checkpoint, not the SAPA
  checkpoint used by the released-project baseline.
- Confirm startup reports teacher/student maximum parameter difference `0`.
- Confirm `ssl/teacher_ema_decay` is `0.9996` from iteration 0 and never becomes
  `0` at 20k.
- Confirm `ssl/teacher_student_param_rms` has no forced collapse at 20k.
- At 5k, inspect anchor coverage before judging raw loss magnitude.
- Compare `tcpm_centroid_query.yaml`, `tcpm_all_reliable_queries.yaml`, and the
  full hard-query configuration before committing to a complete seed sweep.
