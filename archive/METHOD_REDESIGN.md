# Method Redesign: Evidence Before Architecture

Status: **the TRPL--TCPM--BAZR method is quarantined and must not be presented as
the paper's validated method.** This document defines the evidence required before
replacing it.

## 1. Constraints and reference points

- Experiments run on a separate machine. The local checkout is used for code and
  static checks only.
- The locked Stage-I reference is **73.10942 validation mIoU** and **69.4389 test
  mIoU**.
- Model selection and method design use train/validation data only. The
  competition test split remains locked until the method and checkpoint are
  fixed.
- A component is retained because it beats a fair control, not because its loss
  decreases or its mechanism sounds plausible.
- No method can be guaranteed to improve an unseen evaluation before it is run.
  "Must work" is implemented here as a retention rule: a candidate that misses
  the predeclared effect-size gate is removed rather than explained away.
- The new method should expose at most one optimization weight. Geometry and
  inference budget are reported as operating conditions, not searched as a large
  hyperparameter grid.

## 2. Disposition of every current claim

| Item | Intended claim | Evidence | Decision |
|---|---|---|---|
| TRPL reliability | Multi-view confidence and agreement identify cleaner pseudo-labels than confidence alone. | On all 99 validation images, stem precision is 0.782349 for TRPL and 0.784316 for class-matched confidence. Overall accepted accuracy is also lower by 0.000218. The same negative result occurs for Stage I, TRPL-5k, and TRPL-30k checkpoints. | Reject. |
| TRPL stable skeleton | Cross-view skeleton intersection supplies uniquely clean topology. | Against a confidence-matched teacher skeleton, precision changes by -0.011399, -0.011222, and -0.031381 at Stage I, 5k, and 30k. clDice changes by +0.001152, +0.000051, and -0.013345. | Reject. |
| TRPL semantic loss | Class-balanced pseudo supervision recovers rare stems. | The effective per-pixel stem multiplier is 16.22 at Stage I and 16.96 at 5k. By 30k, predicted stem area grows 67% from 5k while false stem pixels grow 170%; validation mIoU falls to 70.65 in the audit and 71.14 in the training evaluation. | Reject. The failure is positive feedback, not insufficient loss weight. |
| TCPM | Topology cores form stable cross-domain anchors that improve unknown-domain features. | Its pseudo bank and pseudo queries depend on rejected TRPL targets. With only 99 labeled images over nine domains, each domain memory is thin. Earlier logs showed almost-zero contrastive/compactness losses and an exactly zero hard-negative loss. The implementation introduces roughly twenty coupled choices without positive causal evidence. | Reject rather than tune. |
| BAZR router | Entropy, endpoints, scale disagreement, and stem probability locate windows where zooming fixes errors. | The four-term score is hand-written and receives no routing supervision. No experiment yet measures whether a selected window improves after a local high-resolution forward. | Reject current router. |
| BAZR fusion | A learned gate decides when a local result should replace the global result. | During training, the gate mixes two detached auxiliary heads from one global forward. During inference, the same gate mixes the main global prediction with a true zoomed crop prediction. It is never trained on the inference pair it must arbitrate. | Reject current gate. |
| BAZR crop source | A selected crop restores image detail discarded by global resizing. | Every competition image is already 512 x 512, which is also the normal test size. Thus the ordinary test tensor and source file contain the same spatial samples; no sensor pixels were discarded. Enlarging a crop may still change the network's effective sampling, but it cannot restore absent image information. | Retract the detail-restoration claim. Treat crop magnification as an empirical intervention, not as evidence of recovered source detail. |
| Unified topology story | TRPL supplies evidence to TCPM during training and BAZR during inference. | The shared upstream evidence fails its matched controls, so dependence makes the stack less identifiable rather than more coherent. | Reject the unifying premise. |
| Stage-I topology supervision | The supervised warm-up already learns explicit stem topology. | The paper states that Stage I uses only the original Mask2Former objectives. The code agrees: `_supervised_topowheat_losses` contributes only when TCPM or BAZR auxiliary heads are enabled, while `competition_baseline.yaml` enables neither. | It is not part of the locked baseline and cannot be counted as an existing contribution. Test established supervised topology losses only as explicit controls. |
| Continuous EMA lifecycle | Warm-starting both branches and updating the teacher from iteration zero is a method contribution. | The refactored path is internally consistent, but checkpoint warm-start plus continuous EMA is standard Mean Teacher engineering. It also changes the released guided-distillation lifecycle, which freezes the teacher until 20k and then resets it from the student. | Keep only as a separately controlled protocol choice; do not claim novelty or mix it into a module ablation. |
| Stem-aware 4,500 selection | The Stage-II unlabeled subset is neutral data. | The list was produced by the winning method by ranking predicted stem-pixel proportion and retaining 500 samples per domain. It is already a task-aware intervention. | Retain for winner reproduction and as a strong inherited baseline. Compare it with domain-balanced random selection; do not claim it as new. |
| SAPA | Dynamic upsampling is the main detail solution. | The original project reports only 0.0007 mIoU over its baseline. | Keep only as an optional historical baseline, not a central contribution. |
| Guided distillation | Stem-aware use of unlabeled data improves the model. | The original project reports 0.7291 to 0.7432 (+0.0141). | Retain as a strong existing baseline; do not rename it as a new contribution. |
| Stage-II winner recipe attribution | The reported Stage-II gain isolates guided distillation or any one new mechanism. | The released recipe simultaneously uses SAPA, the inherited stem-aware 4,500-image subset, a 90k schedule, and a frozen-teacher-then-reset lifecycle. Existing local logs do not contain a completed factorial ablation that separates these choices. | Treat the complete recipe as one strong inherited baseline. Do not assign its gain to a single component until controlled rows are run. |
| Dense test-time scaling | More input pixels materially improve detail segmentation. | The original project reports 0.7432 to 0.7704 (+0.0272), with 512 to 768 single-scale inference already improving about two points. | Retain as the measured upper-bound direction. |
| Full-region generalization | The proposed method improves unknown-region generalization and low-label regimes. | Dataset registration and a supervised region-split config exist, but the claimed 10/25/50/100% matrix, method comparisons, and unknown-region results have not been run and the manuscript tables remain placeholders. | Planned evaluation only. Remove all affirmative claims until the locked competition method is selected and the full-region experiments are completed. |

The full TRPL measurements and definitions are recorded in
[`TRPL_AUDIT.md`](TRPL_AUDIT.md).

## 3. Root-cause diagnosis

The old design repeatedly substitutes a plausible proxy for the actual event of
interest:

1. Agreement is treated as correctness, although two EMA views share parameters
   and can agree on the same error.
2. Skeleton stability is treated as useful topology, although thinning a wrong
   stem mask produces a stable wrong skeleton.
3. Distance from a prototype is treated as correctable domain shift, although it
   can also be class overlap, annotation ambiguity, or a bad anchor.
4. Entropy and endpoints are treated as zoom benefit, although a hard region can
   remain hard at high resolution and a confident low-resolution error can be
   highly correctable.
5. Auxiliary-scale fusion is treated as supervision for global/local crop
   fusion, although the train and inference input pairs differ.

The metrics also expose a gradient-allocation failure. Equal class averaging
turns a small number of correlated stem false positives into much larger
per-pixel gradients than the dominant classes. EMA delays the visible damage but
does not remove it; as the student enters the teacher, the false-stem loop grows.

## 4. Evidence-backed problem statement

The useful observation is simpler than the old topology chain:

- All competition images are 512 x 512. A 512-to-768 input change therefore
  upsamples the same image samples; it provides finer internal feature sampling
  but no additional sensor information.
- Stems are a detail minority: their median occupied area is about 3% when
  present, and their median local diameter is about 8.25 pixels at 512
  resolution.
- Median stem centreline retention drops from about 86% at stride 8 to 40% at
  stride 16 and zero at stride 32 in the repository's morphology analysis.
- Dense enlarged-input inference is the largest measured improvement in the
  original project, but that result combines effective model scale, interpolation,
  and test-time ensembling and does not identify which mechanism caused the gain.

Therefore the next causal question is **whether magnification changes enough
errors to justify extra computation**. Resolution allocation is only a possible
downstream question; it is not yet the paper's method.

## 5. Prior-art collision

The first redesign idea was to supervise a router with the realized loss
difference between global and magnified-crop predictions. That idea is not a
defensible novelty claim:

- RAZN learns zoom/break decisions from a reward built directly from the
  segmentation-loss difference between high- and low-magnification views. It is
  the closest collision with a realized-utility router.
- GeoAgent learns cost-aware scale selection for semantic segmentation.
- Patch Proposal Network learns to select difficult image patches for
  high-resolution processing.
- HRDA combines global context and high-resolution local crops with learned
  scale attention.
- SparseRefine uses uncertainty to invoke a sparse high-resolution extractor and
  then fuses the two resolutions.
- PointRend adaptively allocates computation to uncertain points.
- HarmonySeg already combines deep/shallow features, vesselness-style auxiliary
  supervision, and a growth/suppression loss for tubular segmentation.
- DeformCL learns an explicit deformable centerline representation rather than
  treating a generic skeleton penalty as a new structural representation.
- TopoTTA applies topology-enhanced adaptation to tubular segmentation under
  domain shift.
- Spatial-Aware Topological Loss already improves persistent-topology matching
  by adding spatial-domain evidence.

Consequently, neither selective magnification, loss-difference utility, nor
global/local fusion will be presented as a new contribution. They can still be
used as controls or diagnostic interventions.

The same restriction applies to generic centreline losses, deep/shallow detail
branches, growth/suppression penalties, and topology-aware domain adaptation.
These are established method families, not available names for repackaging the
rejected TRPL--TCPM chain. Cross-resolution distillation is also prior art. A
future contribution must have both a measured causal advantage and a narrower
claim than these broad ideas.

Primary sources:

- [RAZN, MICCAI 2018](https://arxiv.org/abs/1807.11113)
- [GeoAgent, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Seeing_Beyond_the_Patch_Scale-Adaptive_Semantic_Segmentation_of_High-resolution_Remote_ICCV_2023_paper.html)
- [PointRend, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.pdf)
- [Patch Proposal Network, AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6926)
- [HRDA, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1999_ECCV_2022_paper.php)
- [SparseRefine, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8392_ECCV_2024_paper.php)
- [HarmonySeg, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_HarmonySeg_Tubular_Structure_Segmentation_with_Deep-Shallow_Feature_Fusion_and_Growth-Suppression_ICCV_2025_paper.html)
- [DeformCL, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_DeformCL_Learning_Deformable_Centerline_Representation_for_Vessel_Extraction_in_3D_CVPR_2025_paper.html)
- [TopoTTA, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_TopoTTA_Topology-Enhanced_Test-Time_Adaptation_for_Tubular_Structure_Segmentation_ICCV_2025_paper.html)
- [Spatial-Aware Topological Loss, ICCV Workshops 2025](https://openaccess.thecvf.com/content/ICCV2025W/BIC/html/Wen_Topology-Preserving_Image_Segmentation_with_Spatial-Aware_Persistent_Feature_Matching_ICCVW_2025_paper.html)

## 6. Resolution intervention audit

`zoom_audit.sh` is a falsification test, not an implementation of the next
method. It asks five bounded questions on the frozen Stage-I model:

1. Does the audit exactly reproduce ordinary validation inference?
2. Does one fixed full-image 768-short-edge forward improve over 512 for this
   exact checkpoint?
3. Does magnifying a 256 x 256 native-image crop to 512 x 512 ever improve the
   same model on the same pixels?
4. Are those improvements concentrated in at most four windows per image?
5. Do entropy, margin, stem probability, or the old BAZR score locate the
   beneficial windows better than random selection?

The audit runs one frozen global model, recomputes a fixed candidate grid, and
uses deterministic overlap-add. Labels are used only to calculate an oracle and
must never enter a deployable policy. It reports both weighted-mean fusion and
local replacement because a single fusion convention can otherwise hide or
manufacture apparent headroom.

The full-image scale control also reports 512-only, 768-only, probability-mean
ensemble, confidence selection, and a per-pixel ground-truth oracle. The oracle
is diagnostic only. Per-class transition counts separate pixels recovered by
768 from pixels damaged by 768, which prevents an aggregate gain from hiding a
stem-specific regression.

No TRPL, pseudo-label topology loss, prototype memory, learned router, learned
fusion gate, or morphology-specific loss is active. The fixed controls are a
global 512 input from the pure `competition_baseline.yaml` configuration, one
full-image 768-short-edge input, 256-pixel windows, stride 128, and local resize
to 512. These values define one diagnostic condition; they are not a parameter
search.

All label-free scores and the random control use the same fixed NMS and forward
budget, so spatial diversification cannot be mistaken for a better score. The
ground-truth oracle omits NMS deliberately because it is a permissive mechanism
upper bound, not a deployable-policy comparison.

## 7. Pre-registered decision gates

Run `zoom_audit.sh` on the Stage-I validation checkpoint before implementing a
router.

### Gate 0: evaluation parity

The audit's raw global prediction must reproduce the locked validation mIoU
within 0.10 percentage point. A larger discrepancy invalidates the audit until
preprocessing/evaluation is fixed.

### Scale-control classification

The fixed 768-short-edge full-image forward is called materially useful only if
it improves global mIoU by at least **0.50 percentage point**. This is an
independent classification, not a substitute for the sparse gates below: dense
scale can succeed while a crop router fails, or vice versa.

### Gate 1: correctable headroom

The ground-truth greedy oracle with at most four local forwards must improve
global mIoU by at least **0.50 percentage point** under the preregistered
weighted-mean fusion. If not, sparse magnification is too weak for this
checkpoint and that direction is killed. Replacement and dense results remain
diagnostics and cannot override the gate post hoc.

### Gate 2: budget concentration

Oracle (K=4) must recover at least **50%** of the class-balanced NLL reduction
obtained by greedily adding every remaining positive-utility window. If gains
are diffuse across the whole image, a sparse router is the wrong tool.

### Gate 3: routing evidence

Before any learned router is considered, at least one preregistered label-free
score must beat random selection by at least **0.10 percentage point** at the
same budget, with a positive paired bootstrap interval after Bonferroni
correction over the four scores. The interval is calculated on paired per-image
mIoU gains, while the 0.10-point effect-size check uses aggregate dataset mIoU;
both must pass. If no cheap signal ranks magnification benefit, a router trained
on only 99 labeled images is not justified. A later learned router would
additionally have to beat entropy by at least **0.20 percentage point**
validation mIoU; even then, the prior-art collision prevents treating routing
alone as the paper's novelty.

### Gate 4: end-to-end value

Any eventual method must beat 73.10942 validation mIoU by at least **0.50 point**
over three seeds or paired deterministic runs, while reporting forward count,
latency, and memory. Only then is one locked test evaluation allowed. A method
that merely matches the baseline is not retained for its story.

## 8. Decisions after the audit

The audit deliberately does not preselect a replacement method.

| Measured result | Decision |
|---|---|
| No fixed 768 full-image gain and no all-window gain | Abandon the resolution story. Move to supervised-loss/data-quality audits and the established guided-distillation baseline. |
| Full-image 768 helps, but crop magnification does not | The gain needs global context or uniform resampling. Keep fixed dense scale as an accuracy/compute baseline; do not build a router. |
| Dense gain, but Gate 1 or 2 fails | Magnification helps diffusely. Test dense enlarged-input training and scale distillation as baselines; do not build a sparse router. Any new contribution must explain how training transfers that gain, not merely reuse high-resolution inference. |
| Gates 1 and 2 pass, but label-free selectors fail | Sparse headroom exists but is not observable from cheap global evidence. Do not fit a high-capacity router on 99 labels; study a different observable signal or stop. |
| Gates 1--3 pass | Budgeted magnification is viable as an engineering component. Because the component itself is prior art, a separate, literature-checked contribution is still required. |

This table separates an effect from a paper claim. Passing a gate means that an
intervention is useful enough to develop; it does not make that intervention
novel. Failing a gate kills the corresponding story even if a qualitative
example looks favorable.

Topology is reconsidered only as a **supervised baseline** on true labels. If
needed, Skeleton Recall Loss and clDice must first be compared under identical
training seeds; neither can be relabeled as a new loss. High-resolution-to-low-
resolution distillation is likewise prior art and must begin as a control rather
than a claimed contribution.

Relevant controls:

- [clDice, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf)
- [Skeleton Recall Loss, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9904_ECCV_2024_paper.php)
- [Resolution-path distillation for semantic segmentation, ISVC 2021](https://par.nsf.gov/servlets/purl/10335366)

## 9. Experiment order

1. Stage-I smoke audit on 10 validation images.
2. Stage-I full audit on all 99 validation images.
3. Compare random, entropy, margin uncertainty, stem probability, old BAZR score,
   oracle, and all-window policies at identical forward budgets.
4. Follow exactly one branch from the decision table. Do not add a learned
   component before its prerequisite gate passes.
5. Establish the strongest honest reference using supervised Stage I, the
   original guided-distillation Stage II, and fixed dense scale controls. Keep
   the released frozen-then-reset teacher lifecycle separate from any continuous-
   EMA variant so a lifecycle change cannot be attributed to a new loss.
6. Run the inherited Stage-II recipe as a single baseline first. Factor SAPA,
   data selection, and teacher lifecycle only if attribution is needed; never
   use the winner's aggregate gain as evidence for a new component.
7. Only after a candidate beats its controls, freeze its design and geometry and
   run the final validation comparison.
8. Run the full-region protocol only after the competition method is fixed. It
   tests transfer; it must not be used to invent or select the method.
9. Run the locked competition test split once for the selected checkpoint and
   policy.

If any gate fails, the next redesign starts from the measured failure instead of
adding another loss term. Until those measurements exist, this repository has a
rejected old method and a diagnostic protocol, **not a validated replacement
innovation**.

## 10. Remote commands

Run the smoke test first:

```bash
MAX_IMAGES=10 VISUALIZE_BEST=3 \
CHECKPOINT=outputs/competition_baseline/model_best.pth \
OUTPUT_DIR=outputs/zoom_audit_stage1_smoke \
bash zoom_audit.sh
```

Only if it completes and the reported global prediction is plausible, run the
pre-registered full audit:

```bash
MAX_IMAGES=0 VISUALIZE_BEST=10 \
CHECKPOINT=outputs/competition_baseline/model_best.pth \
OUTPUT_DIR=outputs/zoom_audit_stage1_full \
bash zoom_audit.sh
```

The decision requires the console summary and `summary.json`. Do not start
another 30k training run before choosing the branch in Section 8.
