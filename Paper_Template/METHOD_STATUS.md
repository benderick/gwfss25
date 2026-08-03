# Manuscript Method Status

Status: **quarantined draft, not submission ready**.

The current title, abstract, method, experiments, figures, and conclusion still
describe TRPL, TCPM, and BAZR. Full-validation audits reject the central premises
of all three components:

- TRPL reliability does not beat confidence at matched class coverage.
- Its stable skeleton does not beat a matched-confidence skeleton.
- Class-balanced pseudo supervision amplifies stem errors during training.
- TCPM depends on those rejected pseudo targets and has no independent causal
  evidence.
- BAZR's router has no measured utility target, while its fusion gate is trained
  on a different prediction pair from the one used at inference.

The manuscript also contains result placeholders and unexecuted claims for the
full-region and low-label protocols. Those claims are hypotheses, not results.

Locked facts that may be used as references:

- Stage-I validation mIoU: 73.10942.
- Stage-I competition-test mIoU: 69.4389.
- The released winner's complete Stage-II recipe and dense test scaling are
  established baselines, not new contributions of this manuscript.

Do not repair the old prose by renaming its modules. Rewrite the paper only
after the decision gates in `../METHOD_REDESIGN.md` select an empirically
supported direction. Any retained method must show at least +0.50 validation
mIoU point over the locked baseline under the preregistered comparison, followed
by repeated-run verification and one locked test evaluation.
