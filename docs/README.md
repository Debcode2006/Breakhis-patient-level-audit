# Documentation

Start at the [repository README](../README.md) for the result and the protocol.
This directory holds the working record behind it.

## Method and operation

| Document | Contents |
|---|---|
| [`COMMANDS.md`](COMMANDS.md) | Every command in the study, with expected output, timings and troubleshooting. |
| [`retrieval.md`](retrieval.md) | The retrieval-augmented memory: the evidence that fixed each design choice (Part I), the decision register D1–D10 (Part II), the architecture as shipped (Part III), the pre-registered evaluation protocol (Part IV), the single-bank design (Part V), the operator handbook (Part VI), and the diagnosis of why the finished module does not improve accuracy (Part VII). |
| [`FIGURE_STYLE.md`](FIGURE_STYLE.md) | The figure house style. [`scripts/paper_style.py`](../scripts/paper_style.py) is its executable form — no figure script re-decides anything in it. |
| [`RESEARCH_LOG.md`](RESEARCH_LOG.md) | The full chronological record: how each result arrived, the decision register, and what did not work and why. Longer than the paper and more honest about dead ends. |

## Results

Each report is self-contained, names the artifacts it reads, and can be checked
against the committed JSON in [`analysis/`](../analysis/).

| Report | Question it closes |
|---|---|
| [`results/classifier_ladder.md`](results/classifier_ladder.md) | Cross-experiment fold and held-out analysis for Exp1–Exp3n. |
| [`results/threshold_calibration.md`](results/threshold_calibration.md) | Does a validation-tuned decision threshold recover the accuracy AUC says is there? |
| [`results/embedding_geometry.md`](results/embedding_geometry.md) | UMAP projections and separation metrics — direct evidence that SupCon reshapes the space. |
| [`results/magnification_audit.md`](results/magnification_audit.md) | What the magnification block measurably does, and why the retrieval base encoder is Exp3n. |
| [`results/retrieval_heldout.md`](results/retrieval_heldout.md) | The memory module on the 16-patient held-out test set. |
| [`results/retrieval_key_ablation.md`](results/retrieval_key_ablation.md) | 43 key definitions over five folds — the key is not the bottleneck. |
| [`results/crossencoder_screen.md`](results/crossencoder_screen.md) | Does an *un-shared* encoder fix retrieval? Pooled out-of-fold screen; verdict RED, test set untouched. |
| [`results/foundation_baseline.md`](results/foundation_baseline.md) | Frozen CTransPath plus a linear probe, on the same splits and metrics. |

## Reading order

If you are auditing a number in the paper: **README → the results report that owns
that table → the analysis JSON it cites.**

If you are extending the code: **README (architecture) → `retrieval.md` Part III
(how a module attaches to the forward dict) → `COMMANDS.md` (how to run it).**
