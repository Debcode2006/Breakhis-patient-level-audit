# docs/results/foundation_baseline.md — the CTransPath foundation-model baseline

**The foundation-model baseline of paper §3.8 — implemented, run on all five folds, and settled.**

*This is the complete record of the experiment: what was built, why it is built that
way, every number it produced, what those numbers mean for the paper, and what it
deliberately does not do. It is written to the same standard as `docs/retrieval.md` /
`docs/results/retrieval_key_ablation.md` — you should not need to open another file to understand
or defend it. How to run it: `docs/COMMANDS.md` §13.*

---

## Contents

1. [Headline](#1-headline)
2. [Why this experiment, and what "done" means](#2-why-this-experiment-and-what-done-means)
3. [Design](#3-design)
4. [Leakage: five places it is asserted](#4-leakage-five-places-it-is-asserted)
5. [Results](#5-results)
6. [What it means for the paper](#6-what-it-means-for-the-paper)
7. [Debatable choices, pre-answered](#7-debatable-choices-pre-answered)
8. [What this does not do](#8-what-this-does-not-do)
9. [Implementation reference](#9-implementation-reference)
10. [Reproducibility](#10-reproducibility)

---

## 1. Headline

Four results, all measured on the permanent 16-patient held-out test set, 5 folds.

**① The foundation model ties. That is the informative outcome, and it was
pre-registered as such.** The pre-registration named three possible outcomes and said
all three were publishable. This is the middle one: *"It ties you → direct evidence
for your ceiling argument (the task is saturated under honest evaluation)."* At its
locked pooled-OOF threshold CTransPath scores **0.8985** image accuracy against
exp3n's 0.8981 and exp1's 0.9064 — a spread of 0.008, i.e. inside the
patient-quantisation band the whole project reports its effects in. **Nothing has
moved the held-out number outside that band**: not the FPN, not the magnification
embedding, not SupCon, not retrieval memory, and now not pathology-specific
pretraining on 15 M slides.

**② It has the best image AUC in the entire study — and the worst accuracy at 0.5.**
AUC **0.9668** (against exp3's 0.9607, exp3n's 0.9593, exp1's 0.9559) with accuracy
@0.5 of **0.8580** (the lowest of any model here). Those coexist only if the model
*ranks* well and *sits* badly, which is exactly `docs/RESEARCH_LOG.md` headline finding ①. Its
accuracy-optimal threshold is **0.209** — the most miscalibrated model measured,
against exp3n's 0.540. **An entirely different encoder, pretrained on entirely
different data by an entirely different objective, independently reproduces the
project's central mechanism.** That is much stronger evidence for the claim than the
Swin ladder alone could ever be.

**③ At the locked threshold it gets every patient right: 80/80 fold-patient
decisions, 16/16 in all five folds, patient AUC 1.000 in 5/5.** At 0.5 it gets
75/80. The five errors are one patient (mucinous `MC-14-16456`) in four folds.

**④ Its error profile is *different*, not merely similar — and that is the most
actionable thing here.** On the project's pre-registered tracked cases, CTransPath
**solves two of the three hardest** and **breaks one the Swin ladder never had
trouble with**:

| tracked patient | subtype | y | exp3n mean P(mal) | **CTransPath** | change |
|---|---|:--:|---:|---:|---|
| `SOB_M_DC-14-12312` low-grade ductal | DC | 1 | 0.712 (4/5) | **0.930 (5/5)** | **solved** |
| `SOB_B_TA-14-16184` tubular adenoma | TA | 0 | 0.370 (4/5) | **0.039 (5/5)** | **solved** |
| `SOB_M_PC-14-9146` papillary, rare | PC | 1 | 0.726 (5/5) | 0.614 (5/5) | comparable |
| `SOB_M_DC-14-20636` borderline | DC | 1 | 0.986 (5/5) | 0.648 (4/5) | worse |
| `SOB_M_MC-14-16456` mucinous | MC | 1 | 0.849 (5/5) | **0.451 (1/5)** | **broken** |

The two cases the Swin models spent the whole project failing on — the
well-differentiated ductal that reads benign, and the tubular adenoma that mimics
carcinoma — are *not hard for CTransPath at all*. In exchange it loses mucinous
carcinoma, which the Swin models found easy. This is **error decorrelation**, the
exact quantity `docs/results/retrieval_key_ablation.md` §7.4 identified as the binding constraint
on the retrieval module, appearing here for free and much larger than the exp1
cross-encoder control ever produced.

---

## 2. Why this experiment, and what "done" means

### 2.1 The objection it closes

Paper §1 and §3.3 all name the same
risk as the project's single biggest reviewer objection:

> *"Why train Swin-Tiny from ImageNet weights when a frozen pathology foundation
> model + a linear head is a two-day baseline?"*

There is no argument that wins that exchange except a row in the table. It was ranked
"#1 ... it removes your single biggest reviewer objection", and estimated
2–3 days. **It is done.** End-to-end wall-clock, on the 4 GB RTX 3050: **≈ 3 minutes**
(2 min 36 s of it is the one-off encode).

### 2.2 What was actually delivered

- A frozen pathology foundation model (**CTransPath**, 27.5 M params) probed on the
  **exact** existing splits, the **exact** existing eval transform, and the **exact**
  existing metric functions.
- All 5 folds trained and scored on the held-out test set.
- Threshold calibration under the project's existing pooled-OOF protocol.
- Per-patient and per-subtype breakdowns.
- Output files byte-compatible with `scripts/test.py`, so every downstream analysis
  reads it with no special case.
- A second preset (`expfm_mlp`) for the "optionally a small MLP" capacity check,
  implemented but not run.

### 2.3 What is still open on this row

Two things, both listed honestly in §8: **no patient-clustered bootstrap interval**
has been computed for it yet, and the **CTransPath → exp3n cross-encoder retrieval
run** (the "free bonus") has not been done. Neither blocks writing
Methods or the main table's point estimates.

---

## 3. Design

Three stages, deliberately mirroring the Retrieval Memory's two-stage discipline
(**frozen encoder → cached store → tiny fitted head**), so this adds no coupling to
the trainer, the backbone, the FPN or the losses:

```
Stage F1  scripts/extract_foundation_features.py
          frozen CTransPath -> all 7,909 protocol images -> ONE .npz     2 min 36 s
Stage F2  scripts/train_linear_probe.py
          per fold: standardiser (train rows only) + 1,538-param head    2-9 s / fold
Stage F3  scripts/test_linear_probe.py
          the permanent 16-patient held-out set                          ~1 s / fold
```

Two presets, switched with one token exactly like the ladder:

| preset | run dir | head | trainable params |
|---|---|---|---:|
| `expfm` | `expfm_ctranspath_linear` | `Dropout(0.1) → Linear(768 → 2)` | **1,538** |
| `expfm_mlp` | `expfm_ctranspath_mlp` | `→ Linear(768→512) → ReLU → Dropout → Linear(512→2)` | 394,754 |

`expfm` is the reported row; `expfm_mlp` is a sensitivity check, not a second claim.

**The encoder is frozen and never fine-tuned**, and that is enforced rather than
intended: `build_foundation_encoder` calls `requires_grad_(False)` on every
parameter, and Stage F1 asserts the trainable count is zero before encoding a single
image.

### 3.1 Why CTransPath — and why the lack of UNI access is an advantage

UNI is HuggingFace-gated and unavailable. That constraint turns out to *improve* the
experiment, exactly as predicted ("the CTransPath gift"):

| | params | pretrained on |
|---|---:|---|
| exp3n (Swin-Tiny + FPN) | 30,843,388 | ImageNet-1k |
| **CTransPath** | **27,520,038** *(measured)* | ~15 M TCGA + PAIP histology patches, SRCL |
| UNI (ViT-L/16) | ~307 M | pathology |

At 27.5 M against 30.8 M the two encoders are **param-matched to within 11%**. So any
gap between the rows is about **what the encoder was pretrained on**, not **how big
it is**. A UNI comparison (10× larger) would have confounded precisely that. State
this in Methods — it converts a limitation into a tighter control.

CTransPath is also architecturally close: a Swin-Tiny whose linear patch embedding is
replaced by a small CNN stem. So the comparison is nearly "same architecture, same
size, different pretraining data and objective."

### 3.2 Where the weights come from, and the one check that must not be removed

The published CTransPath checkpoint (Wang et al., *Medical Image Analysis* 2022)
ships as a Google-Drive `.pth` requiring a vendored, patched timm 0.5.4. We instead
load the **ungated, timm-native HuggingFace mirror**
`1aurent/swin_tiny_patch4_window7_224.CTransPath`, which needs no vendored timm.
Two timm-version details had to be absorbed (`rap_mst/foundation/encoders.py`):

1. **NHWC.** timm ≥ 0.9 keeps Swin activations channels-last, so our `ConvStem`
   permutes its NCHW conv output. The original TransPath stem flattens to NLC for
   timm 0.5.4 and would silently mis-shape here.
2. **The `downsample` key layout.** The weights are in the pre-0.9 layout, where
   `downsample` sits at the *end* of stage *i* rather than the start of stage *i+1*.
   timm's own `checkpoint_filter_fn` performs exactly that remap — the same code path
   timm uses for the original MSRA Swin weights — but it does so **silently**.

That silence is the hazard, and it is the kind of bug that produces a *plausible*
wrong answer. `timm.create_model(..., pretrained=True)` does **not** raise if its
remap drops a tensor; it leaves that block randomly initialised, and a partly-random
Swin still emits sensible-looking features and a sensible AUC. So
`verify_encoder_load()` re-downloads the raw checkpoint and compares **every**
pretrained tensor *by value* against the built model, and raises if anything failed
to arrive. Stage F1 prints the outcome:

```
weights verified          : 183/183 checkpoint tensors identical in the model ✓
```

> **Do not relax this check to make a timm upgrade pass.** It is the only thing
> standing between this row and a half-random encoder.

### 3.3 The eval transform: the project's, not CTransPath's

CTransPath's own timm data config asks for bicubic resize with `crop_pct=0.9`; this
repo resizes to 224×224 bilinearly with no crop. **We use the repo's**, because the
row's job is to sit in a table whose other rows were produced that way — a row
measured under different preprocessing is not the controlled comparison
the comparison asks for. Both are ImageNet-normalised (which is what CTransPath
was trained with), so this is a resize-policy difference only. It is recorded
verbatim in the cache's `meta.transform` and echoed by every stage:

```
Resize((224, 224)) -> ToTensor -> Normalize(ImageNet mean/std)
[rap_mst.data.transforms, train=False]
```

### 3.4 Everything pinned to the ladder, so the comparison stays fair

Every knob that could flatter the baseline is set to whatever exp1–exp3n actually
did, not to whatever scores best:

| | exp1–exp3n | `expfm` |
|---|---|---|
| splits | `splits/breakhis_splits.json`, patient-level | **same file, same folds** |
| eval transform | `Resize(224) → ToTensor → Normalize(ImageNet)` | **same** |
| class imbalance | inverse-frequency CE weights | **same formula** |
| selection metric | `accuracy` (image-level val) | **same** |
| early stopping | patience on the monitor | same semantics, patience 20 |
| optimiser / schedule | AdamW + cosine to `min_lr` | **same** |
| metrics | `rap_mst.utils.metrics` | **same functions** |
| threshold policy | pooled OOF, never per fold (P2) | **same** |
| test set | permanent 16 patients, never touched | **same** |
| seed | 42 | **same** |

### 3.5 Why a separate script, not `--experiment expfm` through `train.py`

Because the honest version of this experiment has **nothing to train through the
existing loop**. With the encoder frozen and its output cached, a fold is a
`[4541 × 768]` tensor that lives on the GPU and converges in seconds. Routing it
through `Trainer` would mean re-encoding 4,541 images every epoch to update 1,538
parameters — the same waste `exp5` avoids by reusing exp3n's checkpoints.

`expfm` *is* registered in `EXPERIMENT_PRESETS` anyway, so `EXPERIMENT_DIRS` — and
therefore every analysis script that resolves `runs/<experiment>` through it — picks
it up with **no edits**. The risk that creates (someone runs
`train.py --experiment expfm` and silently trains a Swin into the baseline's run
directory, poisoning the table) is closed by `assert_not_foundation()`, wired into
`train.py`, `test.py` and `build_memory_bank.py`:

```
$ python scripts/train.py --experiment expfm --fold 0
'expfm' is a frozen foundation-model baseline and cannot be run by scripts/train.py:
it has no Swin backbone to build or load.
  Stage F1: python scripts/extract_foundation_features.py
  Stage F2: python scripts/train_linear_probe.py --experiment expfm --fold <k>
  Stage F3: python scripts/test_linear_probe.py  --experiment expfm --fold <k>
```

### 3.6 Output compatibility — why nothing downstream needed special-casing

Stage F3 writes `test/test_metrics.json` and `test/test_predictions.csv` with the
**same names and the same columns** `scripts/test.py` writes. `diagnose_folds.py` and
`threshold_calibration.py` therefore read this experiment as they read exp1–exp3n,
and the main table's rows come out of one procedure rather than two.

One addition was needed. `threshold_calibration.py` obtained validation predictions
by rebuilding the model with `build_model` and re-running `best.pt` — impossible for
a probe head over cached features. Stage F2 now saves `val_predictions.csv` per run,
and `val_predictions()` prefers it when present. **The Swin runs predate the file and
fall through to the old path — unchanged behaviour, byte-identical numbers** — while
any future run that saves one gets a free speed-up (5 folds of expfm calibrate in
seconds instead of ~1 h of GPU inference).

---

## 4. Leakage: five places it is asserted

The one design question a reviewer will pick at is *"you cached all 82 patients'
features in one file."* The answer:

**The encoder is frozen and was pretrained on TCGA/PAIP. It never sees a BreaKHis
label, and no gradient flows back into it.** The cache is therefore the analogue of
the dataset itself, not of a memory bank. Everything that *is* fitted from labels —
the standardiser and the head — is fitted on a fold's TRAIN rows only. Asserted five
deep, in the same spirit as the retrieval module's four layers:

1. **`setup_all()`** builds the encode list from the **splits file**, not from
   "everything under `dataset_root`", and asserts it covers exactly 82 patients — an
   out-of-protocol image cannot enter the cache.
2. **`split_views()`** asserts train / val / test patient disjointness **before a
   single feature row is read**, for both Stage F2 and Stage F3 — one implementation,
   so the two cannot drift apart.
3. **`rows_for()`** raises if a requested patient has **no** rows. A silently-empty
   split would make a fold train on 4/5 of its data and never say so.
4. **The `Standardizer` records its own provenance** — split, fold, row count,
   patient count — stored in the checkpoint and **printed at test time**:
   `{'split': 'train', 'fold': 0, 'n_rows': 4541, 'n_patients': 52}`. Stage F3 applies
   the stored one verbatim; re-fitting statistics on the test set would be a leak no
   metric would reveal.
5. **Stage F3 re-checks the checkpoint's own recorded train/val patient lists**
   against the test patients, so a mismatched `--fold` cannot quietly score a probe
   against its own training patients.

Plus two guards that are not about leakage but about not shipping a meaningless
number:

- Stage F1 raises on non-finite features and on near-constant features (a collapsed
  encoder would still yield a plausible probe accuracy on a 70%-malignant set).
- `guard_probe_health` raises a banner if the probe fails to beat the **training
  majority-class rate** — the degenerate all-one-class probe that reads as ~0.70
  accuracy and looks fine.

Neither fired. Every assertion passed on every fold.

---

## 5. Results

RTX 3050 Laptop, seed 42, deterministic. Encoding **2 min 36 s** for 7,909 images.
Probe fit **7.6 / 8.4 / 9.0 / 3.5 / 2.4 s** for folds 0–4.

### 5.1 The feature cache (Stage F1)

| | |
|---|---|
| rows | 7,909 (**82 patients** ✓) |
| feature dim | 768 |
| benign / malignant rows | 2,480 / 5,429 |
| magnification counts | 40: 1,995 · 100: 2,081 · 200: 2,013 · 400: 1,820 |
| feature L2 norm range | 2.885 … 4.638 |
| mean per-dim std | 0.0867 (non-degenerate ✓) |
| non-finite values | 0 ✓ |
| **weights verified** | **183/183 checkpoint tensors identical ✓** |
| file | `analysis/foundation/ctranspath/features.npz`, **22.6 MB** |

### 5.2 Validation, per fold (Stage F2, at the selected checkpoint)

| fold | best epoch | epochs run | fit s | val acc | val AUC | sens | spec | pat acc | pat AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 65 | 85 | 7.6 | 0.8542 | 0.9419 | 0.833 | 0.898 | 0.7857 | 1.0000 |
| 1 | 72 | 92 | 8.4 | 0.8869 | 0.9559 | 0.866 | 0.928 | 0.8462 | 0.9722 |
| 2 | 81 | 100 | 9.0 | 0.8463 | 0.9611 | 0.787 | 0.955 | 0.9231 | 1.0000 |
| 3 | 21 | 41 | 3.5 | 0.9382 | 0.9666 | 0.965 | 0.856 | 0.9231 | 1.0000 |
| 4 | 7 | 27 | 2.4 | 0.8617 | 0.9235 | 0.964 | 0.675 | 0.9231 | 1.0000 |
| **mean** | | | | **0.8775** | **0.9498** | 0.883 | 0.862 | 0.8802 | 0.9944 |

Note the pooled-OOF image accuracy (0.8775) sits **above** the Swin ladder's ~86.5%
that is flagged as the budget gap — with 1,538 trainable parameters
and no training budget at all. That is worth one sentence in the Discussion: it says
the ~86.5% is not a capacity problem.

### 5.3 Held-out test set, per fold (Stage F3, threshold 0.5)

| fold | image acc | image AUC | sens | spec | patient acc | patient AUC |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.8294 | 0.9567 | 0.788 | 0.937 | 0.9375 | 1.0000 |
| 1 | 0.9008 | 0.9720 | 0.880 | 0.954 | 1.0000 | 1.0000 |
| 2 | 0.8584 | 0.9715 | 0.815 | 0.972 | 0.9375 | 1.0000 |
| 3 | 0.8724 | 0.9708 | 0.837 | 0.965 | 0.9375 | 1.0000 |
| 4 | 0.8288 | 0.9632 | 0.777 | 0.965 | 0.8750 | 1.0000 |
| **mean** | **0.8580** | **0.9668** | **0.819** | **0.959** | **0.9375** | **1.0000** |

**Patient AUC is 1.0000 in 5/5 folds.** As with the retrieval study
(`docs/results/retrieval_heldout.md` §5), that means the patient level is **saturated and has no
headroom to distinguish this row from the others in either direction**. Report it,
then set it aside — do not quote patient-level parity as evidence for anything.

### 5.4 Threshold calibration — mandatory for this row

CTransPath is the **most miscalibrated model in the study**: pooled-OOF optimal image
threshold **0.209**, against exp3's 0.360 and exp3n's 0.540. Accuracy at a fixed 0.5
cut badly understates it.

| level | locked thr (pooled OOF) | val acc | test mean @0.5 | test mean @locked | Δ |
|---|---:|---:|---:|---:|---:|
| image | **0.209** | 0.8822 | 0.8580 | **0.8985** | **+0.0405** |
| patient | 0.255 | 0.9394 | 0.9375 | 1.0000 | +0.0625 |

Sensitivity / specificity move 0.819 → **0.899** and 0.959 → 0.898: calibration buys
back exactly the sensitivity the low cut was costing. This is the largest calibration
gain measured in the project (exp3's was +0.0097).

Per-fold thresholds swing **0.017 → 0.512**, reproducing the same instability
`docs/results/threshold_calibration.md` §4 found on 13-patient val sets. **Per-fold
calibration is not used**, per P2.

> **Caveat, applied consistently.** The **image**-level pooled-OOF threshold is fitted
> on ~8,000 validation images and is stable — quote it. The **patient**-level one is
> fitted on 66 validation patients, and that report already showed it overfits and
> *hurts* test for exp1/exp2/exp3n. The 16/16 above is reported as a fact about this
> fitted cut, **not** as a claim that expfm is the best patient-level model.

### 5.5 Per-patient results, 5-fold mean

Mean P(malignant) over the 5 fold models. `pat@.5` / `pat@lk` = how many of the 5
fold-level patient decisions are correct, at 0.5 and at the locked 0.209.

| patient | subtype | y | n img | mean P(mal) | pat@.5 | pat@lk | img acc@.5 | img acc@lk |
|---|---|:--:|---:|---:|:--:|:--:|---:|---:|
| `SOB_B_F-14-23060CD` | F | 0 | 57 | 0.004 | 5/5 | 5/5 | 1.000 | 0.989 |
| `SOB_B_F-14-21998CD` | F | 0 | 137 | 0.034 | 5/5 | 5/5 | 0.988 | 0.956 |
| `SOB_B_TA-14-16184` | TA | 0 | 132 | **0.039** | 5/5 | 5/5 | 0.989 | 0.945 |
| `SOB_B_A-14-22549CD` | A | 0 | 132 | 0.164 | 5/5 | 5/5 | 0.879 | 0.750 |
| `SOB_M_MC-14-16456` | MC | 1 | 178 | **0.451** | **1/5** | 5/5 | 0.440 | 0.639 |
| `SOB_M_PC-14-9146` | PC | 1 | 90 | 0.614 | 5/5 | 5/5 | 0.636 | 0.796 |
| `SOB_M_DC-14-20636` | DC | 1 | 111 | 0.648 | 4/5 | 5/5 | 0.674 | 0.836 |
| `SOB_M_MC-14-19979C` | MC | 1 | 62 | 0.713 | 5/5 | 5/5 | 0.745 | 0.868 |
| `SOB_M_DC-14-14926` | DC | 1 | 73 | 0.738 | 5/5 | 5/5 | 0.792 | 0.888 |
| `SOB_M_LC-14-15570C` | LC | 1 | 125 | 0.893 | 5/5 | 5/5 | 0.930 | 0.978 |
| `SOB_M_DC-14-2985` | DC | 1 | 59 | 0.928 | 5/5 | 5/5 | 0.966 | 0.997 |
| `SOB_M_DC-14-12312` | DC | 1 | 146 | **0.930** | 5/5 | 5/5 | 0.973 | 0.995 |
| `SOB_M_DC-14-16336` | DC | 1 | 60 | 0.968 | 5/5 | 5/5 | 0.997 | 1.000 |
| `SOB_M_DC-14-16188` | DC | 1 | 87 | 0.978 | 5/5 | 5/5 | 0.993 | 1.000 |
| `SOB_M_DC-14-17614` | DC | 1 | 129 | 0.982 | 5/5 | 5/5 | 0.995 | 1.000 |
| `SOB_M_DC-14-2523` | DC | 1 | 75 | 0.986 | 5/5 | 5/5 | 1.000 | 1.000 |

**Patient decisions correct: 75/80 at 0.5 → 80/80 at the locked threshold.** All five
errors at 0.5 are the *same patient*, `MC-14-16456`, sitting at 0.451 — one
patient, 0.05 of probability away from the cut. That is the whole @0.5 accuracy story.

### 5.6 Per-subtype image accuracy, 5-fold mean

| subtype | class | n patients | img acc @0.5 | img acc @locked |
|---|:--:|---:|---:|---:|
| F (fibroadenoma) | benign | 2 | 0.994 | 0.973 |
| TA (tubular adenoma) | benign | 1 | 0.989 | 0.945 |
| A (adenosis) | benign | 1 | 0.879 | 0.750 |
| DC (ductal carcinoma) | malignant | 8 | 0.924 | 0.964 |
| LC (lobular carcinoma) | malignant | 1 | 0.930 | 0.978 |
| **PC (papillary)** | malignant | 1 | **0.636** | 0.796 |
| **MC (mucinous)** | malignant | 2 | **0.593** | 0.754 |

The residual-error structure is *partly* the project's known one (PC stays hard — it
is rare everywhere) and *partly* new: **MC is CTransPath's worst subtype and was not
a problem for the Swin ladder**, while TA — the Swin ladder's signature false
positive — is essentially solved (0.039 mean probability, five-nines benign).

### 5.7 The tracked cases, against exp3n

`docs/RESEARCH_LOG.md` §9.3's pre-registered tracking set, with the new column:

| patient | subtype | y | exp1 | exp3 | exp3n | **expfm** | exp3n pat✓ | **expfm pat✓** |
|---|---|:--:|---:|---:|---:|---:|:--:|:--:|
| `SOB_M_DC-14-12312` low-grade ductal | DC | 1 | ≈0.66 | 0.577 | 0.712 | **0.930** | 4/5 | **5/5** |
| `SOB_M_PC-14-9146` papillary, rare | PC | 1 | ≈0.72 | 0.639 | 0.726 | 0.614 | 5/5 | 5/5 |
| `SOB_M_DC-14-20636` borderline | DC | 1 | — | 0.916 | 0.986 | 0.648 | 5/5 | 4/5 |
| `SOB_M_MC-14-16456` mucinous | MC | 1 | — | 0.852 | 0.849 | **0.451** | 5/5 | **1/5** |
| `SOB_B_TA-14-16184` tubular adenoma | TA | 0 | ≈0.33 | 0.316 | 0.370 | **0.039** | 4/5 | **5/5** |

**Read this table as the most interesting new artifact here.** Two of the three
cases that carried essentially all the test error across every Swin experiment —
`DC-12312` (well-differentiated ductal reading benign-like) and `TA-16184` (tubular
architecture mimicking carcinoma) — are **not hard for a pathology-pretrained
encoder**. In exchange, mucinous carcinoma, trivially easy for the Swin ladder,
becomes CTransPath's dominant failure.

The two encoders are **wrong in different places at comparable overall accuracy.**

---

## 6. What it means for the paper

### 6.1 The objection is closed — say so explicitly

the standing reviewer objection ("Where is the foundation model?") is answered, on its own
splits, with a param-matched encoder. This is the row that must not be missing, and
it is no longer missing.

### 6.2 It strengthens the central thesis rather than complicating it

The paper's thesis is that *none of the architectural additions produce a
statistically defensible gain, the reliable movements are removal and threshold, and
the task is saturated under honest evaluation.* A frozen pathology foundation model
with **1,538 trainable parameters** landing at 0.8985 against exp3n's 0.8981 is the
cleanest possible confirmation. Add one sentence to the Abstract.

### 6.3 It gives finding ① an external witness

`docs/RESEARCH_LOG.md` §1① — *"the accuracy the ladder lost was a threshold shift, not a
discrimination loss"* — was, until now, demonstrated entirely *within* one
architecture and one training pipeline. A reviewer could reasonably ask whether it is
an artifact of that pipeline. It is not: CTransPath, a different architecture
pretrained on different data with a different objective, exhibits the same
dissociation **more extremely than any Swin ladder variant** (best AUC in the study,
worst accuracy at 0.5, optimal threshold 0.209). Promote this to a paragraph in the
magnification/threshold Results subsection — it is nearly free evidence.

### 6.4 It hands the paper its most concrete Future Work

`docs/results/retrieval_key_ablation.md` §7.4 and `docs/RESEARCH_LOG.md` §12① concluded that the only lever
that moved retrieval was **encoder identity**, and that a bank built from a
*different* frozen encoder nearly doubled (×1.88) error-decorrelation with the head.
The tracked-case table in §5.7 is that same phenomenon, measured directly and much
larger — a genuinely different error profile at matched accuracy. That is exactly the
prior that is "a far better error-decorrelation prior than the exp1
cross-encoder control." **Whether or not you run it, the Future Work section can now
name the mechanism *and* show the evidence for it in a table you already have.**

### 6.5 The parameter-efficiency argument survives, in the direction §2.5 permits

CTransPath is **smaller** than the Swin ladder (27.5 M vs 30.8 M), pathology-pretrained, and
it does not win. So the honest framing stays —
*"competitive accuracy at a fraction of the parameters of the transformer and
foundation-model approaches the field is converging on, on commodity hardware"* — and
**not** "fewest parameters", which DenseNet121 (~8 M, ~92.1%) would sink anyway. Do
not claim the Swin ladder beats a foundation model. Claim that under an honest protocol the
two are indistinguishable, which is the more interesting sentence and the one you can
defend.

### 6.6 Where the row goes

- **Table 1 (main benchmark)** — the expfm row, with the
  params column. `docs/RESEARCH_LOG.md` §9.1 already carries it.
- **Methods §4.4 / new subsection** — "how features
  were extracted, the linear-probe head": §3 of this document is that text.
- **Results** — one subsection, or a paragraph inside the main-benchmark subsection.
- **Discussion** — §6.2, §6.3 and §6.5 above.
- **Future Work** — §6.4.

---

## 7. Debatable choices, pre-answered

**"You cached all 82 patients' features in one file — that is leakage."**
No. The encoder is frozen, pretrained on TCGA/PAIP, sees no BreaKHis label and
receives no gradient; the cache is the analogue of the dataset, not of a memory bank.
Everything fitted from labels is fitted on train rows only, asserted in five places
(§4). Put the one-sentence version in Methods and cite the assertions.

**"You used your preprocessing, not CTransPath's."**
Deliberately (§3.3). The row has to be comparable with the rows beside it; both
pipelines are ImageNet-normalised at 224 px and differ only in resize policy. The
transform string is stored in the cache metadata and printed by every stage.

**"A linear probe under-uses a foundation model — fine-tune it."**
Two answers. First, the paper specifies a *frozen* encoder plus a linear
head — that is the standard, cheap, reproducible foundation-model baseline and it is
literally what the objection asks for ("a frozen pathology foundation model + a
linear head is a two-day baseline"). Second, `expfm_mlp` is a one-token capacity
sensitivity check. A full fine-tune stops being a *foundation-model baseline* and
becomes a second architecture with its own training-budget confound; it belongs in
Future Work.

**"You only report the calibrated number because the raw one is bad."**
Both are reported, in every table. The calibration protocol (**pooled OOF, never per
fold**) was fixed by `docs/results/threshold_calibration.md` as decision P2 long before
this experiment existed. The threshold is picked on validation only; the test set is
scored once at the locked value. And the @0.5 number is *also* reported, prominently,
including the fact that it is the worst in the study.

**"Patient accuracy 16/16 at a fitted patient threshold is cherry-picked."**
Flagged as exactly that in §5.4, using the same caveat the project already applies to
exp1/exp2/exp3n. The image-level threshold is the stable one; the patient-level one
is not quoted as a win.

**"Best AUC means it is the best model."**
No, and the paper should not say so. The AUC advantage (+0.006 over exp3) is well
inside the intervals this project reports its effects in, and no bootstrap has been
computed for it yet (§8). The defensible claim is *indistinguishable*, not *better*.

---

## 8. What this does *not* do

Stated plainly so nothing here is mistaken for more than it is.

- **No patient-clustered bootstrap intervals for this row.** The protocol asks
  for "the same bootstrap". With a 16-patient test set and a Δ of 0.0004 against
  exp3n at the locked threshold, the interval will straddle zero — which *is* the
  finding — but it must be computed before the number appears in a table with a CI
  column. This is the one outstanding item on this experiment.
- **The cross-encoder retrieval bonus was not run.** A CTransPath bank against
  the exp3n head was "the last retrieval shot worth taking", and
  §5.7 above now gives it a much better prior than it had. It is genuinely close — the
  cache is a `[7909 × 768]` table with patient ids and magnifications — but it needs
  `retrieval.key_encoder` (`docs/RESEARCH_LOG.md` §12①), and it is a **separate pre-registered
  experiment**. Running it unasked would be exactly the open-ended search
  is what this warns against.
- **`expfm_mlp` was implemented but not run.** One command (`docs/COMMANDS.md` §13.2).
- **No second foundation model.** CTransPath alone suffices for this row;
  the registry in `rap_mst/foundation/encoders.py` takes a second entry plus a config
  block if a Phikon/Virchow2 variant later clears its access terms.
- **No external validation (BACH), no training-budget experiment.** Both are out
  of scope here and untouched by this work.
- **`analysis/threshold_calibration/results.json` was not regenerated** for all five
  experiments; expfm's calibration went to a separate directory to avoid clobbering
  the existing exp1–exp3n artifact. Regenerate once, before building the table
  (`docs/COMMANDS.md` §13.5).
- **A duplicate fold-0 run directory exists** (`20260728_200307_train_fold0`) from the
  initial single-fold verification. Runs are never overwritten by design, and
  `find_runs` resolves each fold to the latest, so the reported numbers come from the
  five `2018xx` runs. Noted so nobody averages six directories over five folds.

---

## 9. Implementation reference

### 9.1 New files

| file | what |
|---|---|
| `rap_mst/foundation/__init__.py` | package doc + lazy imports (importing it does not drag in timm) |
| `rap_mst/foundation/encoders.py` | `ConvStem`, encoder registry, `build_foundation_encoder`, `verify_encoder_load` |
| `rap_mst/foundation/cache.py` | `FeatureCache` (save/load/select/provenance guard), `split_views`, `write_predictions_csv` |
| `rap_mst/foundation/probe.py` | `ProbeHead`, `Standardizer`, `fit_probe`, `predict`, `score`, `guard_probe_health` |
| `rap_mst/foundation/builder.py` | config → encoder spec / probe head / cache path |
| `scripts/extract_foundation_features.py` | Stage F1 |
| `scripts/train_linear_probe.py` | Stage F2 |
| `scripts/test_linear_probe.py` | Stage F3 |
| this report| this document |

### 9.2 Modified files

| file | change | affects existing results? |
|---|---|---|
| `config/config.yaml` | new `foundation:` block | **no** — nothing else reads it |
| `rap_mst/experiments.py` | `expfm` / `expfm_mlp` presets, `FOUNDATION_EXPERIMENTS`, `is_foundation()`, `assert_not_foundation()` | **no** — exp1–exp5 presets byte-identical |
| `rap_mst/data/datamodule.py` | `setup_all()` / `all_dataloader()` | **no** — purely additive |
| `scripts/train.py`, `scripts/test.py`, `scripts/build_memory_bank.py` | one `assert_not_foundation(...)` guard each | **no** — fires only for `expfm*` |
| `scripts/threshold_calibration.py` | prefer a saved `val_predictions.csv`; shared CSV reader; clearer log line | **no** — Swin runs have no such file and take the original path |
| `docs/COMMANDS.md` | new §13 | — |
| `docs/RESEARCH_LOG.md` | status + results | — |

### 9.3 Configuration

The whole experiment is one config block (`config/config.yaml`):

```yaml
foundation:
  encoder: ctranspath
  encoders:
    ctranspath:
      hub_id: "hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath"
      embed_layer: convstem
      feature_dim: 768        # asserted against the encoder's real output dim
      pool: avg
  cache_path: "analysis/foundation/{encoder}/features.npz"
  batch_size: 32
  amp: true
  probe:
    head: linear              # linear | mlp
    hidden_dim: 512
    dropout: 0.1
    standardize: true         # fitted on TRAIN rows ONLY
    use_class_weights: true
    epochs: 100
    batch_size: 256
    lr: 0.001
    weight_decay: 0.0001
    scheduler: cosine
    min_lr: 0.000001
    monitor: accuracy         # the SAME selection rule exp1-exp3n used
    early_stopping_patience: 20
```

Every one of these is overridable with `--set`, as everywhere else in the repo.

### 9.4 Artifacts produced

```
analysis/foundation/ctranspath/features.npz               22.6 MB, 7,909 x 768
analysis/foundation/ctranspath/extract_foundation_features.log
analysis/threshold_calibration_expfm/results.json
runs/expfm_ctranspath_linear/probe_fit_summary.json
runs/expfm_ctranspath_linear/<ts>_train_fold{0..4}/
    config.yaml  train.log  metrics.csv  tensorboard/  val_predictions.csv
    checkpoints/{best,last}.pt
    test/{test_metrics.json, test_predictions.csv, test.log}
```

The feature cache is safe to delete — it is deterministic and regenerates from the
frozen weights in ~3 minutes.

---

## 10. Reproducibility

- Seed 42, deterministic cuDNN, `torch 2.3.1+cu121`, **`timm 1.0.9` (pin it — §3.2)**.
- Weights: `1aurent/swin_tiny_patch4_window7_224.CTransPath` — **public, not gated**,
  GPL-3.0, repo SHA `59c740e4ed91dda4ebe094b64e4cd2e015c31810`, 27,769,814 stored
  tensors of which 27,520,038 are model parameters. ~220 MB, cached under
  `~/.cache/huggingface` on first use.
- The cache stores: encoder name, hub id, embed layer, pooling, feature dim,
  parameter count, timm version, weight-verification counts, dataset root, splits
  path, image size, the transform string, the AMP flag, the seed and a timestamp.
- Every checkpoint stores: the exact config, the probe spec, the standardiser **and
  its fitting provenance**, the encoder metadata, the feature-cache path, the fold,
  and the fold's train/val patient lists.
- A cache and a config that disagree on the encoder **raise** rather than quietly
  producing a number for the wrong row — the same discipline `MemoryBank` applies.
- Runs are never overwritten; each gets its own timestamped directory.

---

*Related: the paper (the brief), §2.5 (the param-matching argument), §3.3
(the objection this closes), §9 (the checklist) · `docs/COMMANDS.md` §13 (how to run it) ·
`docs/RESEARCH_LOG.md` §9.1 (the main table), §1① (the finding this independently reproduces),
§9.3 (the tracked cases), §12① (the cross-encoder direction this now supports) ·
`docs/results/threshold_calibration.md` §4, §6 (the P2 calibration protocol it obeys) ·
`docs/results/retrieval_key_ablation.md` §7.4 (error decorrelation, the quantity §5.7 measures).*
