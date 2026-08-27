# Research Log

Research framework for **binary benign/malignant classification** on the
**BreaKHis** breast-histopathology dataset, with a single unified model trained
across all four magnifications (40X / 100X / 200X / 400X) under a strict
**patient-level** protocol.

The feature extractor is a **Hierarchical Swin Transformer + Feature Pyramid
Network (FPN)**. On top of it sits a modular ladder of optional components
(magnification embedding, SupCon projection head) and a **Retrieval Memory module
(v1)** — a non-parametric bank of frozen training features plus a ~150-parameter
fusion gate.

**This document is the complete record of the research.** It covers the protocol,
the preprocessing, the architecture, every experiment in chronological order, every
decision and the measurement that settled it, and — in equal detail — the parts that
did **not** work and why. You should not need to open another file to understand
what was done.

---

## Contents

1. [Headline findings](#1-headline-findings)
2. [Research protocol](#2-research-protocol)
3. [Data & preprocessing](#3-data--preprocessing)
4. [Architecture](#4-architecture)
5. [The experiment ladder](#5-the-experiment-ladder)
6. [Chronological research narrative](#6-chronological-research-narrative)
7. [Decision register](#7-decision-register)
8. [What did not work, and why](#8-what-did-not-work-and-why)
9. [Consolidated results](#9-consolidated-results)
10. [Repository, configuration, commands](#10-repository-configuration-commands)
11. [Reproducibility](#11-reproducibility)
12. [Status & roadmap](#12-status--roadmap)

---

## 1. Headline findings

Six results, in the order they were established. All are measured, and all
survived attempts to break them.

**① The accuracy the ladder "lost" was a threshold shift, not a discrimination
loss.** Adding the magnification embedding (exp2) and then SupCon (exp3) *lowered*
thresholded test accuracy (0.9054 → 0.8974 → 0.8858 image) while AUC held or rose
(0.9559 → 0.9550 → 0.9607). Each component traded sensitivity for specificity; on a
72%-malignant test set a benign-leaning boundary at a fixed 0.5 cut costs accuracy
even when ranking improves.

**② The culprit was the magnification embedding, and it was a 4-value bias term.**
Because the classification head is a single `Linear`, a magnification vector
*concatenated* to the feature collapses at the decision into **four scalar per-zoom
logit offsets**. Its total contribution to exp3's ranking is **+0.00018 AUC**. Its
cost is severe: it makes the retrieval key and the SupCon projection space
**99.6–100% magnification-locked** (25% is chance).

**③ Removing it (exp3n) improved almost everything, by subtraction.** exp3n = exp3
without the magnification block: image accuracy 0.8858 → **0.8986**, patient accuracy
0.9375 → **0.9750**, sensitivity 0.889 → **0.919**, AUC a wash (0.9607 → 0.9593). Its
accuracy-optimal threshold snaps back from 0.36 to **0.54**, so it is the only model
needing **no threshold calibration**. Its SupCon space un-locks (same-magnification
neighbour rate **0.976 → 0.335**) and posts the best embedding geometry measured
(binary silhouette **0.680**).

**④ The Retrieval Memory module was built, is correct, and does not improve the
classifier.** On the held-out test set with pre-registered thresholds:
ΔAUC = **−0.0003** (95% CI [−0.0010, +0.0008]), Δaccuracy = **−0.0108**, negative in
5/5 folds. The gate honestly collapses onto the parametric head (pooled
`w_param` = 0.799).

**⑤ The binding constraint is *encoder sharing*, not the key.** 43 (key,
key_transform) configurations were measured — every forward-dict vector, four FPN
poolings, every pyramid level, composites, whitening, PCA-dropping, and deleting the
classifier's exact decision direction. Retrieval AUC moves inside a **0.017 band**
and **0/39 beat the parametric head**. Key-space geometry does not predict retrieval
quality (`corr(log effective rank, p_img AUC) = +0.007` across a **338× range** of
effective rank). The one quantity that responds — error-decorrelation from the head —
**nearly doubles (×1.88)** the moment the key comes from a *different* encoder.

> A retrieval branch reading a frozen encoder cannot know anything that encoder does
> not already know. The redundancy is **semantic, not geometric**: you cannot subtract
> it out of the key, because it was never confined to a direction.

What the module *does* deliver, confirmed on held-out patients: a **+0.156 subtype
lift** over base rate in its retrieved neighbourhoods, with named, inspectable
archived slides — i.e. evidence, not accuracy.

**⑥ A frozen pathology foundation model ties — and reproduces finding ① from
outside.** **CTransPath** (27.5 M, pretrained on ~15 M TCGA/PAIP patches) probed
with a **1,538-parameter linear head** on the same splits scores **0.8985** image
accuracy at its locked threshold, against exp3n's 0.8981 and exp1's 0.9064 —
**16/16 patients in 5/5 folds, patient AUC 1.000**. Nothing in this project has
moved the held-out number outside that band: not the FPN, not magnification, not
SupCon, not retrieval, and not pathology-specific pretraining.

It also has the **best image AUC measured here (0.9668)** *and* the **worst accuracy
at 0.5 (0.8580)**, with the study's most extreme accuracy-optimal threshold
(**0.209** vs exp3n's 0.540). A different architecture, pretrained on different data
by a different objective, exhibits finding ① **more strongly than any Swin ladder
variant** — so it is not an artifact of this training pipeline.

And its **errors sit elsewhere**: it solves `DC-14-12312` (0.712 → **0.930**) and
`TA-14-16184` (0.370 → **0.039**), the two patients that carried almost all the Swin
ladder's test error, while breaking on mucinous `MC-14-16456` (0.849 → **0.451**),
which the ladder found easy. Two encoders, comparable accuracy, **wrong in different
places** — the error-decorrelation quantity ⑤ named as the binding constraint,
measured directly. Full record: `docs/results/foundation_baseline.md`.

---

## 2. Research protocol

| Item | Value |
|------|-------|
| Task | Binary classification (benign = 0, malignant = 1) |
| Dataset | BreaKHis (82 patients: 24 benign, 58 malignant, ~7.9k PNGs) |
| Magnifications | 40X, 100X, 200X, 400X (**one unified model** over all four) |
| Splitting | **Patient-level**, no image or patient leakage |
| Held-out test | 16 patients / 1,653 images, permanent, never used for training or selection |
| Cross validation | 5-fold, class-stratified, over the remaining 66 patients |
| Per fold | 52–53 train patients (~4,540 images) / 13–14 val patients (~1,700 images) |
| Metrics | Image-level **and** patient-level (mean-pooled per patient) accuracy, AUC, sensitivity, specificity |
| Seed | 42, everywhere |

Splits are generated **once** (`scripts/prepare_splits.py`), serialized to
`splits/breakhis_splits.json`, and reloaded verbatim by every experiment. The
generator *and* the loader assert there is no patient overlap between train / val /
test; the retrieval bank asserts it a third time at build and a fourth time per
query.

### 2.1 The split policy (and the bug it fixed)

The first version of the protocol stratified the test set by **class only**. That
let rare tumour subtypes clump, and it **pinned image-level test accuracy at ~82%
across every experiment** — the metric could not move regardless of what the model
did. The old split is preserved in the repo at
`splits/breakhis_splits.classstrat_backup.json` rather than deleted.

The replacement stratifies the test set by the **eight tumour subtypes**, under a
documented, seed-reproducible policy recorded as `split_policy` inside the JSON:

| Policy field | Value | Why |
|---|---|---|
| `test_stratified_by` | `tumor_subtype` | the test set must be *representative*, not just class-balanced |
| `fold_stratified_by` | `class` | the rarest subtype has too few patients to stratify across 5 folds |
| `reserve_subtypes` | `("PT",)` | only 3 phyllodes patients exist — too scarce to hold out, and needed in training. All 3 routed to CV |
| `max_test_image_fraction` | `0.12` (cap ≈ 185 images) | no single slide may dominate the image-level metric. `SOB_M_LC-14-15570` (201 images) was routed to CV by this rule |
| `test_subtype_quota` | A:1, F:2, TA:1, DC:8, LC:1, MC:2, PC:1 | the resulting 16-patient test set |

This is a **representativeness** policy applied once and reported as-is — *not*
tuning to a target accuracy. It is stated here because a reviewer will (correctly)
ask whether the test split was chosen after seeing results; it was chosen by rule,
the rule is in the JSON, and the superseded split is still in the repo.

### 2.2 Test-set composition — remember this number

The test set is **4 benign / 12 malignant patients**, i.e. **~72% of test images are
malignant**. Nearly every "surprising" accuracy result in this project is downstream
of that prevalence interacting with a fixed 0.5 threshold.

### 2.3 Patient identity

Patient id = the filename prefix `SOB_<class>_<subtype>-<year>-<slide>`
(e.g. `SOB_B_A-14-22549AB`). Parsing lives **only** in
`rap_mst/data/breakhis.py` (`patient_id_from_filename`, `subtype_from_patient_id`)
and is never re-implemented elsewhere — including by the memory bank, which imports
it for its `subtype` column.

---

## 3. Data & preprocessing

### 3.1 Loading

- Source of truth: `data.dataset_root` (`D:/Downloads/BreaKHis_v1`).
- `BreaKHisDataset` returns the image tensor plus **rich metadata** —
  label, tumour subtype, patient id, magnification index — which is what makes
  patient-level metrics, magnification routing and the memory bank possible without
  a second pass over the filesystem.
- `collate.py` batches the tensors and keeps the metadata as plain lists.
- `datamodule.py` builds the train/val/test loaders for a given fold, plus
  `setup_bank(fold)` / `bank_dataloader()` for the retrieval bank — the fold's TRAIN
  patients under **eval** transforms, no shuffle, no `drop_last` (using the train
  loader would augment the keys and silently drop the last partial batch).

### 3.2 Augmentation — train-time only, exact values

Configured under `augmentation:` in `config/config.yaml`, applied in this order:

| Step | Value | Note |
|---|---|---|
| `Resize` | `(224, 224)` | `data.image_size` |
| `RandomHorizontalFlip` | `p = 0.5` | histology has no canonical orientation |
| `RandomVerticalFlip` | `p = 0.5` | same |
| `RandomRotation` | `degrees = 15` | mild; large rotations introduce border artefacts |
| `ColorJitter` | brightness `0.1`, contrast `0.1`, saturation `0.1`, hue `0.02` | conservative — H&E stain colour *is* signal, so hue is nearly frozen |
| `ToTensor` | — | |
| `Normalize` | ImageNet mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` | the Swin backbone is ImageNet-pretrained |
| `RandomErasing` | `p = 0.0` | **disabled** by default |

**Validation / test / bank encoding use `Resize → ToTensor → Normalize` only**, so
evaluation is deterministic and the memory bank's keys are not augmented.

### 3.3 Class imbalance handling

- `loss.use_class_weights: true` — inverse-frequency weighting on CrossEntropy
  (the dataset is ~31% benign at the image level, 24:58 at the patient level).
- `data.balanced_sampler: false` — oversampling was available but not used; the
  loss weighting was sufficient and keeps the epoch definition honest.

### 3.4 Contrastive views

`data.two_view: false`. SupCon draws its positives from **same-label images inside
the batch** rather than from a second augmented crop of the same image. The two-view
path exists (it emits `[B, 2, D]` pairs) but doubles the forward batch, which does
not fit the 4 GB target hardware at batch 16. This is a real limitation on SupCon —
see §8.

---

## 4. Architecture

`RAPMSTModel` (`rap_mst/models/rap_mst_model.py`) is a **thin assembler** of
independent, optional components. There is no per-experiment model class; every
experiment is the same object with different flags.

```
Input image  [B, 3, 224, 224]
   │
   ▼
SwinBackbone            swin_tiny_patch4_window7_224 (timm, ImageNet-pretrained)
   │                    emits the LIST of hierarchical stage maps, NCHW-normalized
   │                    (timm Swin is NHWC; the wrapper permutes). It does NOT pool.
   │   [B,  96, 56, 56]  [B, 192, 28, 28]  [B, 384, 14, 14]  [B, 768, 7, 7]
   ▼
FPN                     lateral 1x1 convs + top-down pathway + 3x3 output convs
   │                    GroupNorm (batch-size independent), out_channels = 256
   │   [B, 256, 56, 56]  [B, 256, 28, 28]  [B, 256, 14, 14]  [B, 256, 7, 7]
   ▼
FeatureFusion           GAP each pyramid level, then combine (mode: concat)
   │   features  [B, 1024]          <- 4 levels x 256
   ├──────────────────────────────────────────────► the RETRIEVAL KEY (D1)
   ▼
[MagnificationEmbedding]   optional (exp2/exp3): nn.Embedding(4, 64), concat
   │   embeddings  [B, 1024] or [B, 1088]
   ├───────────────► [ProjectionHead]  optional (exp3/exp3n): 512 hidden -> 128, L2-normed
   │                     projections [B, 128]  -> SupCon
   ▼
ClassificationHead      Dropout(0.1) -> Linear(-> 2)      (hidden_dim: null = linear)
    logits [B, 2]
```

**30,843,388 parameters** total, all trainable (exp3n; exp2/exp3 add the 256-parameter
embedding table plus its share of the head).

### 4.1 The forward dict — the extension contract

`forward()` returns a **dict**, not a tensor. This is the single most load-bearing
design decision in the repo, because it is what let the entire retrieval module be
added without touching the model, the trainer or the losses.

| key | shape | meaning |
|---|---|---|
| `logits` | `[B, 2]` | classifier output |
| `features` | `[B, 1024]` | fused vector **after FPN+fusion, before magnification**. The canonical vector a memory/prototype module indexes |
| `embeddings` | `[B, 1024]` or `[B, 1088]` | feature after magnification fusion; equals `features` when magnification is off |
| `projections` | `[B, 128]` | SupCon space (when the projection head is on) |
| `fpn_features` | list of 4 maps | the spatial pyramid, for future dense / region-level modules |

**Do not use `embeddings` or `projections` as a retrieval key on a
magnification-enabled encoder.** §6.4 explains why with numbers.

### 4.2 Losses

`CombinedLoss` is a weighted sum, so new objectives are additive:

```
loss = ce_weight * CrossEntropy(logits, y)        # class-weighted
     + supcon_weight * SupCon(projections, y)     # temperature 0.07
```

`ce_weight = 1.0` everywhere; `supcon_weight = 0.4` for exp3 / exp3n, `0.0` otherwise.

### 4.3 Training engine

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr `3e-4`, weight decay `0.05`, betas `(0.9, 0.999)` |
| Scheduler | cosine to `min_lr = 1e-6` |
| Batch | 16 × grad-accum 2 = **effective 32** |
| Precision | AMP (mixed precision) — essential on 4 GB |
| Grad clip | 1.0 |
| Epochs | 10 (exp1), 15 (exp2 / exp3 / exp3n) |
| Monitor | `accuracy` (image-level val accuracy) for checkpointing + early stop |
| Early stopping | patience 10 |
| Throughput | ~47 img/s on an RTX 3050; ~95 s/epoch; peak ~2.2 GB VRAM |

`trainer.diagnostics.register(name, fn)` (`DiagnosticRegistry`) lets a future module
add per-epoch diagnostics **without editing the training loop**.

### 4.4 The Retrieval Memory module (v1)

Non-parametric, post-fusion, and deliberately **two-stage** (frozen encoder → bank →
gate), so it adds no training-loop coupling. Total trainable parameters: **147**.

```
features [B,1024] (L2-normalised)
   │
   ▼
┌──────────────────── MemoryBank — ONE table (D5) ─────────────────────┐
│  level | key[1024] | label | subtype | patient_id | mag              │
│  image   one row per training image, sharded by magnification        │
│  slide   one row per training PATIENT (centroid, derived at build,   │
│          mag = ALL — slide rows are not magnification-routed)        │
│                                                                      │
│  TWO VIEWS, RANKED SEPARATELY — never one top-k over the union:      │
│   image view: route=same_mag, top-M cosine -> cap <=3/patient        │
│               -> top-k=15 -> softmax(sim/0.07) -> p_img              │
│   slide view: route=all,     top-k=5, cap=1  -> softmax     -> p_slide│
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────── FusionGate (147 params) ─────────────────────┐
│ in: [p_param, |p_param-0.5|, agreement(p_img), mean top-1 sim,       │
│      n_distinct_patients_frac]                                       │
│ 5 -> 16 -> 3, softmax -> (w_param, w_img, w_slide)                   │
│ p_final = w_param*p_param + w_img*p_img + w_slide*p_slide            │
└──────────────────────────────────────────────────────────────────────┘
```

Pipeline:

```
Stage A   train the base encoder per fold                 -> checkpoints/best.pt
Stage B1  scripts/build_memory_bank.py                    -> ONE .npz per fold
          freeze best.pt, encode that fold's TRAIN patients only
Stage B2  scripts/train_gate.py                           -> ONE gate.pt
          fit on POOLED out-of-fold validation (66 patients), never per fold
Stage C   scripts/test.py --retrieval                     -> p_final + exemplars
```

Measured cost, fold 0 on an RTX 3050: encoding 4,541 training images ≈ **2 min**;
bank = 4,541 image rows + 52 centroids = **17.5 MB** compressed `.npz`; gate fit =
seconds; the Stage C pass adds no measurable time. Exact cosine over a magnification
shard is the right choice at this scale — no FAISS, no ANN index.

**Leakage defence, four layers deep:** a fold's bank holds only that fold's TRAIN
patients (asserted at build), `assert_disjoint` re-checks at load, `block_query_patients`
masks a query against its own patient at ranking time, and Stage C logs a
`leakage guard passed` line naming the count.

**Observability.** Each stage echoes its full config, logs bank/gate provenance, and
raises WARNING banners for the known failure modes (`check_retrieval_health`): short
neighbourhoods, one-patient-dominated neighbourhoods, top-1 ≈ 1.0 (possible leak),
top-1 too low (key mismatch), magnification lock, gate collapse. The D5 invariant
("one store, two independent rankings") is **verified at runtime on the first batch
of every run**, not merely asserted in a test — a refactor that merged the ranking
raises `D5 violation` instead of silently degrading.

---

## 5. The experiment ladder

All experiments are the **same** modular model; only config flags change
(`rap_mst/experiments.py`). Switching is one token: `--experiment exp3n`. Every rung
uses the Swin + FPN backbone — presets never touch the FPN.

| Preset | Run dir | Mag embed | Projection | Loss | Epochs |
|---|---|:--:|:--:|---|:--:|
| `exp1` | `exp1_swin_cls` | — | — | CE | 10 |
| `exp2` | `exp2_swin_mag_cls` | ✓ | — | CE | 15 |
| `exp3` | `exp3_swin_mag_supcon_cls` | ✓ | ✓ | CE + 0.4·SupCon | 15 |
| `exp3n` | `exp3n_swin_supcon_cls` | — | ✓ | CE + 0.4·SupCon | 15 |
| `exp5` | *(reuses exp3n)* | — | ✓ | CE + 0.4·SupCon | — |

**Not a rung of the ladder:** `expfm` / `expfm_mlp` are the frozen **CTransPath**
foundation-model baseline — an *external* comparison, not an ablation of this model.
They have no Swin backbone, are fitted by `scripts/train_linear_probe.py` on cached
frozen features, and `train.py` / `test.py` / `build_memory_bank.py` refuse them
loudly. See §5.1 below and `docs/results/foundation_baseline.md`.

`exp5` = exp3n **+ the Retrieval Memory** (`retrieval.enabled: true`). It needs **no
encoder training**: `encoder_experiment("exp5") → "exp3n"`, so the retrieval scripts
load exp3n's frozen checkpoints. Training `exp5` directly would burn ~10 GPU hours
re-training an identical encoder.

### exp1 — Swin + FPN + CE (baseline)

**What it is.** The backbone alone: hierarchical Swin stages → FPN → GAP-fusion →
linear head, cross-entropy.

**Result.** Test image accuracy **0.9054**, patient accuracy **0.9875** (79/80
fold-patient decisions correct), AUC 0.9559, sensitivity 0.924, specificity 0.856.

**Finding.** The FPN's multi-scale aggregation already gives a very discriminative
backbone. The ~13 typical test patients are essentially solved (0.96–1.00 per-patient
image accuracy). Its weakness is the boundary cases and the lowest specificity of the
ladder — it errs slightly toward calling benign tissue malignant.

**It is still the accuracy baseline to beat.** Nothing added later overtook it on raw
image accuracy.

### exp2 — + magnification embedding

**What it is.** A 64-d `nn.Embedding(4, 64)` looked up by magnification and
**concatenated** onto the 1024-d fused feature (→ 1088-d) before the head.

**Intent.** Conditioning. A 40× field shows architecture (gland shapes, stroma); a
400× field shows cytology (nuclear size, chromatin). Telling the model the regime
should let it apply regime-appropriate reasoning instead of averaging four visual
grammars.

**Result.** Image accuracy **0.8974** (−0.008), patient accuracy 0.9625 (−0.025),
AUC 0.9550 (flat), sensitivity 0.905 (−0.019), specificity 0.876 (+0.020).

**Finding.** It improved *specificity* and clean benign low-power fields
(F-23060CD +0.053) and cost *sensitivity* on the hard malignants (DC-12312 −0.048,
PC-9146 −0.049). AUC did not move — it **shifted the operating point** rather than
adding discrimination. Fold 4 test collapsed to 0.837.

**exp2 is not a failed model — it is the control that named the cause.** Without it,
exp1-vs-exp3 confounds magnification with SupCon, and the entire attribution in §6.3
and §6.5 would have been impossible. A negative result that correctly localises a
cause is load-bearing.

### exp3 — + projection head + SupCon

**What it is.** exp2 plus a projection head (`1088 → 512 → 128`, L2-normalised) and
a Supervised Contrastive loss term at weight 0.4, temperature 0.07.

**Result.** **Best AUC of the original three: 0.9607.** Best mean image *validation*
accuracy (0.8739). Lowest thresholded test accuracy: 0.8858 image / 0.9375 patient.
Sensitivity 0.889 (lowest), specificity 0.876.

**Finding.** SupCon genuinely produces a better-*ranked* embedding and tightens the
benign cluster (A-22549CD +0.020, TA-16184 +0.024 vs exp1). But the AUC gain lives in
the **easy mass**, while the images that decide accuracy are the **rare/atypical
malignant tail** (papillary, low-grade ductal) that class-only SupCon strands near the
benign boundary. Combined with the magnification block's benign-leaning bias at a fixed
0.5 cut, that reads as an accuracy loss.

**exp3 is where SupCon's machinery was proven to work at all** — and under the harder
condition of the magnification block being present. exp3n depends on this finding; it
does not retire it.

### exp3n — SupCon **without** the magnification block

**What it is.** exp3 minus the magnification embedding. The ladder's missing cell:
SupCon alone. Added *after* the magnification audit (§6.5), as a set of
**pre-registered falsifiable predictions**.

**Result.** Image accuracy **0.8986** (+0.0128 over exp3), patient accuracy **0.9750**
(+0.0375), sensitivity **0.9193** (+0.030), specificity 0.8445 (−0.032), AUC 0.9593
(**−0.0014 — a wash**), patient AUC **1.0000**.

**Finding — and the reason the AUC wash matters more than the accuracy gain.**
Removing the block did **not** change how well the model *ranks* images. It changed
*where the 0.5 cut falls*. The block was a per-zoom logit bias pushing predictions
benign-ward; delete it and sensitivity recovers on exactly the borderline malignants
(DC-12312 0.577 → **0.712**, PC-9146 0.639 → **0.726**, now 5/5 patient-correct). The
specificity debit is the flip side. On this prevalence the trade nets positive.

**It is also the best-calibrated model of the four** — accuracy-optimal threshold
0.540, calibration gain −0.0005 — so it needs no fitted threshold at all, where exp3
needed 0.36 to reach a *lower* accuracy.

**And it is the retrieval base encoder (D9).** Its SupCon projection space is no
longer magnification-locked (0.976 → 0.335), its embedding geometry is the cleanest
measured (binary silhouette 0.680, above exp1's 0.623; projection silhouette 0.701,
best of any space), and on it the retrieval key, the SupCon space and the classifier's
input are **one and the same 1024-d vector** — one space, one code path.

### exp5 — exp3n + the Retrieval Memory

**What it is.** No new encoder. exp3n's frozen checkpoints + one memory bank per fold
+ one gate fitted on pooled OOF validation.

**Result on the held-out test set (§6.9).** Ensemble ΔAUC **−0.0003**
[−0.0010, +0.0008]; Δaccuracy at the pre-registered threshold **−0.0108**, negative in
5/5 folds; patient level saturated at 16/16 for every stream. The gate keeps a pooled
**0.799** on the parametric head.

**Finding.** The module is correctly built, honestly reports that it is not helping,
and the reason is measured: the memory reads the same frozen encoder as the classifier,
so it is a redundant witness. What it *does* deliver is **evidence**: +0.156 subtype
lift over base rate on unseen patients, with named archived slides a pathologist can
inspect.

### 5.1 expfm — the frozen CTransPath baseline (outside the ladder)

**What it is.** Not a Swin ladder variant. **CTransPath** — a Swin-Tiny with a CNN patch
stem, pretrained by semantically-relevant contrastive learning on ~15 M TCGA/PAIP
histology patches — frozen, its 768-d pooled features cached once for all 7,909
images, with a **1,538-parameter linear head** fitted per fold. Built to answer the
one question the paper cannot leave open (the paper): *why train a Swin
from ImageNet weights instead of probing a pathology foundation model?*

At **27,520,038 parameters** it is param-matched to the Swin ladder's 30,843,388 to within
11%, so the comparison isolates **pretraining corpus** from **model scale**. Same
splits, same eval transform, same class weighting, same monitor, same metric
functions, same seed — every knob pinned to what exp1–exp3n actually did.

**Result.** Image accuracy **0.8580 @0.5** → **0.8985** at its locked pooled-OOF
threshold (0.209), image AUC **0.9668**, sensitivity 0.819, specificity 0.959,
patient **16/16 in 5/5 folds**, patient AUC **1.0000**. Pooled-OOF validation image
accuracy 0.8775 — above the Swin ladder's ~0.865, with no training budget at all.

**Finding.** It **ties** (§1⑥). It also carries the study's largest calibration gain
(+0.0405) and its most extreme optimal threshold (0.209), independently reproducing
finding ①; and it is **wrong on different patients**, which is the error-decorrelation
signal ⑤ named as the binding constraint on retrieval.

**Cost.** ~3 minutes end to end on the 4 GB RTX 3050 (2 min 36 s of it the one-off
encode). Full record: `docs/results/foundation_baseline.md`; commands: `docs/COMMANDS.md` §13.

---

## 6. Chronological research narrative

Every decision, in the order it was taken, with the measurement that settled it.

### 6.1 Protocol first, results second

Splits were generated, asserted disjoint, and frozen before any model was trained.
When the first protocol (class-only stratification) turned out to pin image-level test
accuracy at ~82% regardless of the model, it was replaced **by policy** (§2.1), not by
search, and the old split was kept in the repo. Everything downstream was re-run.

### 6.2 The ladder was trained — and did not improve monotonically

exp1 → exp2 → exp3, 5 folds each, evaluated on the same 16 test patients. Thresholded
accuracy went **down** while AUC held or rose. Documented in `docs/results/classifier_ladder.md`.

The report's own diagnosis of why the "expected" monotonic improvement did not happen:

- **The signal is patient-quantised.** 13–14 val patients, 16 test patients, each with
  60–235 near-duplicate images. Effective sample size is the *patient* count, so every
  cross-experiment delta reduces to **1–2 patients flipping**. The mean-accuracy
  differences (±0.01–0.02) are inside that noise band.
- **`monitor: accuracy` on a 13-patient val set** selects `best.pt` from a noisy signal.
- **Thresholded accuracy is the wrong lens** for components that move the ranking and
  the operating point.
- **The residual errors are structural.** The same 3–5 subtypes fail in *every*
  experiment: **PT (phyllodes)** and **TA (tubular adenoma)** as false positives;
  **PC (papillary)** and **low-grade DC** as false negatives; **MC (mucinous)** on the
  boundary. Rooted in **rarity** (PC) and **morphological overlap**, not in backbone
  capacity.

**Three patients carry essentially all the test error, in every experiment:**

| Patient | Subtype | Label | Why it fails |
|---|---|---|---|
| `SOB_M_DC-14-12312` | ductal | malignant | low-grade / well-differentiated morphology reads benign-like; `mean_prob` sits on the fence |
| `SOB_M_PC-14-9146` | papillary | malignant | **rare subtype** — the parametric head never learns a confident PC boundary |
| `SOB_B_TA-14-16184` | tubular adenoma | benign | tubular/dense architecture mimics a well-differentiated carcinoma |

These five names (plus borderline `MC-16456` and `DC-20636`) became the project's
**pre-registered tracking set**: every later intervention is judged on them, not only
on aggregate accuracy.

### 6.3 Threshold calibration — the hypothesis, tested

**Question.** If the components shifted the operating point rather than the
discrimination, then a threshold calibrated on validation should sit *below* 0.5 and
recover the missing accuracy.

**Protocol.** Threshold picked from **validation predictions only**, locked, then the
untouched test set scored at that value. Two modes: per-fold, and **pooled
out-of-fold** (the 5 val sets are a disjoint partition of all 66 CV patients, so
concatenating gives full OOF coverage — ~8,000 images / 66 patients).

**Result (pooled OOF optimal image thresholds).**

| Model | contains | image opt-thr | Δ accuracy from calibration |
|---|---|---:|---:|
| exp1 | CE | 0.450 | +0.0010 |
| exp2 | CE + **mag** | 0.428 | +0.0042 |
| exp3 | CE + **mag** + SupCon | **0.360** | **+0.0097** |
| exp3n | CE + SupCon | **0.540** | −0.0005 |

The hypothesis is confirmed: the more a component shifted the boundary, the more
calibration recovers. exp3's image sensitivity climbs 0.889 → 0.910 and its *patient*
sensitivity 0.933 → **1.00**, making it the best patient-level model (0.9625) —
reversing its last-place fixed-0.5 standing.

**And the attribution is exact.** The downward drift 0.450 → 0.428 → 0.360 was **not**
"each component"; the two models containing the magnification block are the two with
the lowest thresholds, and exp3n (SupCon *without* the block) jumps to **0.540**,
above 0.5 entirely. SupCon on its own nudges the cut *upward*.

**Two methodological findings that transfer:**

1. **Per-fold calibration overfits and must not be used.** Thresholds picked on ~13 val
   patients swing from **0.05 to 0.91** and *hurt every image-level result*. Only the
   pooled-OOF threshold (66 patients) is stable enough to transfer. This result later
   dictated that the retrieval gate is fitted on pooled OOF, never per fold (D6).
2. **Calibration is not a substitute for fixing the tail.** exp1 still leads image
   accuracy even at every model's own optimal cut, because the deciding errors are
   *mis-ranked*, not mis-thresholded. No global threshold rescues a mis-ranked
   fence-sitter — that is a **memory/exemplar problem**.

### 6.4 Embedding geometry — direct evidence, and a 2×2 that closes

AUC is only *indirect* evidence that SupCon reshaped the representation. The direct
evidence: embed the held-out test set with each model, project to 2-D with UMAP, and
back the pictures with cosine silhouette and a patient-blocked kNN vote
(`docs/results/embedding_geometry.md`).

**What the pictures show.** exp1's `embeddings` are a **single connected manifold** —
one continuous arc with benign and malignant at opposite ends, sharing the middle,
where the fence-sitters live. exp3's have **broken into discrete, mostly single-class
islands**. exp3's `projections` are the sharpest of all.

**The 2×2, once exp3n existed (binary silhouette):**

| | no SupCon | SupCon |
|---|---|---|
| **with mag** | exp2 = 0.538 | exp3 = 0.569 |
| **no mag** | exp1 = 0.623 | **exp3n = 0.680** |

Read *down* each column and SupCon **raises** silhouette in both. Read *across* each
row and the magnification block **drops** it in both. **SupCon is the beneficial
factor; the magnification block is the fragmenting factor; they were fighting inside
exp3, and exp3n lets SupCon win.**

**The magnification-lock table — the number that drove everything after it.** Fraction
of a query's 15 patient-blocked neighbours sharing its magnification (chance ≈ 0.25):

| space | same-mag rate | subtype lift | kNN AUC |
|---|---:|---:|---:|
| exp1 · embeddings (no block) | 0.314 | 0.075 | 0.916 |
| exp3 · embeddings (block **in** the key) | **1.000** | 0.074 | 0.921 |
| exp3 · projections (block upstream) | **0.976** | 0.077 | 0.926 |
| exp3n · embeddings | 0.342 | **0.105** | 0.924 |
| exp3n · projections | **0.335** | **0.106** | **0.928** |

**The limitation this exposed.** Colour the same embeddings by the 8 subtypes and
subtype silhouette is **negative in every space, worst in SupCon's projections
(−0.348)**. Class-only SupCon pulls all malignant subtypes into shared threads; PC, MC
and LC are smeared through the DC mass and never claim their own territory. That is
the exact mechanism behind the residual errors — and the reason a **non-parametric
memory** was the logical next step: the embedding is locally clean enough for
neighbourhood voting (kNN ≈ 0.89) and broken enough on subtypes that the parametric
head cannot isolate the tail.

### 6.5 The magnification audit — probe 4

**Question.** The block is a bad retrieval key. Is it good for *anything*?

**Method.** Load the **frozen classifier weights** out of each `best.pt` and re-run the
decision head under counterfactual magnification inputs. The head is
`Dropout → Linear(1088 → 2)`, so it is exactly reproducible in numpy — no dataset
access, no retraining.

**The structural fact.** For a linear head, concatenating a value that is *constant
per magnification* cannot interact with the image feature:

```
logit_mal − logit_ben = (W_feat[1]−W_feat[0])·f  +  (W_mag[1]−W_mag[0])·E[mag]
                         \___ image-driven ___/     \___ 4 constants ___/
```

The whole 256-parameter embedding collapses at the decision into **four scalars**: one
logit offset per magnification. It is a learned per-zoom *prior*, not conditioning.

**Measured offsets (5-fold mean).** exp3: 40× −0.322, 100× −0.117, 200× −0.044,
400× +0.289 — **monotone in zoom**, i.e. the model learned "be more willing to call
malignant at high power." A sensible pathology prior — and worth a span of ~0.67
logits against an image-driven logit std of **4.22**.

**The counterfactual (exp3, 5-fold mean).**

| head input | image acc | AUC |
|---|---:|---:|
| **true magnification (deployed)** | 0.8858 | **0.96068** |
| block zeroed out entirely | **0.8867** | 0.96050 |
| block = table mean (a pure constant) | 0.8854 | 0.96050 |
| every image forced to 40× | 0.8812 | 0.96050 |
| every image forced to 400× | **0.8904** | 0.96050 |

**Every counterfactual gives 0.96050; the real model gives 0.96068.** The block's total
contribution to ranking is **+0.00018 AUC** (exp2: +0.00032). Deleting it *improves*
accuracy by 0.0009.

**An honest search for where it might still earn its keep — three places, all checked:**

1. **Is it at least new information?** Yes — this is the one result in its favour.
   Magnification is only weakly recoverable from the 1024-d feature (patient-blocked
   kNN recovers it at 0.353–0.393 vs a 0.261 baseline). The problem is not redundancy;
   it is that **a linear head can only use it as a bias**.
2. **Is the per-zoom prior worth anything?** Oracle test — four per-magnification
   accuracy-optimal thresholds fitted *on the test set itself* (an unreachable upper
   bound) beat a single oracle threshold by **≤0.01 accuracy** on every model. And
   exp2's headroom is the *smallest* (+0.0060), because it has already absorbed part of
   the offset into its weights — confirming the mechanism and bounding its value.
3. **Zoom-aware retrieval — the original motivation?** The block delivered it
   implicitly and expensively. Sharding the bank by magnification delivers it
   explicitly and better (D3, §6.6): **+1.1 image points**, which is *exactly* what the
   block-in-key achieved, but with the 1024-d key's better subtype geometry intact.

**Verdict.** *A 4-value per-zoom threshold prior worth +0.0002 AUC, purchased at the
price of a magnification-locked retrieval key and a magnification-locked SupCon space.*
It is a no-op where it could help (the linear classifier) and a real distortion where
it hurts (the projection head SupCon optimises and the vector the memory will index).

**If you ever want real magnification conditioning**, do not concatenate before a
linear head — condition the *features* (FiLM per-magnification scale/shift on the FPN
levels, or a magnification-conditioned gate in `FeatureFusion`) so the signal
multiplies the image evidence instead of adding a constant. That is a separate
experiment (`exp6`), and it should be judged against the ≤0.01 oracle headroom before
anyone invests in it.

### 6.6 Designing the memory — probes 1–3, decisions D1–D8

Four probes on the saved held-out test embeddings, under strict
**leave-one-patient-out** (a query never retrieves, votes with, or fits a metric on any
image from its own patient), averaged over the 5 fold models.

> **Caveat stated up front:** in these probes the bank is *the other 15 test patients*
> (~1,550 images). The deployed bank is a fold's own 52–53 training patients (~4,540
> images) — ~3× larger and far more subtype-diverse. Absolute numbers here are a
> **lower bound**; what transfers is the **relative ordering** — which key space, which
> vote rule, which granularity, which metric.

**D1 — the key is `features`, never `embeddings`, never `projections`.**
The magnification-lock table (§6.4) is the evidence. Stripping the block also *raises*
retrieval AUC (0.9211 → 0.9242) and subtype lift (0.074 → 0.101). Magnification does
not disappear — it is **promoted from a contaminated key dimension to explicit routing
metadata**.

**D2 — build on a SupCon encoder.** The go/no-go question: is the neighbourhood right
where the head is wrong?

| base model | param acc | retrieval acc | **% of head errors rescued** | **oracle-of-two** | prob corr |
|---|---:|---:|---:|---:|---:|
| exp1 | 0.9054 | 0.8886 | 20.8% | 0.9257 | 0.938 |
| exp3 | 0.8858 | 0.8863 | **31.6%** | 0.9266 | **0.906** |
| exp3n | **0.8986** | 0.8801 | 24.2% | **0.9272** | 0.915 |

exp1's manifold is a single arc the head has already read off optimally; exp3's
clustered manifold holds information the linear boundary discards. exp3n's rescue
*percentage* is lower only because its head is stronger — the **oracle ceiling rises to
0.9272, the best measured**, with ~3 points of headroom over the parametric floor.

**D3 — retrieve within magnification.**

| bank restriction (exp3) | img acc | AUC |
|---|---:|---:|
| all magnifications | 0.8863 | **0.9242** |
| **same magnification only** | **0.8974** | 0.9211 |
| cross-magnification only | 0.8824 | 0.9243 |

A 40× low-power field and a 400× high-power field of the same tumour have genuinely
different texture statistics; comparing them dilutes the vote. A pathologist likewise
compares like-for-like magnification before integrating across zooms.

**D4 — softmax vote at T=0.07, hard cap of 3 neighbours per bank patient, no subtype
re-weighting.** Five vote rules compared:

| vote rule | img acc | AUC | spec | DC-12312 ↑ | PC-9146 ↑ | TA-16184 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| *(parametric head)* | 0.8858 | 0.9607 | 0.876 | 0.577 | 0.639 | 0.316 |
| uniform vote | 0.8863 | 0.9204 | 0.805 | 0.73 | 0.59 | 0.49 |
| similarity-softmax (T=0.07) | 0.8863 | **0.9242** | 0.805 | 0.73 | 0.59 | 0.49 |
| **subtype-balanced (1/freq)** | **0.8668** | **0.8962** | 0.784 | 0.71 | **0.50** | 0.53 |
| **patient-capped (≤3/patient)** | **0.8923** | 0.9146 | **0.831** | 0.69 | 0.62 | **0.52** |

- Similarity weighting is nearly free but nearly pointless for accuracy — a temperature
  sweep from 0.01 to ∞ moves accuracy by ≤0.001. It buys AUC, so keep it. **Do not tune it.**
- **Per-patient capping is a real, cheap win** (+0.6 image, +2.6 specificity). BreaKHis
  slides contribute 60–235 near-identical images each, so an uncapped top-15 is
  routinely 15 views of *one* slide. On the deployed bank the largest single patient
  contributes **158** rows, so the cap does *more* work in deployment, not less.

**D5 — two granularities, one store, two rankings.** Image-kNN and slide prototypes fix
**disjoint** cases:

| | DC-12312 (y=1) | PC-9146 (y=1) | TA-16184 (y=0) | spec |
|---|---:|---:|---:|---:|
| parametric head | 0.577 ✗ | 0.639 | 0.316 | 0.876 |
| image-level kNN | **0.73** ✓✓ | 0.59 ✗ | 0.49 ✗ | 0.805 |
| slide prototypes | 0.63 | **0.68** ✓ | **0.43** ✓ | **0.851** |

"Have I seen this individual *field* before?" versus "does this whole *slide* resemble
slides I have seen?" Under patient-blocking, TA-16184's neighbourhood is **48.8%
malignant** and PC-9146's is only **59.0% malignant** — a pure field-level vote on those
two is close to a coin flip.

Probe 4 then tested the *storage* question directly. A single table with a `level`
column, queried once per level, is **bit-identical** to two separate stores — max |Δp|
over every image of every fold = **0.00000000**. So unification is free and removes a
module, a file and a leakage surface.

But a single **index** with one top-k over the union is **not** the same thing:
centroid rows take only **1.2%** of top-k slots (15 centroids vs 1,550 image rows; mean
top-1 cosine 0.987 for centroids vs 0.998 for images; a centroid out-scores the best
image on 0.8% of queries). The merged query silently degenerates to image-only
retrieval and loses exactly the two rescues the slide level exists for. Per-level
z-scoring makes it *worse* (centroid share → 0.000).

> **Unify the store. Never unify the ranking.**

Two details this exposed: per-level `k` and temperature must be independent (centroid
similarities are far more dispersed, std 0.236), and slide rows are **one centroid per
patient**, not per (patient, magnification) — the per-mag variant was tested and is no
better on exp3 and strictly worse on exp1.

**D6 — a learned soft gate, not a hard confidence gate.**

| configuration | gated frac | img acc | pat acc |
|---|---:|---:|---:|
| parametric head alone | — | 0.8858 | 0.9375 |
| param-confidence gate, τ=0.05, α=0.5 | 0.01 | 0.8877 | 0.9375 |
| neighbour-agreement gate, α=0.5 | 0.86 | **0.9003** | **0.9875** |
| **plain blend, α=0.5, no gate** | 1.00 | 0.8997 | **0.9875** |

The hard confidence gate touches 1.2% of images and does **essentially nothing**. What
works is blending nearly everywhere. And the optimal α is a **property of the encoder**
— exp3 wants 0.5, exp1 wants ≈0.8 — so it must be fitted, not hardcoded. Fitted on
**pooled OOF**, per the methodological result from §6.3.

**D7 — plain cosine; no learned metric, no hubness correction.**

| metric | img acc | AUC | neighbour subtype purity |
|---|---:|---:|---:|
| **raw cosine** | **0.8863** | **0.9242** | 0.4026 |
| PCA + whitening | 0.8756 | 0.9021 | 0.4193 |
| LDA (binary-supervised) | 0.8295 | 0.8653 | 0.4260 |
| LDA (subtype-supervised) | 0.8562 | 0.8573 | **0.4456** |

Instructive detail: subtype-supervised LDA does **exactly what it was asked** — best
neighbourhood subtype purity of the four — and binary performance still drops 3 points.
Optimising for subtype coherence and optimising for the binary decision are, in this
regime, **in tension**. Hubness was measured (k-occurrence skew 1.03–1.21, 4% of bank
images never retrieved, top 10% take 26.2% of votes) and is *mild* — enough to justify
the per-patient cap, nowhere near needing CSLS.

**D8 — subtype-awareness is deferred, not folded in.** The subtype geometry that
motivated it is confirmed, but both proposed remedies failed (1/freq voting hurts;
subtype-LDA hurts). A *training-time* subtype-aware SupCon term is a much stronger
intervention and remains plausible — but the evidence is now **suggestive, not
demonstrated**, and there is direct evidence of a subtype-vs-binary tension. It becomes
a separate falsifiable experiment (`exp4`) that the retrieval module must not depend on.
The bank stores subtypes for **diagnostics only**.

### 6.7 exp3n trained — the predictions were registered before the run

| prediction | expected | **actual** | verdict |
|---|---|---|---|
| test AUC ≥ exp3 (no drop > 0.005) | ≥ 0.9607 or within 0.005 | **0.9593** (−0.0014) | ✅ wash |
| retrieval key same-mag neighbour rate | ≈ 0.33, not 1.00 | **0.342** | ✅ |
| `projections` same-mag rate | ≈ 0.33, down from 0.976 | **0.335** | ✅ decisive |
| binary silhouette of `embeddings` | ≥ 0.62, up from 0.569 | **0.680** | ✅ beats exp1 |
| error-rescue rate | ≥ 0.30 | **0.242** | ⚠ below target |

Four of five confirmed. The fifth is read honestly in §6.6 (D2): the rescue *rate* fell
because the head got *better*, while the oracle-of-two ceiling *rose*. Not part of the
predictions, but the headline: **exp3n also beat exp3 as a classifier** (§5).

**D9 settled: the magnification block is retired; exp3n is the retrieval base.**

### 6.8 The module was built — and reported that it does not help

Stage B1 → B2 ran on all five folds. The gate fit raised its own banner:

```
WARNING: The memory is not helping on validation  (p_final 0.8645 vs p_param 0.8654)
```

**First question: is this a bug?** Everything was audited (`docs/retrieval.md` Part VII):
no val/test patient in any bank, query patients blocked, keys unit-norm, D5 verified on
the first batch of every fold, cap/vote/routing matching spec, gate fitted once on
pooled OOF. The gate converging to `w = (param 0.823, img 0.146, slide 0.030)` and
raising the banner is the system **working as designed** — it detected redundant
evidence and refused to lean on it.

**The root cause, as first diagnosed.** On exp3n the magnification block is off, so
`embeddings == features`, and the classifier is a *single linear layer* over that
vector:

```
features (1024-d) ──▶ Linear ──▶ logits            = p_param
        └───────────▶ cosine-kNN vote              = p_img / p_slide
```

Two readouts of the **exact same vector**. A probe of the stored keys found effective
rank **1.19 / 1024**, top singular direction holding 84–96% of variance, random-pair
cosine 0.70, nearest-neighbour cosine 0.999. The alarming `top1_sim ≈ 0.999` in the
logs is **not** a leak — it is the floor of a collapsed cone.

**The generalisation gap proved redundancy rather than a coding fault:** leave-patient-out
kNN AUC on *train* patients is **0.975** (so the geometry genuinely carries class and
nothing is miswired) but on *held-out* patients it is **0.868**, below the linear head's
**0.885** on the same patients. kNN overfits the bank's patients; the linear readout
generalises better.

### 6.9 The key ablation — and the correction it forced

**Question.** The diagnosis blamed the near-rank-1 geometry and listed untested
alternatives: the SupCon `projections`, the spatial `fpn_features`, and an encoder whose
key is not the classifier's input. All were tested.

**What was built.** The key became a first-class, configurable part of the module — a
spec language (`rap_mst/retrieval/keys.py`: forward-dict vectors, `fpn.{gap,max,std,gem}`
poolings, `@level` selection, `.ln` per-level normalisation, `a+b` composition) plus
fitted key transforms (`rap_mst/retrieval/transform.py`: `center`, `pca_drop:n`,
`whiten:n`, `drop_dirs`). Transforms are fitted on the bank's **train rows only** and
stored inside the `.npz`; a bank and a config that disagree **raise** rather than
quietly producing numbers.

**Protocol.** Two stages so 43 configurations cost one forward pass per fold. The
**production classes** (`MemoryBank`, `RetrievalMemory`, `FusionGate`) with the same
seed, epochs and LR. Route, k, cap, temperature and the two-level ranking held fixed —
**the key is the only moving part.** Harness validated: the D1 baseline reproduces
`gate_fit.json` exactly.

**And a control the first analysis lacked.** `p_final` is fitted *and* scored on the
same pooled OOF rows. `p_final_loso` refits the gate on four folds and applies it to the
fifth. The gap between them is the gate's in-sample optimism: **+0.0039 AUC on average,
up to +0.0217.** Every comparison uses LOSO.

**Result — the key does not matter.** Reference: `p_param` image AUC **0.8855**.

- `p_img` AUC across all 39 same-encoder keys spans **0.8584–0.8758**. **0/39 beat the
  head.**
- Under the in-sample protocol 26/39 "beat" `p_param` on fused AUC. Under LOSO only
  8/39, by at most +0.0042 — and **0/39 beat it on image accuracy** (best −0.0009).
- Patient-clustered bootstrap, 2,000 resamples of the 66 CV patients: **every interval
  straddles zero.** The best-ranked row sits inside an interval **eleven times its own
  width** and is negative on accuracy. Treat the ranking as a null result with a spread,
  not a leaderboard.

**Geometry was the wrong suspect — the earlier explanation is falsified.** The
transforms move effective rank across a **338× range** (1.08 → 364.1) while `p_img` AUC
moves within 0.017:

| `features` / transform | eff. rank | random-pair cos | mean top-1 sim | **p_img AUC** |
|---|---:|---:|---:|---:|
| `none` (D1) | 1.19 | 0.700 | **0.9986** | **0.8678** |
| `pca_drop:20` | 33.3 | 0.000 | 0.670 | 0.8593 |
| `whiten:128` | 103.1 | 0.001 | 0.596 | 0.8663 |
| `whiten:512` | **364.1** | 0.000 | **0.386** | 0.8689 |

The collapsed cone is *gone* — and retrieval quality moves by **+0.0011**.
`corr(log effective rank, p_img AUC) = +0.007`. **The near-rank-1 cone is a true
description of the space and a false explanation of the failure.**

**Nor can you subtract the redundancy out.** Between the key space's top principal
direction and the head's decision direction `w = W[1] − W[0]`, |cos| ≈ 0.40–0.63 — they
are *related*, not identical — and **44–72% of ‖w‖² lives outside the top 100 principal
directions**, in the low-variance tail a cosine ranking all but ignores. So the two
readouts are not even the same projection, and they still agree at corr 0.948. Deleting
the head's exact decision direction (`drop_dirs`) changes `p_img` AUC by 0.0021.

> The redundancy is **semantic**: both readouts are functions of the same encoder's
> representation of the same image, and that encoder has already formed its opinion
> about malignancy. Retrieval returns the images this encoder thinks look alike — and
> "looks alike" already carries its class judgement, **including where that judgement is
> wrong**.

**Answers to the three proposed fixes:**

1. *"Key on the spatial pyramid, not the pooled vector"* — **tested, refuted.** Every
   pooling GAP destroys was measured. `fpn.std` (the purest "what GAP throws away"
   signal) gives the best same-encoder retrieval AUC of the study (0.8699) and still
   ranks *below* D1 once fused. Per-level keys span 0.8618–0.8687. Composites add
   nothing.
2. *"Use an encoder whose key is not the classifier's input"* — right instinct, wrong
   reason. Within exp3n, `projections` **is** such a key, and it is the flattest result
   in the study: AUC 0.8660, effective rank **1.08**, the most collapsed space measured.
   SupCon does not spread the space; it tightens class clusters — the same axis again.
3. *"Route/k/T cannot lift the ceiling"* — confirmed; held fixed throughout.

**The one thing that moved: a different encoder.** Build the bank and query keys from
**exp1**'s frozen checkpoints — a CE-only, *weaker* classifier — while `p_param` still
comes from exp3n's head:

| | same encoder (39 configs) | **cross-encoder (exp1 keys, 4 configs)** |
|---|---:|---:|
| **AUC where the head is wrong** | 0.051–0.100 (mean **0.070**) | 0.127–0.136 (mean **0.131**, ×1.88) |
| corr(`p_param`, `p_img`) | 0.885–0.948 | 0.868–0.908 |
| gate weight kept on the memory | 0.027–0.090 | **0.082–0.150** |
| LOSO Δ image accuracy | **negative for all 39** | +0.0013 / +0.0005 |
| LOSO Δ patient accuracy | best +1 patient | **+1 / +2 / +3 / +4 patients — all positive** |

exp1's keys give the two highest genuine retrieval AUCs in the study (0.8723, 0.8718)
*despite being the weaker classifier* — only possible if it ranks similarity on a partly
different notion.

**This is not yet a win** — the bootstrap intervals still straddle zero, and exp1 was
picked because it was already trained, not by a pre-registered rule. It is **a direction
with a measured mechanism**, which is the most this study found.

**D10 recorded: keep `features` / `key_transform: none`.** Not because it won — because
nothing beat it, and it is the simplest, already-validated choice. Adopting `pca_drop:20`
on a +0.0042 AUC that is negative on accuracy and inside an interval eleven times its
width would be tuning to a number. **The key question is closed.**

*The 16-patient test set was not touched by any part of this study.*

### 6.10 Stage C — the production run on the held-out test set

The final loose end: everything retrieval-related above was pooled out-of-fold
validation. B1 → B2 → C was run once, on all five folds, with production config and the
OOF-fitted gate.

**Integrity checks, all passed.** Leakage guard clean in 5/5 folds (independently
re-verified from the dumped neighbour lists: 52–53 distinct bank patients per fold,
**overlap with test patients = 0**). `prob_param` is bit-identical to the plain
non-retrieval test run, so every Δ is a clean **paired** comparison. Routing
`same_magnification_rate = 1.0000` as expected; `mean_neighbours = 15.00`;
`p_slide_std = 0.37–0.46` (non-zero ⇒ the slide ranking is genuinely independent);
`mean_distinct_patients = 7.4–8.0` per query, well above the cap floor.

**Image-level AUC:**

| fold | prob_param | prob_img | prob_slide | prob_final | Δ(final−param) |
|---|---:|---:|---:|---:|---:|
| 0 | **0.9710** | 0.9595 | 0.9187 | 0.9704 | −0.0006 |
| 1 | 0.9194 | **0.9256** | 0.8721 | 0.9149 | −0.0045 |
| 2 | **0.9622** | 0.9476 | 0.9199 | 0.9579 | −0.0043 |
| 3 | **0.9737** | 0.9470 | 0.9230 | 0.9727 | −0.0010 |
| 4 | **0.9701** | 0.9496 | 0.9266 | 0.9646 | −0.0054 |
| **mean** | **0.9593** | 0.9459 | 0.9121 | 0.9561 | **−0.0032** |

`prob_final` loses AUC in **5/5** folds. Patient-level AUC: `prob_param` = **1.0000 in
all five folds**; `prob_final` = 1.0000 in four and 0.9792 in fold 1 — the only movement
the memory produces at the patient level is to *break* a perfect ranking.

**The trap — and why the one apparent gain is not reported.** `prob_final` beats
`prob_param` on accuracy **@0.5** in 5/5 folds (mean +0.0071) while losing AUC in 5/5.
Those can only coexist if the fusion moves the *operating point*, not the ranking. Two
controls confirm it:

| threshold regime | mean Δ accuracy (final − param) |
|---|---:|
| fixed 0.5 | **+0.0071** |
| oracle (test-optimal) threshold | **−0.0027** |
| **pre-registered pooled-OOF threshold** | **−0.0108** (negative 5/5) |

Fold 4 is the whole effect and it is textbook: its head is the *miscalibrated* fold
(accuracy 0.8530 against AUC 0.9701). Mixing in the retrieval vote drags probabilities
toward the middle and lands the fixed cut in a better place — a re-calibration
obtainable with a scalar temperature and no memory at all. Once fold 4 gets its correct
threshold, the memory *costs* it 1.4 points.

**The defensible aggregate** — fold-ensemble with a patient-clustered bootstrap
(2,000 resamples of the 16 test patients), paired against `prob_param`:

| stream | Δ AUC | 95% CI | Δ acc@0.5 | 95% CI |
|---|---:|---|---:|---|
| prob_img | −0.0019 | [−0.0045, +0.0004] | +0.0106 | [−0.0012, +0.0271] |
| prob_slide | **−0.0145** | **[−0.0329, −0.0009]** | +0.0015 | [−0.0158, +0.0167] |
| prob_final | −0.0003 | [−0.0010, +0.0008] | +0.0034 | [−0.0058, +0.0128] |

**The fusion is statistically indistinguishable from the head.** The one interval that
excludes zero is `prob_slide`'s AUC — and it excludes it on the **wrong side**. All four
streams are 16/16 with patient AUC 1.000: **the patient level had no headroom to show a
difference in either direction**, and should not be quoted as a retrieval result.
Decision-level footprint: fusion flips **18 of 1,653** ensemble predictions (1.09%) —
12 become correct, 6 become wrong.

**The mechanism transfers out of sample.** The key ablation named error-decorrelation as
the predictive quantity, on validation. On test:

| quantity | OOF, same encoder | OOF, cross-encoder | **TEST, this run** |
|---|---:|---:|---:|
| AUC of `p_img` on the head's errors | 0.051–0.100 (mean 0.070) | 0.127–0.136 (mean 0.131) | 0.010–0.172 (mean **0.088**); ensemble 0.073 |

The test set lands **inside the same-encoder band and nowhere near the cross-encoder
band**. The D10 conclusion survives contact with held-out data.

**Two things are genuinely new — both on the evidence side, not the accuracy side:**

**① The gate is input-conditioned, and it opens exactly where the head is wrong.** The
`w_param` distribution is strongly bimodal (69% of queries above 0.95, 9% below 0.2,
median 0.991) and it is not noise:

| fold | mean `w_param` where head is **correct** | where head is **wrong** | gap |
|---|---:|---:|---:|
| 0 | 0.899 | 0.564 | −0.335 |
| 1 | 0.712 | 0.417 | −0.295 |
| 2 | 0.705 | 0.533 | −0.172 |
| 3 | 0.936 | 0.765 | −0.171 |
| 4 | 0.871 | 0.576 | −0.295 |

In **every** fold the 147-parameter gate relaxes onto the memory on precisely the images
the head misclassifies, using only five retrieval summary features and never seeing the
label. It is a working, honest **selective-consultation** mechanism — and it buys nothing,
because the witness it consults is wrong in the same places (`awrong` = 0.073). This is
the cleanest demonstration in the project that **the architecture is sound and the
encoder is the constraint.**

**② The subtype lift holds on held-out patients.**

| fold | P(neighbour subtype = query subtype) | base rate | lift |
|---|---:|---:|---:|
| 0–4 mean | **0.396** | 0.240 | **+0.156** |

For 16 patients the module has never seen, the neighbours it returns are the
morphologically related archived slides — at ~65% above chance — with patient ids,
subtypes and similarities dumped to `test_predictions_retrieval.csv` and
`exemplars/*.json`.

**One negative recorded as a design finding.** The slide-centroid level does not earn its
place *on this dataset*: worst AUC of the four streams in 5/5 folds, the only stream
significantly worse than the head, gate weight 0.008–0.089. D5's "unify the store, never
unify the ranking" is architecturally right and was verified again here — but with 52
bank patients ⇒ 52 centroids, the slide ranking is too coarse to contribute. **Report it;
do not delete it** — it is the level a WSI-scale bank would exercise.

**The fold-3 gate-collapse banner.** Fold 3 raised
`WARNING: The gate has collapsed onto the parametric head — mean w_param = 0.922`. That
is **a threshold crossing, not a fold-3 pathology**: pooled `w_param` is 0.799 and 69.8%
of individual queries already exceed 0.9. Fold 3 is simply the most redundant fold, and
the diagnostics agree — `corr(mean w_param, neighbour agreement)` across folds =
**+0.946**; fold 3 has the most unanimous neighbourhoods (0.955), the highest head↔memory
correlation (0.993) and the lowest error-decorrelation (0.010). The module is reporting
the truth about that fold's encoder state. **Do not act on the banner** — its own message
says to report *"the module does not help on this encoder"* rather than tuning until it does.

---

## 7. Decision register

| # | Component | Decision | Primary evidence |
|---|---|---|---|
| **P1** | Split protocol | Patient-level; 16-patient test **subtype-stratified** by documented policy; folds class-stratified | class-only split pinned test accuracy at ~82% across every experiment |
| **P2** | Threshold calibration | On **pooled out-of-fold** predictions only, never per fold | per-fold thresholds swing 0.05–0.91 on 13 val patients and hurt every image-level result |
| **D1** | Memory key | 1024-d pre-magnification `features`, L2-normalised | same-mag neighbour rate 1.000 vs 0.251 chance; mag block alone: kNN AUC 0.361, subtype lift −0.221 |
| **D2** | Base model | A CE + SupCon encoder (→ exp3n per D9) | 31.6% error-rescue on exp3, corr 0.906; exp3n oracle-of-two **0.9272**, best measured |
| **D3** | Magnification | Metadata + routing, not a key dimension. Shard the bank; retrieve same-mag | 0.8863 → 0.8974 image accuracy |
| **D4** | Vote rule | softmax T=0.07, cap 3/patient, **no** frequency re-weighting | cap +0.6 img / +2.6 spec; 1/freq −2.0 img / −5.0 pat |
| **D5** | Granularity | image kNN **+** slide prototypes, **one bank, two views** | unified store bit-identical (Δp = 0); merged *index* → centroid share 1.2%, loses the PC/TA rescues |
| **D6** | Fusion | learned soft gate (147 params), α init 0.5, fitted on pooled OOF | α=0.5 → +1.4 img / +5.0 pat on exp3; hard confidence gate ≈ no-op |
| **D7** | Metric | plain cosine; no learned projection; no CSLS/hubness correction | every learned metric worse; k-occurrence skew only 1.03–1.21 |
| **D8** | Subtype-awareness | store labels for **diagnostics only**; defer the loss to `exp4` | 1/freq voting and subtype-LDA both raise subtype purity and lower binary accuracy |
| **D9** | Magnification block | **Dropped.** The module is built on `exp3n`. Final | block = 4 scalar logit offsets worth +0.00018 AUC; projections un-lock 0.976 → 0.335; exp3n beats exp3 as a classifier |
| **D10** | Key is configurable; default stays `features` / `none` | 43 configurations measured; **none** beats it | `p_img` AUC spans 0.858–0.876 vs `p_param` 0.8855 (0/39 clear it); every bootstrap CI straddles zero; `corr(log eff. rank, AUC) = +0.007` over a 338× range. **The binding constraint is encoder sharing** |

---

## 8. What did not work, and why

The project's most useful output is arguably this list. Each entry was a real
hypothesis, tested directly, and refuted.

**① The magnification embedding did not condition anything.** It was a 4-value per-zoom
logit bias worth +0.0002 AUC. *Why:* concatenating a per-class-constant vector before a
**linear** head can only add a constant to the logit. *What it cost:* a
magnification-locked retrieval key and a magnification-locked SupCon space.
*Correct alternative:* magnification as routing metadata (D3), which recovers the same
accuracy with better geometry. *If you want real conditioning:* FiLM on the FPN levels
(`exp6`) — but the oracle headroom for magnification-conditioned decisions is ≤0.01
accuracy, so justify the investment first.

**② SupCon's AUC gain did not become accuracy.** *Why:* two reasons. The fixed 0.5
threshold on a 72%-malignant set, and — more fundamentally — **class-only SupCon pulls
all malignant subtypes into one blob**, so the rare/atypical malignants that decide
accuracy get stranded near the benign boundary. Subtype silhouette is negative in every
space and *worst* in SupCon's projections (−0.348). SupCon improved separation of the
easy mass at the expense of the tails.

**③ Rare-subtype up-weighting in the vote made things worse.** Inverse-frequency
weighting was the **worst rule on every metric** (−2.0 image, −2.8 AUC, −5.0 patient) and
made the rare papillary patient *worse* (0.59 → 0.50) — the exact case it was invented to
fix. *Why:* up-weighting a rare class amplifies its few, noisy, atypical exemplars
instead of adding signal.

**④ The hard confidence gate was a no-op.** *Why:* its premise was that confident
predictions must be protected from the memory. The α/k surface shows the opposite — the
blend is flat and forgiving across α ∈ [0.4, 0.6] and k ∈ [5, 30], and the confident
images are the ones where head and memory already agree.

**⑤ Learned retrieval metrics all lost to raw cosine.** Including the one that succeeded
at its own objective: subtype-supervised LDA raised neighbourhood subtype purity to the
best of four and still dropped binary accuracy 3 points. Post-hoc supervised
re-projection also throws away most of the 1024-d space (LDA yields ≤7 discriminant
directions).

**⑥ A single merged top-k over the union of image and slide rows silently degenerates.**
Centroids take 1.2% of slots and lose the nearest-neighbour race *by construction*
(averaging denoises a slide *away* from any individual field: mean top-1 cosine 0.987 vs
0.998). Per-level z-scoring makes it worse. This is scale-invariant — the ratio does not
improve with a bigger bank.

**⑦ The near-rank-1 key geometry was a true description and a false explanation.**
The first diagnosis of the retrieval failure blamed effective rank 1.19/1024. Whitening
moved effective rank across a **338× range** and retrieval AUC moved by +0.0011. The
correction was written up rather than quietly dropped.

**⑧ No key fixes a retrieval branch that shares the classifier's encoder.** 43
configurations, 0/39 beat the head. Not pooling, not dimensionality, not geometry, not
deleting the head's own decision direction. The redundancy is **semantic** — it was never
confined to a direction, so it cannot be subtracted out.

**⑨ The Retrieval Memory did not improve accuracy on held-out data.** ΔAUC −0.0003,
Δaccuracy −0.0108 at the pre-registered threshold. *Why, in four layers:* (a) the memory
reads the same frozen encoder as the head; (b) kNN generalises *worse* than the linear
head off-distribution (0.868 vs 0.885 on held-out patients, despite 0.975 within the bank);
(c) error-decorrelation on test (0.088) sits in the same-encoder band, not the
cross-encoder band; (d) there was no headroom left — the head is at patient AUC 1.000 and
16/16 on the test ensemble.

**⑩ The slide-centroid level did not earn its place at this scale.** 52 bank patients ⇒
52 centroids is too coarse a ranking. Kept, because it is the level a WSI-scale bank
would exercise, and because it is architecturally correct.

**A methodological failure caught by a control:** fitting the gate on the same pooled OOF
rows it is scored on inflates AUC by **+0.0039 on average, up to +0.0217**, and turns
8/39 "wins" into 26/39. The leave-one-fold-out gate control exists specifically to
prevent that, and every comparison in the key ablation uses it.

**A trap avoided, worth naming:** the retrieval fusion shows a **+0.0071 mean accuracy
gain at a fixed 0.5 threshold**. It reverses at the oracle threshold (−0.0027) and at the
pre-registered threshold (−0.0108). It is a fold-4 calibration artefact. Quoting it would
be exactly the "tuning to a number" that the project's own protocol forbids.

---

## 9. Consolidated results

### 9.1 Held-out test set, 5-fold mean (16 patients / 1,653 images)

| metric | exp1 | exp2 | exp3 | **exp3n** | exp5 (retrieval) | **expfm** (CTransPath) |
|---|---:|---:|---:|---:|---:|---:|
| trainable params | 30.8 M | 30.8 M | 30.8 M | 30.8 M | +147 | **1,538** *(27.5 M frozen)* |
| image accuracy @0.5 | **0.9054** | 0.8974 | 0.8858 | 0.8986 | 0.9057 | 0.8580 |
| image accuracy @ locked thr | **0.9064** | 0.9016 | 0.8955 | 0.8981 | 0.8874 | 0.8985 |
| image AUC | 0.9559 | 0.9550 | 0.9607 | 0.9593 | 0.9561 | **0.9668** |
| sensitivity | 0.9242 | 0.9054 | 0.8894 | 0.9193 | — | 0.8190 |
| specificity | 0.8563 | 0.8764 | 0.8764 | 0.8445 | — | **0.9590** |
| patient accuracy @0.5 | **0.9875** | 0.9625 | 0.9375 | 0.9750 | 0.9750 | 0.9375 |
| patient AUC | 0.9958 | — | 0.9958 | **1.0000** | 1.0000 | **1.0000** |
| accuracy-optimal threshold | 0.450 | 0.428 | 0.360 | **0.540** | 0.692 | **0.209** |

*The exp5 column is the retrieval fusion. Its @0.5 number is the calibration artefact of
§6.10; the locked-threshold row is the honest one.*

*The expfm column is the **frozen CTransPath foundation-model baseline**
(`docs/results/foundation_baseline.md`): a 1,538-parameter linear head on
27.5 M frozen pathology-pretrained weights, same splits, same transform, same
metrics. It is the clean **"it ties"** outcome — best image AUC of anything measured
here, and level with exp3n once given its own threshold. It is also the study's most
miscalibrated model (0.209), so it independently reproduces headline finding ①: the
movement is in **where the cut falls**, not in discrimination. No bootstrap interval
has been computed for this row yet.*

### 9.2 Embedding geometry (5-fold mean, held-out test set)

| space | binary silhouette | subtype silhouette | kNN_blocked | same-mag rate |
|---|---:|---:|---:|---:|
| exp1 · embeddings | 0.623 | −0.210 | 0.889 | 0.314 |
| exp2 · embeddings | 0.538 | −0.132 | 0.888 | — |
| exp3 · embeddings | 0.569 | −0.158 | **0.898** | **1.000** |
| **exp3n · embeddings** | **0.680** | −0.184 | 0.880 | 0.342 |
| exp3 · projections | 0.669 | −0.348 | 0.895 | **0.976** |
| **exp3n · projections** | **0.701** | **−0.274** | 0.884 | **0.335** |

exp3n's raw kNN dip (0.898 → 0.880) is a **routing artefact**, not a geometry
regression — exp3's key had the block acting as an implicit same-magnification filter.
Route exp3n's bank by magnification and it returns (same-mag kNN 0.887; slide-level vote
0.9031, the best pure-retrieval accuracy of any experiment).

### 9.3 The five tracked patients

Mean P(malignant) over 5 folds — higher is better for the malignants, lower for the
benign:

| patient | y | exp1 | exp3 | **exp3n** | **expfm** | exp3 pat✓ | exp3n pat✓ | **expfm pat✓** |
|---|---|---:|---:|---:|---:|---|---|---|
| `SOB_M_DC-14-12312` low-grade ductal | 1 | ≈0.66 | 0.577 | **0.712** | **0.930** | 3/5 | **4/5** | **5/5** ✅ |
| `SOB_M_PC-14-9146` papillary, rare | 1 | ≈0.72 | 0.639 | **0.726** | 0.614 | 3/5 | **5/5** ✅ | 5/5 |
| `SOB_M_DC-14-20636` borderline | 1 | — | 0.916 | **0.986** | 0.648 | 5/5 | 5/5 | 4/5 |
| `SOB_M_MC-14-16456` mucinous | 1 | — | 0.852 | 0.849 | **0.451** | 5/5 | 5/5 | **1/5** ✗ |
| `SOB_B_TA-14-16184` tubular adenoma | 0 | ≈0.33 | 0.316 | 0.370 | **0.039** | 4/5 | 4/5 | **5/5** ✅ |

Nothing on the easy 13-patient list regresses at the patient level under exp3n.

**Read the expfm column as the error-decorrelation result, not as a ranking.** The two
patients that carried almost all the Swin ladder's test error across every experiment —
`DC-12312` (well-differentiated ductal reading benign-like) and `TA-16184` (tubular
architecture mimicking carcinoma) — are **not hard for a pathology-pretrained encoder**.
In exchange, mucinous carcinoma, trivially easy for every Swin variant, becomes
CTransPath's dominant failure. Two encoders at comparable overall accuracy, **wrong in
different places**. At expfm's locked threshold (0.209) all five recover: **80/80**
fold-patient decisions correct, against 75/80 at 0.5. Per-patient and per-subtype detail:
`docs/results/foundation_baseline.md` §5.5–§5.7.

### 9.4 The honesty caveat that applies to every table above

The test set is **16 patients**, so patient accuracy moves in steps of 1/16 = 0.0625;
val folds are 13–14 patients. Most cross-experiment deltas reduce to **one or two
patients flipping**. Image-level pooled numbers (8,265 predictions across folds) are far
more stable. Treat every effect as **real in direction and mechanism, modest in
magnitude** — that is how it is reported throughout this repo, and it is why several
nominal "wins" were declined.

---

## 10. Repository, configuration, commands

### 10.1 Structure

```
RAP-MST-v1/
├── config/config.yaml           # single source of truth — nothing important is hardcoded
├── rap_mst/                     # importable research package
│   ├── constants.py             # class / magnification encodings, protocol sizes
│   ├── experiments.py           # exp1 / exp2 / exp3 / exp3n / exp5 presets
│   ├── data/
│   │   ├── breakhis.py          # filename parsing, patient id, subtype  (the ONLY parser)
│   │   ├── splits.py            # patient-level split generation/loading + leakage checks
│   │   ├── dataset.py           # BreaKHisDataset (returns rich metadata)
│   │   ├── datamodule.py        # fold loaders + setup_bank / bank_dataloader
│   │   ├── transforms.py        # config-driven augmentation
│   │   └── collate.py           # batches tensors, keeps metadata lists
│   ├── models/
│   │   ├── backbone.py          # hierarchical Swin -> multi-scale maps (timm)
│   │   ├── fpn.py               # lateral + top-down pyramid
│   │   ├── fusion.py            # pyramid -> single pooled vector
│   │   ├── magnification.py     # optional magnification embedding
│   │   ├── projection.py        # optional SupCon projection head
│   │   ├── classifier.py        # classification head
│   │   ├── rap_mst_model.py     # modular assembler (returns the forward dict)
│   │   └── builder.py           # config -> model
│   ├── losses/                  # supcon.py, combined.py, builder.py
│   ├── engine/                  # trainer.py, evaluator.py, optim.py
│   ├── retrieval/
│   │   ├── bank.py              # MemoryBank — ONE store, image + slide levels
│   │   ├── memory.py            # RetrievalMemory (two views), RetrievalRunner
│   │   ├── gate.py              # FusionGate (147 params) + fit_gate
│   │   ├── keys.py              # the key spec language (D10)
│   │   ├── transform.py         # fitted key transforms (D10)
│   │   └── builder.py           # config -> retrieval
│   ├── foundation/              # frozen foundation-model baseline (expfm)
│   │   ├── encoders.py          # ConvStem + encoder registry + weight verification
│   │   ├── cache.py             # FeatureCache, split_views, prediction CSVs
│   │   ├── probe.py             # ProbeHead, Standardizer, fit_probe
│   │   └── builder.py           # config -> encoder / head / cache path
│   └── utils/                   # config, seed, experiment dirs, logging, metrics,
│                                # checkpoint, runs, reporting (the narrative layer)
├── scripts/
│   ├── prepare_splits.py        # generate the permanent splits (run once)
│   ├── train.py  test.py        # train one fold / evaluate the held-out test set
│   ├── build_memory_bank.py     # Stage B1
│   ├── train_gate.py            # Stage B2
│   ├── extract_foundation_features.py   # Stage F1 (frozen CTransPath -> one .npz)
│   ├── train_linear_probe.py            # Stage F2 (probe head, per fold)
│   ├── test_linear_probe.py             # Stage F3 (held-out test set)
│   ├── diagnose_folds.py        # per-patient fold diagnosis
│   ├── threshold_calibration.py # §6.3
│   ├── visualize_embeddings.py  # UMAP + separation metrics (§6.4)
│   ├── retrieval_probe{,2,3,4,5}.py   # the design probes (§6.5, §6.6, §6.8)
│   └── retrieval_key_ablation.py      # the 43-config study (§6.9)
├── splits/                      # generated splits JSON (versioned, + the old backup)
├── analysis/                    # embeddings/, retrieval/, retrieval_keys/,
│                                # retrieval_probe/, threshold_calibration/,
│                                # foundation/ (the frozen feature cache)
├── runs/<exp>/<timestamp>_train_fold<k>/   # config, logs, metrics.csv, TB, checkpoints
└── *.md                         # the reports this README consolidates
```

### 10.2 Configuration

`config/config.yaml` is the **only** place hyper-parameters live. Three ways to change
behaviour, in increasing precedence:

1. **Edit `config.yaml`** — permanent defaults.
2. **`--experiment {exp1,exp2,exp3,exp3n,exp5}`** — applies a preset.
3. **`--set key=value`** — override any dotted key, repeatable, YAML-parsed
   (`--set train.epochs=30`, `--set data.batch_size=8`).

The **exact** config used by a run is saved to that run's directory *and* inside every
checkpoint.

**Hardware defaults are tuned for a 4 GB RTX 3050.** OOM remedy:
`--set data.batch_size=8 --set train.grad_accum_steps=4` (keeps the effective batch)
and/or `--set model.fpn.out_channels=128`.

### 10.3 Commands

```powershell
# once — generate the permanent patient-level splits
python scripts/prepare_splits.py --config config/config.yaml

# train a fold (repeat for folds 0..4)
python scripts/train.py --experiment exp3n --fold 0

# evaluate on the held-out test set
python scripts/test.py --experiment exp3n --fold 0

# --- Retrieval Memory: Stage B1 -> B2 -> C ---
python scripts/build_memory_bank.py --experiment exp3n --fold 0   # ...and 1..4
python scripts/train_gate.py        --experiment exp3n            # ONCE, pooled OOF
python scripts/test.py              --experiment exp5 --fold 0 --retrieval

# the ablation ladder — each is a one-line --set
python scripts/test.py --experiment exp5 --fold 0 --retrieval --set retrieval.levels.image.route=all
python scripts/test.py --experiment exp5 --fold 0 --retrieval --set retrieval.levels.image.per_patient_cap=0
python scripts/test.py --experiment exp5 --fold 0 --retrieval --set retrieval.merge_levels=true
python scripts/test.py --experiment exp5 --fold 0 --retrieval --set retrieval.gate.enabled=false

# the key ablation (43 configs, no test-set access)
python scripts/retrieval_key_ablation.py --stage cache
python scripts/retrieval_key_ablation.py --stage eval --cross-encoder exp1

# --- CTransPath foundation-model baseline: Stage F1 -> F2 -> F3 (~3 min total) ---
python scripts/extract_foundation_features.py                 # ONCE, all 7,909 images
python scripts/train_linear_probe.py --experiment expfm       # all 5 folds, ~15 s
python scripts/test_linear_probe.py  --experiment expfm       # all 5 folds, ~5 s

tensorboard --logdir runs
```

`docs/COMMANDS.md` has the copy-paste versions with the per-stage sanity checks, timings and
a troubleshooting table.

### 10.4 Evaluation protocol — pre-registered

Judge the retrieval module on the axes this analysis showed are meaningful:

1. **Primary — sensitivity at matched specificity** vs the exp3n parametric baseline
   (reported in the same file, so the comparison is like-for-like on identical images).
   Accuracy at a fixed 0.5 cut is threshold-dominated.
2. **Named-case tracking** — `pat✓` out of 5 folds for the five tracked patients.
   **A regression on the easy 13 fails the module** regardless of aggregate gains.
3. **The ablation ladder** — each switch's effect was predicted in advance.
4. **Retrieval-quality metrics independent of the classifier** — patient-blocked kNN
   accuracy, neighbour subtype purity, k-occurrence skew.
5. **Qualitative panel** — top-5 retrieved exemplars per hard case, with subtype and
   patient id. This is the interpretability deliverable and goes in the paper regardless
   of the accuracy outcome.

---

## 11. Reproducibility

- Global seed (42) for Python / NumPy / PyTorch, recorded in every saved config.
- Deterministic cuDNN toggle (`deterministic: true`); DataLoader workers and generator
  seeded.
- Splits generated once and version-controlled; the superseded split kept alongside.
- Checkpoints store optimizer / scheduler / scaler / RNG state for exact resume.
- Every run gets its own timestamped directory (config, `train.log`, `metrics.csv`,
  TensorBoard events, `checkpoints/{best,last}.pt`). **Runs are never overwritten**, and
  Stage C writes `test_metrics_retrieval.json` *alongside* the parametric
  `test_metrics.json` rather than replacing it.
- Banks record their encoder, key, key_transform, fold and source checkpoint; the loader
  **refuses a mismatch** rather than quietly producing numbers.
- Research logging (`utils/reporting.py`) writes a structured narrative through the same
  logger as everything else: startup/env, architecture + parameter counts, dataset +
  hard leakage guard, a one-time first-forward shape trace, per-epoch resources
  (grad norm, img/s, GPU alloc/reserved/peak, CPU RAM), SupCon diagnostics, and automatic
  WARNING banners for NaN/Inf, exploding gradients, projection collapse, val collapse,
  invalid LR and config mismatch.

---

## 12. Status & roadmap

**Implemented and run:** the data protocol and splits; the modular Swin + FPN model;
CE + SupCon; the training/testing engine with AMP, checkpointing, early stopping and
resume; the research logging system; **exp1, exp2, exp3, exp3n trained on all 5 folds**
with test metrics, per-patient fold diagnosis, threshold calibration, UMAP + separation
metrics and five retrieval probes; the **Retrieval Memory module v1** (bank, two-level
memory, fusion gate, key spec language, fitted key transforms) with the **full B1 → B2 → C
production run on the held-out test set**; the **43-configuration key ablation** with
a cross-encoder control and patient-clustered bootstrap intervals; and the **frozen
CTransPath foundation-model baseline** (`expfm`, all 5 folds — `docs/results/foundation_baseline.md`,
the paper).

**The retrieval key question is closed** (D10). The magnification question is closed
(D9). The base encoder is settled (exp3n). **The "where is the foundation model?"
objection is closed** (`expfm` ties; the paper).

**Open on the foundation-model row:** patient-clustered bootstrap CIs, the `expfm_mlp`
capacity check (one command), and the CTransPath→exp3n cross-encoder retrieval run
(needs `retrieval.key_encoder`; a separate pre-registered experiment, and the paper is
complete without it).

**Not implemented — future work, in the order the evidence recommends:**

1. **Encoder diversity** — the only lever that measurably moved error-decorrelation,
   and the direction the CTransPath baseline (§5.1) has now strengthened: at matched
   accuracy it is wrong on *different patients* than exp3n, which is a far better
   prior than the exp1 control ever gave.
   *Free:* promote the exp1 cross-encoder control to a production flag
   (`retrieval.key_encoder`, a contained change to `build_memory_bank.py`), pre-register
   it, and run B1 → B2 → C once. exp1's five folds already exist — **and CTransPath's
   7,909×768 feature cache already exists too**, so a CTransPath bank against the exp3n
   head is now the cheapest and best-motivated version of this.
   *Cheap:* a bank fused over several frozen encoders, one `p_img` per encoder, gate over
   4–5 terms — `FusionGate`'s `term_mask` machinery already supports extra terms.
   *Real work:* train a bank encoder **for** complementarity, with a loss term penalising
   agreement with the frozen head's errors. This is the only route that attacks the
   mechanism head-on.
2. **`exp4` — subtype-aware SupCon** (D8). Deferred, explicitly falsifiable, and the
   retrieval module is designed not to depend on it.
3. **`exp6` — magnification conditioning done properly** (FiLM on the FPN levels), judged
   against the ≤0.01 oracle headroom measured in §6.5.
4. **Prototype Learning and the Reasoning Module.** New granularities are a new `level`
   value in the existing bank — no new store, no new leakage surface.
5. **End-to-end training with a live bank** (out of scope for v1 by design).

Future modules consume `features` (the vector) and/or `fpn_features` (the pyramid) from
the forward dict, and register their loss term in `losses/combined.py` and their
per-epoch diagnostics via `trainer.diagnostics.register(...)` — **the trainer, backbone,
FPN and losses do not need to change.** v1 of the retrieval module is the proof: it
required no edits to the model, the trainer or the losses.

---

*Detailed source reports, all consolidated above:* `docs/results/classifier_ladder.md` (cross-experiment fold &
test analysis) · `docs/results/threshold_calibration.md` · `docs/results/embedding_geometry.md` ·
`docs/retrieval.md` (design decisions D1–D10 + architecture) · `docs/retrieval.md`
(magnification audit, unified bank, operator handbook, exp3n verdict) ·
`docs/retrieval.md` Part VII · `docs/results/retrieval_key_ablation.md` · `docs/results/retrieval_heldout.md` (Stage C
on held-out data) · **`docs/results/foundation_baseline.md`** (the frozen CTransPath baseline,
the paper) · the paper· the paper(the write-up
plan) · the paper (engineering brief) · `docs/COMMANDS.md`.
