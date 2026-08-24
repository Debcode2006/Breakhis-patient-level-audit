# Cross-encoder retrieval memory — the pooled-OOF screen

*A CTransPath-built memory bank behind exp3n's unchanged head. Pre-registered as a
test of the paper's own mechanism, not as an accuracy attempt. Stage: **screen
only** — the 16-patient held-out test set was not touched.*

**Recommendation: RED.** Do not proceed to the production held-out experiment.
Demote the paper's prescriptive sentence and report this screen
instead — it is a stronger result than the prescription would have been.

Artefacts: `analysis/retrieval_crossencoder/{crossencoder_screen.json,
probs_primary.npz, screen.log}`. Code: `rap_mst/retrieval/foreign.py`,
`scripts/retrieval_crossencoder_screen.py`.

---

## Contents

1. [What was implemented](#1-what-was-implemented)
2. [What changed](#2-what-changed)
3. [Validation that the implementation is correct](#3-validation-that-the-implementation-is-correct)
4. [Results](#4-results)
5. [Comparison against previous experiments](#5-comparison-against-previous-experiments)
6. [Interpretation of every metric](#6-interpretation-of-every-metric)
7. [Does the evidence support the hypothesis?](#7-does-the-evidence-support-the-hypothesis)
8. [Remaining implementation artefacts](#8-remaining-implementation-artefacts)
9. [Risks before running the held-out experiment](#9-risks-before-running-the-held-out-experiment)
10. [Recommendation](#10-recommendation)

---

## 1. What was implemented

The experiment proposed in the paper Part B: build the
retrieval memory bank and the query keys from **frozen CTransPath** while leaving
the classifier, the two-level bank machinery, the vote rule and the fusion gate
unchanged.

    bank keys + query keys  <-  CTransPath      (encoder A, foreign, 768-d)
    p_param                 <-  exp3n's head    (encoder B, unchanged)
    bank / vote / cap / two-level ranking / gate  <-  the PRODUCTION classes

Two new files, nothing modified:

**`rap_mst/retrieval/foreign.py`** — `ForeignKeyCache`, which presents the Stage-F1
`FeatureCache` (one file, 7,909 × 768, ordered by the global dataset scan) under
the duck-type the ablation harness's cross-encoder path already consumes
(`keys(split, spec)`, `col(split, name)`, `fold`, `device`, `clf_dir`, `meta`).
It carries the row alignment and the assertions, and nothing else. The key spec
language is deliberately restricted to `features`: a foundation encoder emits one
pooled vector, so `fpn.*` and `projections` do not exist and raise a specific
error rather than a `KeyError`. `key_transform` is unaffected — it is fitted by
the bank on its own rows and works in any key space.

**`scripts/retrieval_crossencoder_screen.py`** — the screen. It *imports* the
scoring helpers from `scripts/retrieval_key_ablation.py` rather than copying them,
so `awrong`, the geometry numbers, the route-`all` diagnostics, the Stage-B2 gate
protocol and the leave-one-fold-out control are computed by the **same code** that
produced the 43-key study this must be compared against.

### The pre-registration, as executed

Written into the output JSON before the numbers, and reproduced here verbatim in
substance:

- **H₁.** Error-decorrelation from the parametric head (`awrong` = AUC of `p_img`
  restricted to the images the head misclassifies) is a property of **encoder
  identity**, not of key definition. A bank encoded by a pathology foundation
  model whose error profile is measurably different (`docs/results/foundation_baseline.md`
  §5.7) should produce `awrong` **above the exp1 cross-encoder band
  (0.127–0.136)**.
- **Primary endpoints are mechanism, not accuracy**: `awrong`, the gate weight
  kept on foreign evidence, the subtype lift, the neighbour overlap with the
  same-encoder memory.
- **Secondary endpoint** is accuracy, expected null, with the bar set higher than
  "beats `p_param`": `p_final` must beat **both** `p_param` **and** the two-probe
  ensemble ½(`p_param`^exp3n + `p_probe`^CTransPath). Anything less is an ensemble
  wearing a retrieval costume.
- **Held fixed**: route / k / cap / two-level ranking (D3–D7) and the gate fit
  protocol. The grid is three configurations mirroring exp1's `CROSS_GRID` so the
  two cross-encoder conditions are compared like for like.
- **The test set is not touched.**

---

## 2. What changed

**Nothing in the production path.** Verified by inspection, and by the fact that
the same-encoder baseline reproduces bit-for-bit (§3):

| module | status |
|---|---|
| `rap_mst/retrieval/{bank,memory,gate,keys,transform,builder}.py` | **untouched** |
| `rap_mst/foundation/{cache,probe,builder,encoders}.py` | **untouched** |
| `scripts/{build_memory_bank,train_gate,test,retrieval_key_ablation}.py` | **untouched** |
| `config/config.yaml` | **untouched** — no new config keys |
| `rap_mst/retrieval/__init__.py` | 3 lines: lazy export of the new module |
| `rap_mst/retrieval/foreign.py` | **new** |
| `scripts/retrieval_crossencoder_screen.py` | **new** |

No `retrieval.key_encoder` production flag was added. That was deliberate: the
screen is what decides whether the flag is worth building, and the answer is no
(§10).

---

## 3. Validation that the implementation is correct

### 3.1 The harness reproduces the published bands exactly

The screen recomputes the same-encoder baseline and both exp1 cross-encoder
controls **in the same process, from the same caches**, before adding the new
condition. Every published figure returns identically:

| row | `awrong` | corr | `p_img` AUC | gate w_img | ΔAUC | Δacc | Δpat | s-lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp3n `features` (D1) — **published** | 0.0559 | 0.948 | 0.8678 | 0.147 | −0.0022 | −0.0027 | +0.0152 | +0.133 |
| exp3n `features` (D1) — **this run** | 0.0559 | 0.948 | 0.8678 | 0.147 | −0.0022 | −0.0027 | +0.0152 | +0.133 |
| exp1 `features` — **published** | 0.1333 | 0.908 | 0.8718 | 0.132 | +0.0017 | +0.0005 | +0.0606 | +0.118 |
| exp1 `features` — **this run** | 0.1333 | 0.908 | 0.8718 | 0.132 | +0.0017 | +0.0005 | +0.0606 | +0.118 |
| exp1 `fpn.std` — **published** | 0.1281 | 0.906 | 0.8723 | 0.150 | +0.0028 | −0.0002 | +0.0455 | +0.121 |
| exp1 `fpn.std` — **this run** | 0.1281 | 0.906 | 0.8723 | 0.150 | +0.0028 | −0.0002 | +0.0455 | +0.121 |

`p_param` is 0.8855 image AUC / 0.8654 accuracy / 0.9598 patient AUC over 6,256
pooled OOF rows and 66 patients, matching `docs/results/retrieval_key_ablation.md` §3.

### 3.2 Row alignment — the one place a bug could hide

The exp3n ablation cache stores `patient_id` / `label` / `mag_index` but **not**
`image_path`, so the mandated image_path join has no left-hand side. Rather than
assume the two caches share a row order, the row order is **re-executed** through
the production objects (`scan_dataset` → `BreaKHisDataset` under the fold's patient
list — the same construction `BreaKHisDataModule.setup_bank` / `setup_fold` uses,
with `shuffle=False`, `drop_last=False`), then checked against the cache
element-wise and joined into the foundation cache by `image_path`.

    ROW ALIGNMENT VERIFIED: 31,280 rows (5 folds x {bank, val})
      (patient_id, label, mag_index) sequence match ......... 10/10 splits
      image_path resolved in the CTransPath cache ........... 100% (0 missing)
      cache 'subtype' column vs subtype_from_patient_id ..... agree, all rows

`ForeignKeyCache.assert_aligned_with` **raises** on any mismatch and names the
first offending index. A silent misalignment would produce a beautifully plausible
null; it cannot occur here.

### 3.3 Leakage

Unchanged and unweakened. The foundation cache holds all 82 patients, but a fold's
bank is filled only from that fold's TRAIN rows; `assert_disjoint` runs per fold
per configuration; `block_query_patients` is on. The encoder never saw a BreaKHis
label (TCGA/PAIP pretraining, frozen, no gradient), which is why the single-file
cache is the analogue of the dataset rather than of a bank
(`rap_mst/foundation/cache.py`). Fold 0: 4,541 bank rows / 52 patients,
1,715 val rows / 14 patients, disjoint.

### 3.4 D5 invariant

`RetrievalMemory._verify_views_once` ran on the first batch of every one of the
5 folds × 16 configurations; no violation. One store, two independent rankings.

---

## 4. Results

### 4.1 The pre-registered grid (pooled OOF, 6,256 images / 66 patients)

| configuration | key encoder | `awrong` | corr(param,img) | `p_img` AUC | gate w_img | ΔAUC | Δacc | Δpat-acc | subtype lift |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp3n `features` (same-encoder D1) | exp3n | 0.0559 | 0.948 | 0.8678 | 0.147 | −0.0022 | −0.0027 | +0.0152 | +0.133 |
| exp1 `features` (control) | exp1 | 0.1333 | 0.908 | 0.8718 | 0.132 | +0.0017 | +0.0005 | +0.0606 | +0.118 |
| exp1 `fpn.std` (control) | exp1 | 0.1281 | 0.906 | 0.8723 | 0.150 | +0.0028 | −0.0002 | +0.0455 | +0.121 |
| **CTP `features`** (primary) | ctranspath | **0.5676** | 0.728 | **0.9149** | **0.404** | +0.0344 | +0.0030 | +0.0000 | **+0.170** |
| CTP `features` whiten:128 | ctranspath | 0.7272 | 0.681 | 0.9337 | 0.450 | +0.0557 | +0.0182 | +0.0303 | +0.135 |
| CTP `features` pca_drop:10 | ctranspath | 0.7393 | 0.634 | 0.9169 | 0.517 | +0.0464 | +0.0141 | +0.0758 | +0.115 |

Δ columns are `p_final_loso` minus `p_param` — the leave-one-fold-out gate, the
honest ordering.

**Every primary endpoint moved in the predicted direction, by a large margin.**
`awrong` is 0.568 against a pre-registered bar of "above 0.127–0.136": **10× the
same-encoder value and 4.3× the exp1 band**. The gate keeps 0.404 weight on
foreign evidence against 0.147 same-encoder.

### 4.2 The ensemble control — the bar `p_final` had to clear

| stream | image AUC | image acc@0.5 | patient acc | patient AUC |
|---|---:|---:|---:|---:|
| `p_param` (exp3n head, 30.8 M) | 0.8855 | 0.8654 | 0.8636 | 0.9598 |
| `p_img` (CTransPath memory vote) | 0.9149 | 0.8625 | 0.9091 | 0.9609 |
| `p_slide` (CTransPath centroids) | 0.8836 | 0.8449 | 0.9091 | 0.9337 |
| **`p_final`** (gate, LOSO) | **0.9199** | 0.8684 | 0.8636 | 0.9620 |
| `p_probe` (CTransPath linear head, 1,538 params) | 0.9415 | 0.8764 | 0.8788 | 0.9717 |
| **`p_ens` = ½(param + probe)** | **0.9482** | 0.8832 | 0.8939 | 0.9761 |
| `p_ens` fitted (LOSO α) | 0.9472 | 0.8841 | 0.9091 | 0.9739 |

Paired, patient-clustered bootstrap, 2,000 resamples over the 66 CV patients:

| comparison | ΔAUC | 95% CI | ΔAcc | 95% CI | verdict |
|---|---:|---|---:|---|---|
| `p_final` − `p_param` | +0.0330 | [−0.0130, +0.0894] | +0.0034 | [−0.0216, +0.0252] | straddles 0 |
| **`p_final` − `p_ens`** | **−0.0278** | **[−0.0524, −0.0065]** | −0.0144 | [−0.0375, +0.0053] | **excludes 0, wrong side** |
| **`p_final` − `p_ens` fitted** | **−0.0266** | **[−0.0553, −0.0035]** | −0.0150 | [−0.0562, +0.0205] | **excludes 0, wrong side** |
| `p_ens` − `p_param` | +0.0608 | [+0.0152, +0.1201] | +0.0178 | [−0.0010, +0.0374] | **excludes 0, positive** |
| `p_img` − `p_param` | +0.0276 | [−0.0433, +0.1068] | −0.0026 | [−0.0510, +0.0433] | straddles 0 |

**The pre-registered success criterion fails, and it fails significantly.**
`p_final` does not beat `p_param` distinguishably, and it is *significantly worse*
than a two-line probability average of the same two models.

### 4.3 Where the loss is — the read-out, not the encoder

The decorrelation is genuinely there, and the memory captures almost all of it:

| quantity | `p_img` (kNN over CTransPath) | `p_probe` (linear head on CTransPath) |
|---|---:|---:|
| `awrong` (AUC on the head's 842 errors) | 0.5676 | 0.6148 |
| corr with `p_param` | 0.7276 | 0.7331 |
| image AUC | 0.9149 | **0.9415** |

Same frozen features, near-identical decorrelation from the head — and a
0.027 AUC gap in discrimination. `p_img` − `p_probe` = **−0.0259
[−0.0589, +0.0034]**.

**This is gate-independent.** The best achievable *scalar* blend — an upper bound
on what any 3-weight gate can extract — is:

    best  alpha * p_param + (1-alpha) * p_img    : alpha = 0.400, AUC 0.9383
    best  alpha * p_param + (1-alpha) * p_probe  : alpha = 0.325, AUC 0.9499
    plain 1/2 (p_param + p_probe)                :               AUC 0.9482

Even with the gate deleted and the blend weight chosen by oracle, routing
CTransPath's information through a cosine-kNN vote lands **below** the naive
average of the two models' probabilities. The memory is not being let down by the
gate; it is a lossy re-reading of a representation that a 1,538-parameter linear
head reads better.

### 4.4 Neighbour overlap — is it even a different memory?

New diagnostic, not in the mandated list, and the cleanest available test of
"different encoder ⇒ different memory". It is label-free, so it cannot be
confounded by the vote rule or the temperature. Top-15 image-level neighbours for
the same 600 queries per fold, production routing:

| overlap with the exp3n memory | value |
|---|---:|
| top-15 neighbour set overlap | **0.075** |
| top-15 Jaccard | 0.041 |
| top-15 **patient** overlap (the generous version) | 0.338 |
| top-1 neighbour identical | **0.024** |

The two memories agree on 2.4% of nearest neighbours and 7.5% of the top-15. This
is a genuinely different memory, not a perturbation of the same one — which makes
the null in §4.2 informative rather than a failure to change anything.

### 4.5 Gate behaviour

| | same-encoder D1 | exp1 control | **CTransPath** |
|---|---:|---:|---:|
| mean `w_param` where head **correct** | 0.838 | 0.857 | 0.456 |
| mean `w_param` where head **wrong** | 0.721 | 0.698 | 0.394 |
| gap | −0.118 | −0.159 | −0.062 |
| mean `w_img` where head wrong | 0.215 | 0.217 | **0.458** |
| fraction of queries with `w_param` > 0.95 | 0.723 | 0.681 | **0.000** |

The gate is behaving correctly at both ends of the dose–response. Same-encoder it
is closed (72% of queries above 0.95) and opens slightly on errors. On CTransPath
it is *globally* open — it never once assigns >0.95 to the head — and still leans
further toward the memory where the head is wrong. The smaller *gap* is a ceiling
effect: there is little room left to open. This is the positive control from
`docs/results/retrieval_heldout.md` §7.1, confirmed under a foreign bank.

### 4.6 Neighbourhood quality — two findings worth keeping

| | same-encoder D1 | exp1 | **CTransPath** | chance |
|---|---:|---:|---:|---:|
| subtype match rate | 0.357 | 0.341 | **0.397** | 0.228 |
| **subtype lift** | +0.133 | +0.118 | **+0.170** | — |
| same-mag rate under `route=all` | 0.411 | 0.381 | **0.740** | 0.25 |
| mean top-1 similarity | 0.9986 | 0.9987 | 0.8165 |  |
| std of top-1 similarity | 0.0021 | 0.0026 | 0.0492 |  |
| effective rank of the key space | 1.19 | 1.27 | **22.61** |  |

1. **The subtype lift is the highest measured anywhere in this project** — +0.170,
   against +0.107…+0.151 across all same-encoder keys and +0.156 on the held-out
   test set. A pathology-pretrained key returns morphologically better neighbours,
   independently of whether the label vote helps.
2. **CTransPath is substantially magnification-locked**: 74% of unrouted
   neighbours share the query's zoom against 25% chance, versus 34–41% for every
   Swin encoder. A foundation model trained without any magnification supervision
   encodes zoom far more strongly than a model that was given an explicit
   magnification embedding. That is a directly reportable observation for §4.2.

### 4.7 Temperature — the confound, closed

`T = 0.07` was fixed on exp3n's key space, where the top-1-to-top-15 similarity
spread is **0.0008**, i.e. `softmax(sim/T)` is effectively a uniform 15-NN
majority. CTransPath's spread is **0.071**, so the same number is a ~2.8× sharper
rule. Measured rather than assumed:

| T | `p_img` AUC | `awrong` | gate w_img | ΔAUC vs param |
|---:|---:|---:|---:|---:|
| 0.01 | 0.9055 | 0.6086 | 0.323 | +0.0459 |
| 0.02 | 0.9115 | 0.6009 | 0.413 | +0.0426 |
| 0.035 | 0.9151 | 0.5872 | 0.439 | +0.0380 |
| **0.07** | **0.9149** | **0.5676** | **0.404** | **+0.0344** |
| 0.15 | 0.9125 | 0.5528 | 0.396 | +0.0330 |
| 0.30 | 0.9112 | 0.5479 | 0.389 | +0.0324 |
| 1.00 | 0.9110 | 0.5475 | 0.378 | +0.0318 |

`p_img` AUC spans 0.9055–0.9151 across a 100× range of T. The vote is flat; T is
not the explanation for anything, and the null is not a temperature artefact. The
best value (0.9151 at T=0.035) is still 0.026 below `p_probe`.

### 4.8 Threshold freedom

Reported so the `docs/results/retrieval_heldout.md` §4 trap cannot reappear:

| stream | acc @0.5 | OOF-optimal thr | acc at optimum (in-sample) |
|---|---:|---:|---:|
| `p_param` | 0.8654 | 0.540 | 0.8657 |
| `p_final` | 0.8684 | 0.335 | 0.8726 |
| `p_ens` | 0.8832 | 0.510 | 0.8875 |
| `p_img` | 0.8625 | 0.465 | 0.8651 |

The ordering is identical at 0.5 and at each stream's own optimum, so no
conclusion here depends on threshold freedom. Note `p_final`'s optimum has drifted
to 0.335 — fusion drags probabilities toward the middle, exactly the
recalibration effect §4 of `docs/results/retrieval_heldout.md` warned about.

### 4.9 Mechanical footprint

Fusion changes **371 of 6,256** predictions (5.9%): 195 to correct, 176 to wrong,
**net +19 images**. Compare the same-encoder production run on test: 18 of 1,653
flips (1.09%), net +6. The foreign memory moves five times more decisions and
still converts barely more than half of them.

### 4.10 Per-fold

| fold | n | `p_param` | `p_img` | `p_probe` | `p_final` | `p_ens` |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1,715 | 0.8413 | 0.9337 | 0.9419 | 0.9245 | 0.9272 |
| 1 | 1,123 | 0.9625 | 0.8881 | 0.9559 | 0.9269 | 0.9672 |
| 2 | 1,119 | 0.9258 | 0.9421 | 0.9611 | 0.9658 | 0.9781 |
| 3 | 1,229 | 0.9975 | 0.9861 | 0.9666 | 0.9991 | 0.9981 |
| 4 | 1,070 | 0.8290 | 0.8644 | 0.9235 | 0.8711 | 0.9224 |

`p_final` beats `p_ens` in **1 of 5 folds** — fold 3, by +0.0010, on a fold where
the head is already at 0.9975 and the gate simply stays out of the way. In the
four folds where there is anything to decide, the ensemble wins by 0.003–0.051.
The apparent aggregate "gain" over
`p_param` is concentrated in folds 0 and 4, precisely where exp3n's head is
weakest (0.841 / 0.829) and CTransPath is much stronger. In fold 1, where the head
is strong, the memory actively *hurts* (0.9625 → 0.9269). This is model
substitution tracking encoder quality, not memory contributing evidence.

---

## 5. Comparison against previous experiments

### 5.1 The dose–response curve, completed

| condition | `awrong` | corr | gate w on memory | outcome |
|---|---:|---:|---:|---|
| same encoder, 39 OOF key configs | 0.051–0.100 (mean 0.070) | 0.885–0.948 | 0.027–0.090 | 0/39 beat the head |
| same encoder, held-out test | 0.010–0.172 (mean 0.088) | mean 0.945 | pooled 0.201 | negative 5/5 at locked thr |
| cross-encoder, exp1 keys (4 OOF) | 0.127–0.136 (mean 0.131) | 0.868–0.908 | 0.082–0.150 | +0.0016…+0.0028 AUC, all CIs straddle 0 |
| **cross-encoder, CTransPath (3 OOF)** | **0.568–0.739** | **0.634–0.728** | **0.404–0.517** | **+0.033…+0.056 AUC vs head, but significantly < the plain ensemble** |

The independent variable is now sampled at three well-separated doses and the
diagnostic quantity is monotone across all three: **0.070 → 0.131 → 0.568**. The
correlate `docs/results/retrieval_key_ablation.md` §6 identified is confirmed as a genuine
dose–response, on an encoder never used to derive it.

**And the outcome column does not follow.** That is the finding.

### 5.2 Against `docs/retrieval.md` Part VII §4

That report measured, *within* one encoder, that a kNN read-out costs
discrimination against a fitted head on the same representation: 0.868 vs 0.885 on
held-out patients. The assessment predicted "there is no reason it reverses across
encoders." It does not reverse — it **widens**: 0.9149 vs 0.9415 on CTransPath,
a 0.027 gap against 0.017. The asymmetry is a property of non-parametric read-out,
not of the particular encoder it was first measured on.

### 5.3 Against `docs/results/retrieval_key_ablation.md` §7.4

That report ranked encoder diversity as "tier 1, free — promote the cross-encoder
control." This screen is the better version of tier 1 (selection by measurement,
not by convenience) and it **retires the tier**. The reason exp1 looked promising
is now visible: at `awrong` ≈ 0.13 the memory is still mostly redundant, so the
gate keeps 87% on the head and the fused output is a small perturbation whose CI
straddles zero. Push the dose 4× higher and the mechanism reveals what it actually
does — it substitutes a second classifier, badly.

### 5.4 Against `docs/results/foundation_baseline.md` §5.7

That section's decision-level observation (CTransPath solves DC-14-12312 and
TA-14-16184, breaks on MC-14-16456) is confirmed distributionally:
corr(`p_param`, `p_probe`) = 0.733 and `awrong` of the probe = 0.615 on OOF. The
two encoders really are wrong in different places. §4.5's claim that this is "the
single best link between two otherwise separate subsections" is upheld — but the
link now runs to a *negative* conclusion about retrieval, not a positive one.

---

## 6. Interpretation of every metric

| metric | value | what it means |
|---|---|---|
| `awrong` 0.5676 | 10× same-encoder | The memory now carries real information about the head's errors. H₁'s primary prediction is confirmed. |
| corr(param, img) 0.728 | vs 0.948 same-enc | The two readouts are genuinely less redundant. |
| gate w_img 0.404 | vs 0.147 same-enc | The gate is not stubborn; given useful foreign evidence it takes it. |
| `frac w_param > 0.95` = 0.000 | vs 0.723 | The gate never fully trusts the head any more. Maximum possible openness. |
| neighbour overlap 0.075 / top-1 0.024 | — | A genuinely different memory, not a perturbation. Rules out "nothing changed". |
| `p_img` AUC 0.9149 | > `p_param` 0.8855 | The memory vote alone out-discriminates the classifier it is supposed to assist. |
| `p_probe` AUC 0.9415 | > `p_img` | …but the *same features* read by a linear head do better. The read-out costs 0.027. |
| `p_ens` AUC 0.9482 | > `p_final` 0.9199 | A two-line probability average beats the whole memory apparatus. |
| ΔAUC final−ens −0.0278 [−0.0524, −0.0065] | excludes 0 | Not noise. The memory is significantly the worse way to combine these two models. |
| ΔAUC ens−param +0.0608 [+0.0152, +0.1201] | excludes 0 | The *only* significant positive in the whole screen belongs to plain ensembling. |
| oracle blend 0.9383 < `p_ens` 0.9482 | — | Gate-independent. Not a fitting artefact. |
| subtype lift +0.170 | best ever measured here | The evidence contribution strengthens, independently of the accuracy null. |
| same-mag rate 0.740 | vs 0.25 chance | CTransPath encodes magnification strongly. New, reportable. |
| effective rank 22.6 | vs 1.19 | A much healthier key geometry — and §4.2 of the key ablation still holds: it does not buy accuracy. |
| flips 371, net +19 | 5.9% of rows | Large footprint, near-coin-flip conversion. |
| T flat 0.9055–0.9151 over 100× | — | The vote rule is not the explanation. |
| `p_final` beats `p_ens` in 0/5 folds | — | Not driven by one fold. |

---

## 7. Does the evidence support the hypothesis?

**Split verdict, and the split is the result.**

**Confirmed — the diagnostic claim.** `awrong` is governed by encoder identity,
not key definition. Predicted in advance to exceed 0.136 on an encoder chosen by an
independent measurement; observed 0.568. Three doses, monotone. The gate weight
tracks it (0.147 → 0.132 → 0.404). `docs/results/retrieval_key_ablation.md` §7.2's decision D10
and the paper's mechanism sentence are **strengthened**: this is now a
law with a third point, not a correlate with two.

**Contradicted — the prescriptive claim.** the paper says: *"do not
build a retrieval bank on the same frozen encoder as the classifier — the
decorrelation quantity predicts in advance whether it can help."* The first clause
is now shown to be *insufficient*: un-sharing the encoder at the largest available
dose raised decorrelation 10×, opened the gate 3×, produced a genuinely different
memory (2.4% top-1 overlap) — and the fused output was still significantly worse
than averaging the two models' probabilities. Decorrelation is **necessary but not
sufficient**.

**The second constraint, newly isolated.** What un-sharing the encoder buys you is
a *second classifier*. A cosine-kNN vote is a lossy way to read one: same features,
`p_img` 0.9149 vs `p_probe` 0.9415, and the oracle scalar blend of (`p_param`,
`p_img`) at 0.9383 sits below the naive average at 0.9482. The honest replacement
sentence:

> Error-decorrelation is necessary for a retrieval branch to contribute, and it is
> a property of encoder identity — but it is not sufficient. Once the encoders are
> decorrelated enough for the memory to matter, the memory is competing against a
> fitted read-out of the same representation, and a non-parametric neighbourhood
> vote loses that competition. On this dataset the configuration that maximises
> decorrelation is one whose gains are fully explained by ensembling.

This is the second time this project has falsified its own explanation and printed
the correction next to the claim it replaces (`docs/results/retrieval_key_ablation.md` §4 did it
to `docs/retrieval.md` Part VII's rank-1 story). It is Branch B1 of
the paper §B.5 — assessed there at ~25% — arriving in a
sharper form than anticipated, because the decorrelation overshot the predicted
band by 4× and the null survived anyway.

---

## 8. Remaining implementation artefacts

Every candidate was tested. The null survives all of them.

**Closed by measurement:**

| suspicion | check | result |
|---|---|---|
| row misalignment | image_path join + element-wise verification | 31,280 rows, 0 mismatches (§3.2) |
| stale gate | gate refit from scratch on the new statistics, LOSO control | done throughout |
| temperature mis-specified for a foreign key space | 7-point sweep, 100× range | `p_img` AUC flat 0.9055–0.9151 (§4.7) |
| gate too weak to exploit the memory | oracle scalar blend, gate deleted | 0.9383 < `p_ens` 0.9482 (§4.3) |
| threshold freedom manufacturing/hiding a gain | acc @0.5 and at each stream's OOF optimum | same ordering (§4.8) |
| one fold driving it | per-fold table | 1/5 folds beat `p_ens`, and only the saturated one (§4.10) |
| routing/cap/k tuned on exp3n's geometry | 6 artefact checks, reported whatever they said | below |

**The routing/cap/k artefact checks** — labelled in the code as checks, *not*
candidate configurations, because adopting a value selected here would be the
open-ended search the paper forbids:

| check | `p_img` AUC | `p_final` AUC | vs `p_param` | vs `p_ens` (95% CI) | beats `p_ens`? |
|---|---:|---:|---:|---|---|
| production (`same_mag`, cap 3, k 15) | 0.9149 | 0.9199 | +0.0344 | −0.0278 [−0.0524, −0.0065] | no |
| `route=all` | 0.9210 | 0.9199 | +0.0344 | −0.0283 [−0.0518, −0.0072] | no |
| `route=cross_mag` | 0.9112 | 0.9209 | +0.0354 | −0.0273 [−0.0518, −0.0063] | no |
| `cap=1` | 0.9150 | 0.9164 | +0.0309 | −0.0319 [−0.0601, −0.0072] | no |
| `cap=15` (no cap) | 0.9120 | 0.9259 | +0.0404 | −0.0223 [−0.0454, −0.0028] | no |
| `k=5` | 0.8975 | 0.9291 | +0.0436 | −0.0191 [−0.0385, −0.0024] | no |
| `k=30` | 0.9173 | 0.9138 | +0.0283 | −0.0344 [−0.0620, −0.0097] | no |

**7/7 lose to the plain ensemble with the 95% CI excluding zero.** Note that
`route=all` is the check that mattered most a priori — CTransPath's unrouted
same-magnification rate is already 0.740, so `same_mag` routing discards much less
there than it does for Swin — and it changes `p_final` by 0.0000.

**Not closed, and stated honestly:**

1. **The screen surface flatters every row equally.** Pooled OOF is the surface the
   gate is fitted on; `p_final_loso` removes the fit-and-score bias but not the
   bias of having looked at this surface at all. This is why the recommendation is
   not "run it on test to be sure" — see §9.
2. **A different fusion form was not tried.** `FusionGate` produces a *convex*
   combination of probabilities. A logit-space or stacked combiner might extract
   more. That is a different module and a different pre-registration, and it would
   still be competing against `p_ens`, which is itself convex and already wins.
3. **One foundation encoder, one dataset.** UNI/Virchow are gated; whether the
   read-out penalty is universal or CTransPath-specific is untested.

---

## 9. Risks before running the held-out experiment

Listed as if the decision were still open, because that is the question asked.

1. **It cannot rescue the claim, and it can damage the paper.** The pre-registered
   criterion has already failed on 6,256 images / 66 patients with an interval
   excluding zero. The test set is 16 patients where `p_param` is already at
   **1.000 patient AUC and 16/16** on the ensemble (`docs/results/retrieval_heldout.md` §5).
   There is no headroom for the primary claim to be rescued, and a large amount of
   noise available to produce a misleading point estimate in either direction.
2. **A held-out "win" would be indefensible.** Any reviewer who computes the
   ensemble control — two existing CSVs, five lines of pandas — finds that the
   proposed system loses to `½(p_param + p_probe)`. Publishing the memory version
   after seeing that would be the "ensemble wearing a retrieval costume" outcome
   §B.4 warns is fatal if unanswered.
3. **It spends a scarce, pre-registered resource.** This project's discipline is
   one test-set touch per pre-registered question. Spending one on a configuration
   that failed its own screen is exactly the practice
   `docs/results/retrieval_key_ablation.md` §7.1 draws the science/tuning line against.
4. **The tempting escape is a trap.** `k=5` gives the best `p_final` (0.9291) and
   the smallest deficit to the ensemble. Promoting it would be selecting a
   configuration on the screening surface and then confirming it on test — the
   exact protocol violation the ablation script's own docstring flags.
5. **Inference cost.** The configuration carries 30.8 M + 27.5 M parameters plus a
   4,541-row bank, against a paper that sells 30.8 M on a 4 GB GPU
   (the paper). It would have to be conceded in the main table for a
   result that is null at best.
6. **Branch A2's cost, for completeness.** If a test run *did* produce a
   significant positive, the paper §B.5 is right that it
   would require rewriting the abstract, the thesis sentence, the ceiling
   argument, the efficiency claim and the novelty positioning — on a 15-page
   budget, for a result the screen says is unlikely and the ensemble control says
   would be attributed to ensembling anyway.

---

## 10. Recommendation

# RED

**Do not proceed to the production held-out experiment.** Demote
the paper's prescriptive sentence and keep the mechanism claim.

The reasoning, compressed:

- The experiment did its job. It was pre-registered around the mechanism, and the
  mechanism answered clearly: `awrong` 0.070 → 0.131 → **0.568**, monotone across
  three encoders, with the gate weight tracking it. That is the dose–response
  curve the paper wanted, and it is now in hand **without touching the test set**.
- The pre-registered success criterion failed, and failed *significantly*:
  `p_final` − `p_ens` = −0.0278 [−0.0524, −0.0065]. Seven operating-point variants,
  a 100× temperature sweep and a gate-free oracle blend all agree, and the only
  fold where fusion wins is the one where the head is already at 0.9975.
- Held-out data cannot change this. The 16-patient test set is saturated for the
  head at the patient level, and at the image level it would add noise to a
  conclusion already established with intervals on 4× more patients.
- **The demotion is now an asset, not a retreat.** Before this screen, the honest
  version of §4.6 was *"we showed a correlate move on a control that straddled
  zero, and we did not run the un-shared configuration."* After it, the paper can
  say: *"we ran the un-shared configuration at the largest available dose of
  encoder diversity; error-decorrelation rose tenfold, the gate opened threefold,
  the retrieved neighbourhoods were 97.6% different — and the fused prediction was
  still significantly worse than averaging the two models' probabilities.
  Decorrelation is necessary and not sufficient; a non-parametric memory loses to
  a fitted read-out of the same representation."* That is a sharper boundary
  condition than the prescription it replaces, and it is defensible.

### What to do with the result

1. **Paper, mechanisms paragraph** — replace the prescription with the
   necessary-but-not-sufficient sentence from §7 above.
2. **Paper §4.4** — one paragraph and one row. The row is the CTransPath bank with its `awrong` and its
   deficit to the ensemble control. Full protocol and diagnostics to supplementary.
3. **Paper §4.2** — add the CTransPath same-magnification rate (0.740 vs 0.25 chance,
   against 0.34–0.41 for every Swin encoder). A foundation model trained without
   magnification supervision encodes zoom more strongly than the model that was
   given an explicit magnification embedding. That is a free, on-topic finding.
4. **§4.4's evidence sentence** — the subtype lift can be upgraded: **+0.170** is
   now the highest measured in the project, and it comes from the pathology-
   pretrained key. The interpretability contribution strengthens while the accuracy
   contribution stays null, which is exactly the framing §4.4 already uses.
5. **Future work** — replace "encoder diversity for the bank (the ×1.88 lever)"
   with the honest successor: the constraint is the *read-out*, so the next
   increment is a learned combiner over multiple frozen encoders, or an encoder
   trained for complementarity — not another bank.
6. **Figure 2** — the paper §C.2 Fix 2 (swappable
   encoder slot) is still worth doing. The slot now carries a measured negative on
   the A≠B setting, which makes the figure carry an argument.

### What was left undone, deliberately

The production `retrieval.key_encoder` flag (`build_memory_bank.py` +
`train_gate.py` + `test.py`, ~4–8 h per the cost table) was **not** built. The
screen's purpose was to decide whether it was worth building. It is not.

---

## 11. Reproducing

```powershell
# Prerequisites (already present): the exp3n/exp1 ablation caches and the
# CTransPath Stage-F1 feature cache + Stage-F2 per-fold val predictions.
python scripts/retrieval_key_ablation.py --stage cache                     # exp3n
python scripts/retrieval_key_ablation.py --stage cache --experiment exp1
python scripts/extract_foundation_features.py                              # Stage F1
python scripts/train_linear_probe.py --experiment expfm                    # Stage F2

# The screen itself -- ~26 min, no training, TEST SET NOT TOUCHED
python scripts/retrieval_crossencoder_screen.py
#   -> analysis/retrieval_crossencoder/crossencoder_screen.json
#      analysis/retrieval_crossencoder/probs_primary.npz
#      analysis/retrieval_crossencoder/screen.log
```
