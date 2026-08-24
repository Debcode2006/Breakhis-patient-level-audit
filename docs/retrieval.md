# Retrieval-Augmented Memory — design, evidence, implementation, handbook

*The single design record for the module the paper describes in §3.5 and Fig. 2, and
reports on in §4.4 and §4.6.*

**Status: v1 — implemented, run on all five folds, and reported as a null result.**
The module is non-parametric, post-fusion and two-stage: the encoder is trained and
frozen first, then used both to write the memory and to answer queries, and the only
trainable component is a 147-parameter fusion gate. Nothing in the training loop, the
backbone, the FPN or the losses changes when it is switched on, so any measured
difference belongs to the module alone.

Code: [`rap_mst/retrieval/`](../rap_mst/retrieval/) ·
[`scripts/build_memory_bank.py`](../scripts/build_memory_bank.py) ·
[`scripts/train_gate.py`](../scripts/train_gate.py) ·
[`scripts/test.py --retrieval`](../scripts/test.py) ·
the `retrieval:` block of [`config/config.yaml`](../config/config.yaml).
Every decision below is a config value, not a hardcoded constant.
Copy-paste commands: [`docs/COMMANDS.md`](COMMANDS.md) §11.

**Where the results are.** This document is *design and method*. The measured
outcomes live in their own reports:

| Report | Question it closes |
|---|---|
| [`results/retrieval_heldout.md`](results/retrieval_heldout.md) | What the module does on the 16-patient held-out test set (paper §4.4). |
| [`results/retrieval_key_ablation.md`](results/retrieval_key_ablation.md) | What the memory should index on — 43 (key, transform) configurations. |
| [`results/crossencoder_screen.md`](results/crossencoder_screen.md) | Whether an *un-shared* encoder fixes it — pooled-OOF screen (paper §4.6). |
| [`results/magnification_audit.md`](results/magnification_audit.md) | Why the base encoder is `exp3n` and not `exp3` (paper §4.2). |

---

## How this document is organised

It merges what were four working documents into one, in the order a reader needs
them: the evidence that fixed each design choice, the decision register that
summarises them, the architecture as shipped, the operating protocol, and finally
the diagnosis of why the finished module does not improve accuracy.

| Part | Contents |
|---|---|
| **I** | Findings 0–8 — the measurements that fixed the key, the routing, the vote rule, the granularity pair and the gate. |
| **II** | Decision register D1–D10, one line each. |
| **III** | Architecture as shipped: placement, files, interfaces, config block, flow, integration points, observability. |
| **IV** | Pre-registered evaluation protocol and known risks. |
| **V** | Why the two granularities live in **one** bank with a `level` column rather than two stores. |
| **VI** | Operator handbook — Stage B1 → B2 → C, step by step, with the ablation ladder. |
| **VII** | Diagnosis: why the module does not beat the parametric head on a shared encoder. |

**A note on numbering.** Section numbers inside each part are inherited from the
source documents and restart per part (Part I–IV use §0–§10, Parts V–VI use their own
Stage/§ numbering). Part headings are the stable anchors; cross-references name the
part.

**Base encoder: `exp3n`** (exp3 minus the magnification embedding) — settled, not a
recommendation. exp3n was trained on all five folds and put through the full suite
([`results/magnification_audit.md`](results/magnification_audit.md) Part 4). Its SupCon
projection space is not magnification-locked (same-magnification neighbour rate
0.976 → **0.335**), its embedding geometry is the best measured (binary silhouette
**0.680** against exp1's 0.623), and it beats exp3 as a standalone classifier. The
bank, the gate and the test run must all come from exp3n; the bank file records its
encoder and the loader refuses a mismatch.

**Reproducing the evidence in Parts I–V:**

```
python scripts/retrieval_probe.py      # key space, hubness, prototypes, per-patient
python scripts/retrieval_probe2.py     # complementarity, mag routing, vote rules, gating
python scripts/retrieval_probe3.py --experiments exp1 exp3   # learned metrics, two-level blend
python scripts/retrieval_probe4.py     # magnification audit; unified vs split bank storage
python scripts/retrieval_probe5.py     # key-space geometry behind Part VII
```

→ `analysis/retrieval_probe/{probe,probe2,probe3,probe4}.json`. These read the saved
held-out test embeddings (`analysis/embeddings/embeddings/<exp>_fold<k>.npz`), not the
dataset, so they are cheap to re-run. Prior context: `docs/results/classifier_ladder.md`,
`docs/results/embedding_geometry.md`, `docs/results/threshold_calibration.md`.

---

# Part I — The evidence

## 0. Evidence protocol and its one caveat

Every number below is computed on the **saved held-out test embeddings**
(`analysis/embeddings/embeddings/<exp>_fold<k>.npz`, 1653 images / 16 patients),
averaged over the 5 fold models, under strict **leave-one-patient-out**: a query
image never retrieves, votes with, or fits a metric on any image from its own
patient. Slide near-duplicates therefore cannot inflate anything.

> **The caveat, stated up front.** In these probes the memory bank is *the other 15
> test patients* (~1550 images). The deployed bank is **fold *k*'s own training
> patients — 52–53 of the 66, ~4,540 images** (measured on fold 0; the other 13–14
> CV patients are that fold's validation set and must never enter its bank) — so
> ~3× larger and far more subtype-diverse. So absolute
> retrieval accuracies here are a **lower bound**, and the subtype-purity numbers
> are pessimistic (a subtype with only one test patient can *never* retrieve its own
> subtype under patient-blocking — those cells are reported as `None`, not as zero).
> What transfers is the **relative** ordering: which key space, which vote rule,
> which granularity, which metric. Those comparisons are what set the architecture.

---

## 1. Finding 1 — the currently-proposed index vector is magnification-locked. **This is the single biggest result.**

`docs/results/classifier_ladder.md` §8 said: *"index the training set's `embeddings`."* That is wrong, and
measurably so. For exp2/exp3 the model builds
`embeddings = concat(fused_features[1024], mag_embedding[64])`
(`models/magnification.py`, `fusion="concat"`). The 64-d block is a **lookup table
value — literally identical for every image at the same magnification**. Under
cosine similarity it acts as a near-hard magnification filter:

| key space (k=15, patient-blocked) | kNN acc | kNN AUC | pat acc | **same-mag neighbours** | (chance) | subtype lift |
|---|---:|---:|---:|---:|---:|---:|
| exp1 · `embeddings` (= 1024-d, no mag) | 0.8886 | 0.9163 | 0.950 | **0.314** | 0.251 | 0.075 |
| exp2 · `embeddings` (1088) | 0.8886 | 0.9144 | 0.963 | **0.996** | 0.251 | 0.062 |
| exp2 · pre-mag slice (1024) | 0.8857 | 0.9173 | 0.975 | 0.332 | 0.251 | 0.069 |
| exp3 · `embeddings` (1088) | 0.8975 | 0.9211 | 0.963 | **1.000** | 0.251 | 0.074 |
| exp3 · pre-mag slice (1024) | 0.8863 | **0.9242** | 0.963 | 0.337 | 0.251 | **0.101** |
| exp3 · `projections` (128) | 0.8953 | 0.9257 | 0.963 | **0.976** | 0.251 | 0.077 |
| **`mag_block` alone (64)** | **0.4731** | **0.3612** | 0.625 | 1.000 | 0.251 | **−0.221** |

**Read.** For exp2/exp3, **99.6–100% of retrieved neighbours share the query's
magnification** against a 25% chance rate. The block carries *no* class signal on
its own — kNN accuracy 0.473 (below chance) and AUC **0.361** (anti-correlated) —
and its subtype lift is **−0.221**, i.e. it actively pushes same-subtype images
apart. exp1, which has no mag block, retrieves at 0.314 ≈ chance: **morphology-
driven, as intended.**

The SupCon `projections` inherit the lock (0.976) because the projection head is
built *on top of* the mag-concatenated vector. So the space `docs/results/embedding_geometry.md`
identified as cleanest by silhouette (0.669) is **not** a safe retrieval index either.

**Decision D1 — the memory key is the 1024-d pre-magnification `features` vector,
never `embeddings` and never `projections`.** This is exactly the `features` entry
already in the forward dict, and it is identical to `embeddings` when magnification
is disabled, so one code path covers all three experiments. Stripping the block also
*raises* retrieval AUC (0.9211 → 0.9242) and subtype lift (0.074 → **0.101**).

Magnification does not disappear — it is **promoted from a contaminated key
dimension to explicit routing metadata** (Finding 3). This is the correct way to
use the exp2 signal, and it is a cleaner story than `docs/results/classifier_ladder.md` §8's "store per
magnification so retrieval is zoom-aware": the block was already forcing that, but
blindly and at the cost of the geometry.

> **Follow-up (probe 4, `docs/results/magnification_audit.md` Part 1).** If the block is that bad
> as a key, is it good for anything? Re-running the **frozen linear classifier**
> under counterfactual magnification inputs answers it: because the head is a
> single `Linear`, the whole 256-parameter embedding collapses into **four scalar
> per-magnification logit offsets** (spread 0.67 vs an image-driven logit std of
> 4.22). Deleting the block entirely changes exp3's test AUC by **0.00018**
> (0.96068 → 0.96050) and its accuracy by −0.0009. Meanwhile it makes exp3's
> SupCon `projections` 97.6% magnification-locked, because the projection head is
> an MLP for which the block is *not* a harmless bias. **Decision D9** below.

---

## 2. Finding 2 — retrieval is genuinely complementary to the head, and *most* so under SupCon. **This is the go/no-go.**

A memory module is only worth building if the neighbourhood is right where the
parametric head is wrong. Measured on the 1024-d key, k=15:

| base model | param acc | retrieval acc | **param wrong → retrieval right** | param right → retrieval wrong | **oracle-of-two** | **% of param errors rescued** | prob corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp1 | 0.9054 | 0.8886 | 0.0203 | 0.0371 | 0.9257 | 20.8% | 0.938 |
| exp2 | 0.8974 | 0.8857 | 0.0260 | 0.0377 | 0.9234 | 18.9% | 0.927 |
| **exp3** | 0.8858 | **0.8863** | **0.0408** | 0.0403 | **0.9266** | **31.6%** | **0.906** |

**Read.** exp3 is the only model where retrieval **matches or beats** its own head
(0.8863 vs 0.8858), where the two are **least redundant** (correlation 0.906, lowest),
and where the memory **rescues ~32% of the head's errors** — 1.5× exp1's rate. The
oracle-of-two ceiling is **0.9266 vs a 0.8858 parametric floor: ~4 accuracy points of
headroom** available to any fusion rule.

This closes the argument `docs/results/embedding_geometry.md` left open. SupCon's value was
previously visible only as AUC (+0.005) and blocked-kNN purity (+0.010) — small,
easy to dismiss. Its *real* payoff is that it makes the embedding **complementary**
to the classifier, which is precisely the property a memory exploits. exp1's
manifold is a single arc that the head has already read off optimally (corr 0.938,
only 21% rescuable); exp3's clustered manifold holds information the head's linear
boundary discards.

**Decision D2 — the retrieval module is built on a CE + SupCon encoder, not exp1**,
despite exp1's better standalone accuracy at the time. exp1 remains the accuracy
baseline to beat.

> **Finalised (this revision).** The SupCon encoder the module is built on is
> **exp3n**, not exp3 — see D9 and `docs/results/magnification_audit.md` Part 4. exp3n's numbers on
> this same table: param acc **0.8986**, retrieval acc 0.8801, rescue rate 24.2%,
> **oracle-of-two 0.9272 (the best of the three)**, correlation 0.915 (still well
> under exp1's 0.938). The rescue *percentage* falls only because exp3n's head is
> stronger and therefore has fewer errors left to rescue; the combined ceiling
> rises and the two sources stay genuinely complementary. The go/no-go this
> section establishes is unchanged, and it is met on exp3n.

---

## 3. Finding 3 — retrieve *within* magnification

With the mag block removed from the key, magnification becomes a routing choice:

| bank restriction (exp3) | img acc | AUC | pat acc |
|---|---:|---:|---:|
| all magnifications | 0.8863 | **0.9242** | 0.9625 |
| **same magnification only** | **0.8974** | 0.9211 | 0.9625 |
| cross-magnification only | 0.8824 | 0.9243 | 0.9500 |

Same-mag routing is worth **+1.1 image points** on exp3 (+0.8 on exp1: 0.8886 →
0.8969) and cross-mag-only is the worst of the three. Intuition: a 40× low-power
field and a 400× high-power field of the *same* tumour have genuinely different
texture statistics; comparing them dilutes the vote. A pathologist likewise
compares like-for-like magnification before integrating across zooms.

Note this recovers exactly the accuracy the mag-block-in-key achieved
(0.8974 ≈ 0.8975) — confirming the block's only useful function *was* implicit
same-mag routing — but does it explicitly, keeping the 1024-d key's better subtype
geometry and leaving multi-magnification evidence fusion available as a future step.

**Decision D3 — shard the bank by magnification; retrieve from the query's own
shard by default (`route="same_mag"`), with `"all"` and `"cross_mag"` as config
options.** Keep AUC-oriented all-mag retrieval available since it ranks marginally
better.

---

## 4. Finding 4 — vote rule: cap per patient, do **not** rebalance by subtype

Five vote rules on the exp3 key (k=15). Hard-patient columns are mean P(malignant);
**for the malignant patients higher is better, for TA-16184 (benign) lower is better.**

| vote rule | img acc | AUC | pat acc | sens | spec | DC-12312 (y=1) | PC-9146 (y=1) | TA-16184 (y=0) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *(parametric head, reference)* | 0.8858 | 0.9607 | 0.9375 | 0.889 | 0.876 | 0.577 | 0.639 | 0.316 |
| uniform vote | 0.8863 | 0.9204 | 0.9625 | 0.917 | 0.805 | 0.73 | 0.59 | 0.49 |
| similarity-softmax (T=0.07) | 0.8863 | **0.9242** | 0.9625 | 0.917 | 0.805 | **0.73** | 0.59 | 0.49 |
| **subtype-balanced (1/freq)** | **0.8668** | **0.8962** | **0.9125** | 0.899 | 0.784 | 0.71 | **0.50** | 0.53 |
| **patient-capped (≤3 / patient)** | **0.8923** | 0.9146 | 0.9625 | 0.916 | **0.831** | 0.69 | 0.62 | **0.52** |
| patient-level (slide centroids, k=5) | 0.8901 | 0.92 ±0.02 ⚠ | 0.9625 | 0.905 | **0.851** | 0.63 | **0.68** | **0.43** |

⚠ The slide-level AUC is not resolvable to better than ±0.02 — 63% of queries get a
unanimous 5-centroid vote, so the score is 0/1 for them and AUC is tie-dominated.
See the correction in §5. Judge this row on **specificity** and the hard-patient
columns, which are stable.

Three conclusions, one of which reverses `docs/results/classifier_ladder.md`:

- **Similarity weighting is nearly free but nearly pointless for accuracy.** A
  temperature sweep from T=0.01 to T=∞ (uniform) moves image accuracy by ≤0.001
  (0.886–0.887 throughout). It *does* buy AUC (0.9204 → 0.9242), so keep it — but do
  not expect the softmax to do work. **T=0.07, and do not tune it.**
- **❌ `docs/results/classifier_ladder.md` §8 point 4 — "rare-subtype up-weighting in the bank" — is refuted.**
  Inverse-frequency weighting is the **worst rule on every metric**: −2.0 image
  points, −2.8 AUC, −5.0 patient points, and it makes the rare papillary patient
  **worse** (PC-9146 0.59 → 0.50), which is the exact case it was invented to fix.
  Mechanism: up-weighting a rare class amplifies its few, noisy exemplars — including
  the mislabelled-looking and atypical ones — instead of adding signal. **Dropped.**
- **✅ Per-patient capping is a real, cheap win.** Allowing at most 3 neighbours from
  any single bank patient gives **+0.6 image points and +2.6 specificity** on exp3
  (and +0.5/+3.4 on exp1). Mechanism: BreaKHis slides contribute 60–235 near-identical
  images each, so an uncapped top-15 is routinely dominated by *one* slide. The cap
  turns "15 views of one patient" into "evidence from ≥5 patients." This is the
  correct, working version of the "don't let the DC mass outvote the query" instinct
  that inverse-frequency weighting failed to deliver.

**Decision D4 — similarity-softmax vote at T=0.07, hard cap of 3 neighbours per bank
patient, no class/subtype frequency re-weighting.**

---

## 5. Finding 5 — two granularities, because they fix *different* patients

Look again at the last two rows of the table above. Image-level kNN and
slide-prototype retrieval have **opposite** strengths on exactly the named misses:

| | DC-12312 (low-grade ductal, y=1) | PC-9146 (papillary, y=1) | TA-16184 (tubular adenoma, y=0) | sens | spec |
|---|---:|---:|---:|---:|---:|
| parametric head | 0.577 ✗ | 0.639 | 0.316 | 0.889 | 0.876 |
| image-level kNN | **0.73** ✓✓ | 0.59 ✗ | 0.49 ✗ | **0.917** | 0.805 |
| slide prototypes | 0.63 | **0.68** ✓ | **0.43** ✓ | 0.905 | **0.851** |

Image-level kNN rescues the **low-grade ductal** case decisively (0.577 → 0.73;
its per-image accuracy goes 0.563 → 0.760) because a handful of *individual fields*
elsewhere in the bank look unmistakably like it. But it **degrades the two
architecture-mimicry cases**: the benign tubular adenoma drifts 0.316 → 0.49
(toward the malignant side) and the papillary carcinoma slips 0.639 → 0.59.

Slide prototypes do the reverse: averaging each bank patient into one centroid
suppresses the individually-ambiguous fields, which is what those two cases need —
TA-16184 improves to 0.43 and PC-9146 to 0.68 — at the cost of blurring DC-12312's
sharp local evidence. It also delivers the **best specificity of any pure retrieval
rule (0.851 vs 0.805)**.

> **⚠ Correction (probe 4).** An earlier version of this section also claimed
> slide prototypes give the **best retrieval AUC (0.9266)**. That claim is
> **withdrawn**. With `slide_k=5` on a 15-patient bank, **63% of queries retrieve
> five centroids that all share one label**, so `p_slide` is exactly 0 or 1 for
> those queries and AUC is dominated by tie-handling: two implementations whose
> probabilities agree to 3e-13 score **0.9159** and **0.9364**. The slide level is
> resolvable to ~±0.02 in AUC and must be judged on **specificity and the named
> cases** instead. `slide_k=5` remains correct — a sweep to k=15 raises AUC only by
> removing tie mass, while specificity falls monotonically 0.851 → 0.833.

This is a textbook complementary pair, and it maps onto a real diagnostic
distinction: *"have I seen this individual field before?"* versus *"does this whole
slide resemble slides I have seen?"* The subtype-purity numbers explain why the
second question is needed — under patient-blocking, **TA-16184's neighbourhood is
48.8% malignant** and **PC-9146's is only 59.0% malignant**, so a pure field-level
vote on those two queries is close to a coin flip.

**Decision D5 — a two-level memory: image-level kNN *and* slide-level prototypes,
combined as separate evidence terms — held in ONE unified bank with a `level`
column, queried as two separate views.**

Probe 4 tested the storage question directly (Part V). A
single store with a `level` column, queried once per level, is **bit-identical** to
two separate stores (max |Δp| over every image of every fold = **0.00000000**), so
unification is free and removes a module, a file and a leakage surface. But a
single *index* with **one** top-k over the union is **not** the same thing and must
be avoided: centroid rows take only **1.2%** of top-k slots (15 centroids vs 1550
image rows; mean top-1 cosine 0.987 for centroids vs 0.998 for images), so the
merged query silently degenerates into image-only retrieval and loses exactly the
two rescues the slide level exists for — PC-9146 0.681 → 0.590, TA-16184 0.425 →
0.488, specificity 0.851 → 0.805. Per-level z-scoring does not rescue it. Slide
rows are **one centroid per patient**, not per (patient, magnification) — the
per-mag variant was tested and is no better on exp3 and worse on exp1.

> **Unify the store. Never unify the ranking.**

---

## 6. Finding 6 — fuse with a soft learned weight, not a hard confidence gate

`docs/results/classifier_ladder.md` §8 proposed gating: consult memory only when `|p − 0.5|` is small.
Tested directly (parametric-confidence gate vs neighbour-agreement gate vs plain
blending), on exp3:

| configuration | gated frac | img acc | pat acc | sens | spec |
|---|---:|---:|---:|---:|---:|
| parametric head alone | — | 0.8858 | 0.9375 | 0.889 | 0.876 |
| param-confidence gate, τ=0.05, α=0.5 (§8's proposal) | 0.01 | 0.8877 | 0.9375 | 0.892 | 0.876 |
| neighbour-agreement gate, thr=0.5, α=0.5 | 0.86 | **0.9003** | **0.9875** | 0.918 | 0.854 |
| **plain blend, α=0.5, no gate** | 1.00 | 0.8997 | **0.9875** | 0.917 | 0.854 |

**The hard confidence gate does essentially nothing** — at τ=0.05 it touches 1.2% of
images, moves image accuracy by +0.002 and leaves patient accuracy unchanged at
0.9375. What actually works is **blending nearly
everywhere at α≈0.5**. The gate's premise was that confident predictions must be
protected from the memory; the α/k surface shows the opposite — the blend is
*flat and forgiving* across α∈[0.4, 0.6] and k∈[5, 30], and the confident images
are ones where head and memory already agree, so blending them is harmless.

The best measured settings, and the optimum's dependence on the base model:

| model | best blend | img acc | pat acc | sens | spec |
|---|---|---:|---:|---:|---:|
| exp3 param-only | — | 0.8858 | 0.9375 | 0.889 | 0.876 |
| exp3 + retrieval | α=0.5, k=15 | **0.8997** | **0.9875** | **0.917** | 0.854 |
| exp1 param-only | — | 0.9054 | 0.9875 | 0.924 | 0.856 |
| exp1 + retrieval | α=0.75–0.8, k=15–20 | 0.9064 | 0.9875 | 0.926 | 0.855 |

exp3 wants an even split (α=0.5); exp1 wants to stay mostly parametric (α≈0.8) —
consistent with Finding 2, where exp1's memory was largely redundant. **α is
therefore a property of the trained encoder, not a universal constant, and must be
fitted rather than hardcoded.**

**Decision D6 — a small learned gate predicting the blend weight from
`[parametric confidence, neighbour agreement, mean top-1 similarity]`, initialised
to α=0.5, fitted on pooled out-of-fold validation predictions. No hard threshold.**
Fitting on pooled OOF (not per fold) follows the methodological result already
established in `docs/results/threshold_calibration.md` §4: a 13-patient fold cannot
estimate an operating parameter without overfitting.

---

## 7. Finding 7 — do **not** add a learned retrieval metric, and do **not** add hubness correction

Two components that modern retrieval systems commonly include were tested and
**rejected on evidence**, which keeps the module small.

**Learned metric.** Four similarities compared under leave-one-patient-out, each
fitted on the bank only (query labels never seen), exp3:

| metric | img acc | AUC | pat acc | sens | spec | neighbour subtype purity |
|---|---:|---:|---:|---:|---:|---:|
| **raw cosine** | **0.8863** | **0.9242** | **0.9625** | 0.917 | 0.805 | 0.4026 |
| PCA + whitening | 0.8756 | 0.9021 | 0.9375 | 0.930 | 0.734 | 0.4193 |
| LDA (binary-supervised) | 0.8295 | 0.8653 | 0.8750 | 0.808 | 0.884 | 0.4260 |
| LDA (subtype-supervised) | 0.8562 | 0.8573 | 0.9250 | 0.902 | 0.736 | **0.4456** |

Raw cosine wins on accuracy, AUC and patient accuracy; every learned re-projection
is worse. Note the instructive detail: **subtype-supervised LDA does exactly what it
was asked to — it raises neighbourhood subtype purity 0.403 → 0.446, the best of the
four — and binary performance still drops** (−3.0 image points, −6.7 AUC). Optimising
the space for subtype coherence and optimising it for the benign/malignant decision
are, in this regime, in tension. Post-hoc supervised re-projection also throws away
most of the 1024-d space (LDA yields ≤7 discriminant directions), which the numbers
show is a bad trade.

**Hubness.** Skewness of the k-occurrence distribution is **1.03 (exp1) – 1.21
(exp3)**, 4.0% of bank images are never retrieved, and the top 10% most-retrieved
images account for **26.2%** of all votes (uniform would be 10%). That is *mild* —
enough to justify the per-patient cap of Finding 4, which addresses the same concern
more directly and more cheaply, but nowhere near the regime where CSLS or
hubness-corrected similarity earns its complexity.

**Decision D7 — cosine similarity on L2-normalised keys, no learned projection
head in the memory, no CSLS / hubness correction.** The per-patient cap is the
only concentration control.

---

## 8. What this means for the subtype problem (an honest downgrade)

`docs/results/classifier_ladder.md` §8 point 3 and `docs/results/embedding_geometry.md` §4 both concluded that
class-only SupCon collapses the rare-malignant tail and that **subtype-aware SupCon**
is the fix. The subtype geometry that motivated this is confirmed — under
patient-blocking, papillary queries sit in a neighbourhood that is only **59.0%
malignant** and tubular-adenoma queries in one that is **48.8% malignant**, which is
precisely why those two patients fail — but the *proposed remedies* did not survive
testing:

- inverse-frequency vote re-weighting **hurts** (Finding 4);
- explicit subtype supervision of the retrieval metric raises subtype purity but
  **lowers binary accuracy** (Finding 7).

Neither result rules out a *training-time* subtype-aware SupCon term — that is a
much stronger intervention than a post-hoc linear re-projection, and it reshapes the
space the classifier itself is trained on. But the evidence for it is now
**suggestive, not demonstrated**, and there is direct evidence of a subtype-vs-binary
tension.

**Decision D8 — subtype-aware SupCon is deferred to a separate, explicitly
falsifiable experiment (`exp4`), not folded into the retrieval module.** The
retrieval module must not depend on it. The bank *does* store subtype labels — but
for **diagnostics and interpretability** (reporting which subtypes a decision was
based on), not as a vote weight. This keeps this document's claims to what the
measurements support.

---

# Part II — Decision register

## 9. Decision summary

| # | Component | Decision | Primary evidence |
|---|---|---|---|
| D1 | Memory key | 1024-d pre-mag `features`, L2-normalised | same-mag rate 1.000 vs 0.251 chance; mag block AUC 0.361, subtype lift −0.221 |
| **D10** | **Key is configurable, and the default stays `features` / `key_transform: none`** | **`retrieval.key` (spec language: forward-dict vectors + `fpn.{gap,max,std,gem}`, per-level, composites) and `retrieval.key_transform` (`center` / `pca_drop:n` / `whiten:n` / `drop_dirs`) exist as a first-class ablation. 43 configurations measured; **none** beats the pooled `features` key.** | `docs/results/retrieval_key_ablation.md`: `p_img` AUC spans only 0.858–0.876 across every key vs `p_param` 0.8855 (0/39 clear it); 0/39 beat `p_param` on LOSO image accuracy; every bootstrap CI straddles zero. Key-space geometry does **not** predict quality (corr(log effective rank, `p_img` AUC) = +0.007 over a 338× rank range). The one quantity that responds is error-decorrelation, and it responds to changing the **encoder** (AUC-where-head-is-wrong 0.070 → 0.131 on exp1 keys), not the key |
| D2 | Base model | **exp3n (CE + SupCon, no magnification)** — a SupCon encoder, per D9 the one without the block | complementarity established on exp3 (31.6% rescue, corr 0.906, oracle 0.9266) and carried by exp3n (24.2% rescue off a stronger head, corr 0.915, **oracle 0.9272 — best measured**) |
| D3 | Mag handling | shard bank by mag, retrieve same-mag | 0.8863 → 0.8974 img acc |
| D4 | Vote | softmax T=0.07, cap 3/patient, **no** freq re-weighting | cap +0.6 img / +2.6 spec; 1/freq −2.0 img / −5.0 pat |
| D5 | Granularity | image kNN **+** slide prototypes, in **one bank, two views** | fix disjoint cases: DC-12312 vs {TA-16184, PC-9146}; unified store bit-identical (Δp = 0), merged *index* loses them (centroid share 1.2%) |
| D6 | Fusion | learned soft gate, α init 0.5, fitted on pooled OOF | α=0.5 → +1.4 img / +5.0 pat on exp3; hard gate ≈ no-op (+0.002 img, +0.0 pat) |
| D7 | Metric / hubness | plain cosine; no learned projection; no CSLS | all learned metrics worse; skew 1.03–1.21 only |
| D8 | Subtype-awareness | store labels for diagnostics; defer the loss to `exp4` | 1/freq voting and subtype-LDA both hurt binary accuracy |
| **D9** | **Base encoder's magnification block** | **dropped — the module is built on `exp3n` (exp3 minus magnification). Trained, verified, FINAL.** | exp3n un-locks the SupCon space (projections same-mag 0.976 → **0.335**), posts the best embedding silhouette (0.680) and subtype lift (0.106) measured, and *beats* exp3 as a classifier (img acc +0.0128, pat acc +0.0375, AUC wash). Full results: this document Parts 1 & 4 |

**Expected effect on exp3 (lower bound, 15-patient bank; the numbers the design
was fitted on):**
image accuracy 0.8858 → **0.8997**, patient accuracy 0.9375 → **0.9875**,
sensitivity 0.889 → **0.917**, specificity 0.876 → 0.854, with per-patient capping
and same-mag routing expected to recover part of that specificity.

**Expected effect on `exp3n`, the encoder actually deployed** (also the 15-patient
probe bank, §4.2): the head starts at 0.8986 image accuracy,
its slide-level vote alone reaches **0.9031** — the best pure-retrieval accuracy of
any experiment — and the head + memory blend at α=0.5 posts **0.9509 AUC**. Because
the deployed bank is the fold's ~52 training patients (~4,500 images) rather than
15 test patients, these remain **lower bounds**; the real numbers come out of
Stage C.

**Read this honestly.** The patient-level jump is +0.05 on a 16-patient test set —
well inside the one-patient quantisation band that `docs/results/classifier_ladder.md` §6 and
`docs/results/threshold_calibration.md` §5 both stress, and it must not be reported as a
headline result. The **defensible** claims are: (a) the mechanism is now measured,
not assumed; (b) the named failure cases move in the right direction for identified
reasons; (c) there is 4 points of oracle headroom that the current head cannot
reach; and (d) the module produces **retrieved exemplar evidence** — a pathologist
can inspect *which* archived slides drove a decision — which is a research
contribution independent of the accuracy delta.

---

# Part III — Architecture, as shipped

## 10. Architecture

### 10.1 Placement

The module is **post-fusion and non-parametric**; it consumes the forward dict and
adds terms, exactly as the paper anticipates. The backbone, FPN, fusion, and
trainer are untouched.

On **exp3n** the magnification embedding is absent (D9), so `embeddings` *is*
`features`: the space SupCon optimises, the space the classifier reads and the
space the memory indexes are the **same 1024-d vector**. That is the architectural
payoff of D9 — one space, one code path — and it is why the D1 slicing gymnastics
below are no longer needed at runtime (the option remains, for the key ablation).

```
image ──▶ SwinBackbone ──▶ FPN ──▶ FeatureFusion ──▶ features [B,1024] ─┬────────────────────────────┐
                                                                        │                            │
                                       [MagnificationEmbedding: OFF]    ▼                            │
                                              embeddings [B,1024] = features ──▶ ClassificationHead ──▶ logits
                                                             │                                       │
                                                    [ProjectionHead] ──▶ projections (SupCon only)   │
                                                                                                     │
   ┌─────────────────────────────────────────────────────────────────────────────────────────────────┘
   │  features (D1: the key — NOT embeddings, NOT projections, on any encoder that has the mag block)
   ▼
┌───────────────────────────── RetrievalMemory ─────────────────────────────┐
│  ONE MemoryBank table (D5)                                                 │
│    columns: level | key[1024] (L2) | label | subtype | patient_id | mag    │
│    level="image"  one row per training image  · sharded by mag (D3)        │
│    level="slide"  one row per training PATIENT (centroid, derived on build,│
│                   mag="ALL" — slide rows are not mag-routed)               │
│                                                                            │
│  TWO VIEWS, ranked separately — never one top-k over the union:            │
│   view(level=image, route=same_mag) ─▶ cosine top-M ─▶ cap ≤3/patient      │
│                                     ─▶ top-k=15 ─▶ softmax T=0.07 ─▶ p_img │
│   view(level=slide, route=all)      ─▶ top-k=5    ─▶ softmax T=0.07 ─▶ p_slide│
└────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────── FusionGate ────────────────────────────────┐
│ features_g = [p_param, |p_param−0.5|, agreement(p_img), mean top-1 sim,     │
│               n_distinct_patients]                                          │
│ (w_param, w_img, w_slide) = softmax(MLP(features_g))       init → (.5,.5,0) │
│ p_final = w_param·p_param + w_img·p_img + w_slide·p_slide            (D6)   │
└────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 New files — **all written; this is the shipped layout**

```
rap_mst/retrieval/
    __init__.py          lazy imports, mirroring rap_mst/data/__init__.py
    bank.py              MemoryBank      — ONE store: build_from_loader / save /
                                           load / view(level) / query  (D5)
                         BankEntry, RetrievalResult, l2_normalize, MAG_ALL
    memory.py            RetrievalMemory — nn.Module, orchestrates the two views
                         LevelConfig     — per-level route/k/cap/temperature
                         RetrievalRunner — memory + gate, what evaluate() calls
    gate.py              FusionGate      — nn.Module, learned blend weights (147
                                           params: 5→16→3), + fit_gate()
    builder.py           build_retrieval(cfg) — config-driven, matches build_model
scripts/
    build_memory_bank.py Stage B1: encode the fold's train split → ONE bank .npz
                         (image rows + derived slide-centroid rows) + sanity table
    train_gate.py        Stage B2: fit FusionGate on pooled OOF val, and lock the
                         pooled-OOF thresholds inside gate.pt
```

Supporting additions (small, additive, nothing rewritten):

| file | addition |
|---|---|
| `rap_mst/data/breakhis.py` | `subtype_from_patient_id()` — the bank's `subtype` column is parsed **here**, never re-implemented (the paper's single-parser rule) |
| `rap_mst/data/datamodule.py` | `setup_bank(fold)` / `bank_dataloader()` — the fold's TRAIN patients under **eval** transforms, no shuffle, no `drop_last`. Using `train_dataloader()` would augment the keys and silently drop the last partial batch |
| `rap_mst/utils/runs.py` | `find_runs` / `resolve_runs_root` / `find_fold_run`, de-duplicated out of the three analysis scripts that each carried a copy |
| `rap_mst/utils/metrics.py` | `best_accuracy_threshold()` moved here verbatim from `scripts/threshold_calibration.py` so the gate's pooled-OOF calibration uses the *same* sweep |
| `rap_mst/utils/config.py` | `merge_section()` (the `retrieval:` block did not exist when the checkpoints were trained) and the shared `parse_set_overrides()` |
| `rap_mst/utils/reporting.py` | `retrieval_diagnostics()`, `check_retrieval_health()` (WARNING banners for the failure modes in this document), `report_retrieval_config()` / `report_retrieval_diagnostics()` |
| `rap_mst/utils/logging_utils.py` | mirrors each run's console + file sinks onto the `rap_mst` package logger, so module-level loggers inside `rap_mst/retrieval/*` land in the **same** run log instead of being silently dropped (see §10.9) |

A separate prototype store is **not** needed — centroids are derived rows of the
same table (D5). See Part VI for the operator's handbook and
`docs/COMMANDS.md` §11 for the copy-paste commands.

### 10.3 Interfaces — **as implemented**

```python
# bank.py
@dataclass(frozen=True)
class BankEntry:                      # one row of the unified value store
    level: str                        # "image" | "slide"  -> which view owns it (D5)
    label: int                        # 0 benign / 1 malignant  -> the vote
    subtype: str                      # one of A,F,TA,PT,DC,LC,MC,PC -> diagnostics (D8)
    patient_id: str                   # per-patient cap + leakage blocking
    mag_index: int                    # 0..3 -> shard routing; MAG_ALL (-1) for slide rows

class MemoryBank:
    """ONE sharded, L2-normalised key store holding every level (D5).
    Pure tensor ops, no learned parameters."""
    def __init__(self, dim: int | None = None, key: str = "features",
                 meta: dict | None = None): ...          # dim=None -> inferred
    def add(self, keys: Tensor, entries: Sequence[BankEntry]) -> None: ...
    def build_from_loader(self, model, loader, device, key=None, amp=False) -> int:
        """Encode a split with the FROZEN encoder, taking out['features'] (D1)."""
    def derive_slide_rows(self) -> int:
        """Group image rows by patient, average, re-normalise, append as level='slide'."""
    def view(self, level: str | None) -> "BankView":
        """A level-restricted, independently-ranked slice. The two views are NEVER
        merged into one top-k — that collapses to image-only retrieval (D5)."""
    def query(self, keys: Tensor, mag_index: Tensor | None = None,
              level: str | None = "image",              # None = merge_levels ABLATION
              k: int = 15, route: str = "same_mag", per_patient_cap: int = 3,
              temperature: float = 0.07,
              block_patients: Sequence[str] | None = None,      # global block list
              query_patient_ids: Sequence[str] | None = None,   # per-query self-block
              candidate_pool_multiplier: int = 8,
              ) -> RetrievalResult: ...
    def assert_disjoint(self, patient_ids, what="query") -> None:   # LEAKAGE GUARD
    def assert_compatible(self, *, key=None, experiment=None, fold=None) -> None:
    def summary(self) -> dict:         # the Stage B1 sanity table
    def save(self, path) -> None: ...  # .npz: levels, keys, labels, subtypes, pids, mags, meta
    @classmethod
    def load(cls, path) -> "MemoryBank": ...

@dataclass
class RetrievalResult:
    prob:       Tensor      # [B]    softmax(sim/T)-weighted P(malignant)  (D4)
    sim:        Tensor      # [B, k] cosine similarities, descending
    idx:        Tensor      # [B, k] bank row indices (-1 where invalid)
    labels:     Tensor      # [B, k]
    weights:    Tensor      # [B, k] vote weights (0 where invalid)
    valid:      Tensor      # [B, k] bool -- masks can leave fewer than k neighbours
    n_distinct_patients: Tensor   # [B] -- gate input + cap diagnostic
    top1_sim:   Tensor      # [B] -- gate input
    subtypes:   list[list[str]]   # diagnostics / interpretability only (D8)
    patient_ids: list[list[str]]
    magnifications: list[list[int]]
    # .agreement property -> 2*|prob - 0.5|
```

```python
# memory.py
@dataclass
class LevelConfig:                    # per-level knobs; separate k/T are mandatory
    enabled: bool = True; route: str = "same_mag"; k: int = 15
    per_patient_cap: int = 3; temperature: float = 0.07
    candidate_pool_multiplier: int = 8

class RetrievalMemory(nn.Module):
    """Non-parametric. Returns retrieval evidence; never mutates the forward dict.
    Queries the ONE bank twice — once per level, with independent k/temperature."""
    def __init__(self, bank: MemoryBank, key: str = "features",
                 image: LevelConfig | None = None, slide: LevelConfig | None = None,
                 merge_levels: bool = False, block_query_patients: bool = True): ...
    def term_mask(self) -> tuple[int, int, int]:   # (param, img, slide) availability
    def forward(self, features: Tensor, mag_index: Tensor | None = None,
                query_patient_ids: Sequence[str] | None = None,
                block_patients: Sequence[str] | None = None) -> dict:
        """-> {p_img, p_slide, agreement, top1_sim, n_distinct_patients,
               image_result, slide_result}

        p_img    softmax(sim/T)-weighted mean of neighbour labels        (D4)
        p_slide  softmax-weighted vote over top-`slide.k` patient centroids (D5)
        agreement  2*|p_img - 0.5| in [0,1] -- neighbourhood consensus, gate input
        """

@dataclass
class RetrievalRunner:                # memory + gate; what evaluate() calls
    memory: RetrievalMemory; gate: FusionGate | None; key: str
    fixed_weights: Sequence[float]    # used when gate is None (ablation)
    thresholds: dict | None           # pooled-OOF cuts locked in Stage B2
    def __call__(self, features, mag_index, p_param, patient_ids) -> dict: ...
```

```python
# gate.py
GATE_INPUTS = ("p_param", "conf_param", "agreement", "top1_sim", "n_distinct_frac")

class FusionGate(nn.Module):
    """5 -> 16 -> 3 MLP + softmax. 147 params: fittable on OOF val without overfitting."""
    def __init__(self, hidden: int = 16, init_weights=(0.5, 0.5, 0.0),
                 init_weight_floor: float = 0.01, term_mask=(1, 1, 1)): ...
    def forward(self, p_param, p_img, p_slide, agreement, top1_sim,
                n_distinct_frac) -> tuple[Tensor, Tensor]:
        """-> (p_final [B], weights [B,3]); the final layer's zero-init weights +
        log(init_weights) bias reproduce `init_weights` exactly before fitting."""
    def save(self, path, extra=None); @classmethod load(cls, path)

def fit_gate(gate, features, probs, labels, *, epochs=400, lr=0.01, ...) -> dict:
    """Full-batch Adam on binary cross-entropy over the POOLED out-of-fold rows."""
```

### 10.4 Config block — **as shipped in `config/config.yaml`**

Every design decision above is one line here. Nothing in the Python hardcodes a
route, a `k`, a cap or a temperature.

```yaml
retrieval:
  enabled: false                 # exp1-exp3n unchanged when false; no behaviour drift
  base_experiment: exp3n         # D9 -- the frozen encoder the module is built on
  key: features                  # D1 -- 'features' | 'embeddings' | 'projections'
  bank_path: "analysis/retrieval/{experiment}/bank_fold{fold}.npz"  # ONE file, all levels (D5)
  gate_path: "analysis/retrieval/{experiment}/gate.pt"              # one gate, shared by folds
  merge_levels: false            # D5 -- MUST stay false; true = one top-k over the
                                 #       union, which degenerates to image-only
  block_query_patients: true     # a query never retrieves its own patient, ever
  levels:
    image:
      enabled: true
      route: same_mag            # D3 -- 'same_mag' | 'all' | 'cross_mag'
      k: 15                      # D6 -- flat over 5..30
      per_patient_cap: 3         # D4
      temperature: 0.07          # D4 -- do not tune
      candidate_pool_multiplier: 8   # cap search pool = k * this
    slide:
      enabled: true              # D5
      route: all                 # slide centroids are per patient, not per mag
      k: 5                       # specificity-optimal; larger k trades it away
      per_patient_cap: 1         # one centroid per patient by construction
      temperature: 0.07
      candidate_pool_multiplier: 8
  gate:
    enabled: true                # D6 -- false = fixed blend at `fixed_weights`
    hidden: 16
    init_weights: [0.5, 0.5, 0.0]
    init_weight_floor: 0.01      # log(0) is a dead unit; floor it (see below)
    fixed_weights: [0.5, 0.5, 0.0]   # used only when gate.enabled=false
    fit_on: pooled_oof           # never per-fold (docs/results/threshold_calibration.md §4)
    epochs: 400
    lr: 0.01
    weight_decay: 0.0
  diagnostics:
    log_retrieved_subtypes: true # D8 -- interpretability, not a vote weight
    dump_exemplars: true
    exemplars_top_k: 5
    exemplars_per_patient: 5
    dump_exemplars_for: [SOB_M_DC-14-12312, SOB_M_PC-14-9146, SOB_B_TA-14-16184,
                         SOB_M_MC-14-16456, SOB_M_DC-14-20636]
```

Five implementation notes where the code had to make a call the design did not
specify — each is a config value, not a silent constant:

- **`{experiment}` in the paths.** Banks are namespaced per encoder because
  "do not mix" (§10.5) is the easiest rule in this module to break by accident.
  Belt *and* braces: the `.npz` records `experiment`, `key`, `fold` and the
  checkpoint it came from, and `MemoryBank.assert_compatible` refuses a mismatch.
- **`init_weight_floor`.** D6 asks for an init of exactly `(0.5, 0.5, 0.0)`, but a
  softmax weight of 0 is `log(0) = −inf` — an unrecoverable dead unit that could
  never learn to use the slide level. The floor starts it at
  `(0.495, 0.495, 0.010)` instead. The final layer's *weights* are zero-initialised,
  so the gate reproduces its init blend exactly for every input until it is fitted.
- **`candidate_pool_multiplier`.** The per-patient cap is a greedy descending scan;
  it needs a candidate pool deeper than `k`. `8` matches `retrieval_probe2.py`.
- **`block_query_patients`.** Defence in depth on top of the build-time leakage
  assertion: a query is masked against its own `patient_id` at ranking time, so a
  mis-built bank degrades neighbours rather than silently inflating the score.
- **Disabling a level masks the gate term** rather than blending a neutral 0.5.
  `levels.slide.enabled=false` yields an honest 2-way blend — and because that
  changes what the gate was fitted on, loading a 3-term gate under a 2-term config
  is a **hard error** with instructions (refit, or `gate.enabled=false`).

### 10.5 Training / inference flow

Deliberately **two-stage** rather than end-to-end. The encoder is frozen when the
bank is built, so the keys stay consistent with the vectors that were probed, the
bank never has to be refreshed mid-training, and the whole module adds **~100
trainable parameters** — trainable on 66 OOF patients without overfitting. End-to-end
training with a bank that drifts every step is out of scope for v1.

```
Stage A  (DONE)       train the base encoder per fold -> checkpoints/best.pt
         exp3n's five folds are trained and scored; D9 is settled in its favour.
         Nothing downstream reads exp1/exp2/exp3 checkpoints.

Stage B1 build bank    scripts/build_memory_bank.py --experiment exp3n --fold k
         freeze best.pt, encode that fold's TRAIN patients, store out['features']
         as level='image' rows, then derive one centroid per patient and append it
         as level='slide' rows -- ONE .npz per fold (D5).
         LEAKAGE RULE: the bank holds only the fold's own training patients; val and
         test patients are never inserted. Same-patient blocking is then automatic,
         and is additionally asserted at query time.

Stage B2 fit gate      scripts/train_gate.py --experiment exp3n
         Recompute val predictions per fold (as threshold_calibration.py already does),
         retrieve each val image against its own fold's bank, POOL all 5 folds' val
         predictions (66 patients, full OOF coverage), fit FusionGate by BCE on the
         pooled set. One gate, shared across folds. Freeze it.

Inference  scripts/test.py --checkpoint <fold best.pt> --retrieval
         p_param from the head; p_img/p_slide from the fold's bank; p_final from the
         frozen gate. Report image- and patient-level metrics, both at 0.5 and at the
         locked pooled-OOF threshold, so results stay comparable to docs/results/classifier_ladder.md.
         Writes test_metrics_retrieval.json + test_predictions_retrieval.csv +
         exemplars/ ALONGSIDE the existing test_metrics.json -- the parametric
         baseline files are never overwritten.
```

**Measured cost on the target hardware** (RTX 3050, verified end-to-end on fold 0):
encoding one fold's 4,541 training images takes **~2 min**, the bank is **17.5 MB**
compressed (`.npz`), the gate fit is seconds, and a Stage C test pass adds
negligible time to the existing 1,653-image evaluation. Exact cosine over a
magnification shard is the right choice at this scale — no FAISS, no ANN index.

**A first-timer's step-by-step version of this flow — with the commands, the
sanity checks at each stage, the failure table and the timings — is
Part VI.**

### 10.6 Integration points in existing code

Small and additive; nothing is rewritten.

| File | Change | Status |
|---|---|---|
| `config/config.yaml` | the `retrieval:` block (defaults `enabled: false`) | **done** |
| `rap_mst/experiments.py` | `exp3n` preset (D9) + `exp5` = exp3n + `retrieval.enabled: true`, and `encoder_experiment()` mapping `exp5 → exp3n` so exp5 reuses exp3n's frozen checkpoints instead of re-training an identical encoder. exp1–exp3n stay byte-identical | **done** |
| `rap_mst/models/rap_mst_model.py` | **no change** — `features` is already in the forward dict | — |
| `rap_mst/engine/evaluator.py` | optional `retrieval=` argument: fills `prob_param` / `prob_img` / `prob_slide` / `prob_final`, the gate weights and the retrieved neighbours, and scores `p_final`. `retrieval=None` is byte-identical to the previous behaviour, so the trainer's validation path is untouched | **done** |
| `rap_mst/engine/trainer.py` | **no change** for v1 (no retrieval loss term) | — |
| `rap_mst/losses/combined.py` | **no change** for v1 | — |
| `rap_mst/utils/reporting.py` | `retrieval_diagnostics(raw)` — mean top-1 similarity, neighbour agreement, distinct-patient count, retrieved-subtype histogram. Returned as a dict so a future training-time integration wraps it in a closure for `trainer.diagnostics.register(name, fn)` with no loop edits | **done** |
| `rap_mst/data/datamodule.py` | `setup_bank(fold)` / `bank_dataloader()` (eval transforms, ordered, no `drop_last`) | **done** |
| `rap_mst/data/breakhis.py` | `subtype_from_patient_id()` | **done** |
| `scripts/test.py` | `--retrieval` / `--bank` / `--gate` / `--experiment` / `--set` | **done** |

The modularity claim in the paper held up: because `features` was already exposed
and losses are already additive, **v1 required no edits to the model, the trainer or
the losses.** The one genuine mismatch found during the review was documentation,
not code — the paper and `docs/results/classifier_ladder.md` §8 both named `embeddings` as "the canonical
vector a future Retrieval/Prototype module should index", which Finding 1 shows is
wrong whenever the magnification embedding is enabled. **Both are now corrected to
`features`.**

### 10.9 Observability — the run tells you what it did

The module logs like the rest of the codebase (through `utils/reporting.py` over the
existing `get_logger` console+file logger), and is **self-describing**: reading a
stage's log is enough to know exactly what ran, without opening the config.

- **Logger plumbing.** `rap_mst/retrieval/*` use library loggers
  (`logging.getLogger(__name__)`) that propagate to the `rap_mst` package logger,
  onto which `get_logger` now mirrors the run's console + file handlers. Before this,
  those loggers had no handlers and did not propagate, so a module-level `info(...)`
  was silently discarded and a `warning(...)` escaped to bare stderr — which is
  exactly how the bank/gate provenance lines and the `merge_levels` warning went
  missing in the first smoke run. The mirror is idempotent (one console handler, one
  file handler per path), so nothing is duplicated.
- **Config echo.** Each stage opens with a `report_retrieval_config` block — key,
  base encoder, both level configs, `merge_levels`, gate settings — so the log
  records the decisions the run used.
- **Provenance on load.** The bank logs its row counts, key dim, encoder, fold and
  build timestamp; the gate logs the rows/patients it was fitted on, its mean
  weights and whether thresholds are locked. A leakage-guard "passed" line is
  emitted when the disjointness assertion runs.
- **The D5 invariant is checked at runtime, not just asserted in a test.** On the
  first batch, `RetrievalMemory` verifies that the image view returned only
  `level='image'` rows, the slide view only `level='slide'` rows, and the slide view
  one row per distinct patient — then logs
  `ONE bank -> TWO independent rankings (D5 verified on the first batch)`. A future
  refactor that merged the ranking would raise `D5 violation: ...` instead of
  silently degrading. (This is the "even though the store is unified it still serves
  two separate rankings" guarantee, enforced rather than trusted.)
- **Diagnostics + health banners.** `report_retrieval_diagnostics` logs
  neighbourhood size, top-1 similarity, agreement, distinct-patient count, the
  same-magnification rate (interpreted against the active route — 1.00 is expected
  under `same_mag`, but under `all` it would signal a magnification-locked key), the
  two-level disagreement (`p_slide_std`, `img_slide_disagreement` — both zero if the
  slide level were missing or merged), the gate weights and the retrieved-subtype
  histogram. `check_retrieval_health` then raises WARNING banners — mirroring
  `check_supcon_health` / `check_numeric` — for the failure modes the handbook
  enumerates: short neighbourhoods, one-patient-dominated neighbourhoods, top-1 ≈ 1.0
  (possible leak), top-1 too low (key mismatch), magnification lock, and a gate that
  has collapsed onto the parametric head. Every threshold is a documented constant in
  `reporting.py`.

---

# Part IV — Evaluation protocol and known risks

### 10.7 Evaluation protocol

Judge on the axes this analysis showed are meaningful, and pre-register them:

1. **Primary — sensitivity at matched specificity** vs the **exp3n** parametric
   baseline (`p_param`, which Stage C reports in the same file, so the comparison is
   like-for-like on identical images).
   Accuracy at a fixed 0.5 cut is threshold-dominated (`docs/results/threshold_calibration.md`).
2. **Named-case tracking.** `pat✓` out of 5 folds for DC-12312, PC-9146, TA-16184,
   MC-16456, DC-20636. Target: DC-12312 and PC-9146 → 5/5, TA-16184 stays 5/5, the
   easy 13 patients unchanged. **A regression on the easy 13 fails the module**
   regardless of aggregate gains.
3. **Ablation ladder**, each a one-line `--set` on `scripts/test.py`, each already
   predicted above (exact commands: `docs/COMMANDS.md` §11.4):
   `retrieval.key=embeddings` (expect ~0 subtype lift, mag-locked — **needs its own
   bank**, and on exp3n it is a no-op since `embeddings == features`) ·
   `retrieval.levels.image.route=all` (expect −1.1 img) ·
   `retrieval.levels.image.per_patient_cap=0` (off; expect −0.6 img, −2.6 spec) ·
   `retrieval.levels.slide.enabled=false` (expect PC/TA regress) ·
   **`retrieval.merge_levels=true` (expect degeneration to image-only: centroid
   share ~1%, PC/TA regress — D5)** · `retrieval.gate.enabled=false` (fixed α=0.5;
   expect ≈ equal — the gate should be judged on cross-model transfer, not on one
   encoder). The last two rows remove an evidence term, so they need either a
   refitted gate or `gate.enabled=false`; the loader refuses the mismatch rather
   than silently reusing a gate fitted on different evidence.
4. **Retrieval-quality metrics independent of the classifier**: patient-blocked kNN
   accuracy, neighbour subtype purity, and the k-occurrence skew, all recomputed on
   the **real per-fold bank (52–53 patients)** — where the subtype-purity cells that are `None` here
   finally become measurable.
5. **Qualitative panel.** For each named hard case, dump the top-5 retrieved exemplars
   with subtype and patient. This is the interpretability deliverable and should go in
   the paper regardless of the accuracy outcome.

### 10.8 Known risks

- **Bank-size extrapolation.** All numbers here come from a 15-patient bank. The
  deployed per-fold bank is 52–53 patients / ~4,540 images, which should improve
  retrieval, but the per-patient cap and `k` may need re-checking at that scale —
  re-run `retrieval_probe2.py`'s vote-rule comparison on the real bank before
  locking `k=15` and `cap=3`. One number already shifted at scale: the largest
  single bank patient contributes **158** image rows on fold 0 (vs 60–235 across
  the whole dataset), so the cap is doing *more* work in deployment, not less.
- **Specificity trade.** Every retrieval variant tested trades specificity for
  sensitivity (exp3: 0.876 → 0.854 at α=0.5). On a 72%-malignant test set this is
  net-positive for accuracy, but it must be reported explicitly, not buried.
- **Gate transfer.** The gate is fitted on OOF validation and applied to test. exp1
  and exp3 want very different α (0.8 vs 0.5), so the gate is doing real work — but
  its generalisation across encoders is untested and should be checked before it is
  claimed as a contribution.
- **Memory cost.** Measured, fold 0: 4,541 image rows + 52 centroid rows × 1024
  float32 = **17.5 MB** on disk (compressed `.npz`), ~19 MB resident — trivial; no
  ANN index (FAISS) needed at this scale. Exact cosine over the shard is the right
  choice, and a full 1,653-image test pass adds no measurable time.

---

# Part V — One unified memory bank instead of two modules


## 2.1 The question

D5 specifies two granularities — image-level kNN and slide-level
prototypes — because they fix **disjoint** failure cases (image kNN rescues
DC-12312; slide prototypes rescue PC-9146 and TA-16184). The architecture section
implied two stores. Can they instead be **one bank, like a database table**, and
does that cost anything?

## 2.2 The answer, in one measurement

Three storage/query designs, identical evidence, 5-fold mean, exp3:

| design | image acc | AUC | pat acc | sens | spec | centroid share of top-k |
|---|---:|---:|---:|---:|---:|---:|
| **B1** split stores, image view | 0.8863 | 0.9242 | 0.9625 | 0.917 | 0.805 | — |
| **B1** split stores, slide view | 0.8901 | 0.9266\* | 0.9625 | 0.905 | **0.851** | — |
| **B1** split stores, 50/50 blend | **0.8961** | **0.9304** | 0.9625 | 0.920 | 0.832 | — |
| **B2** ONE index, ONE top-k over the union | 0.8864 | 0.9239 | 0.9625 | 0.917 | 0.805 | **0.012** |
| **B2b** same, per-level z-scored scores | 0.8863 | 0.9242 | 0.9625 | 0.917 | 0.805 | **0.000** |
| **B3** ONE store + `level` column, two views | 0.8863 / 0.8901 | 0.9242 / 0.9266\* | 0.9625 | — | — | — |

**B3 reproduces B1 exactly.** Max absolute difference in the predicted probability,
over every image of every fold of every experiment: **0.00000000**. Bit-identical,
not merely close.

\* see the AUC caveat in §2.5 — this particular number is fragile.

**So: yes. A unified bank is a pure refactor with provably zero metric cost —
provided you unify the *storage*, not the *ranking*.**

## 2.3 The one constraint, and why it bites

B2 — the naive "just put every row in one table and take the top-k" reading — looks
harmless in the aggregate table and is actually a silent failure. Look at the
`centroid share` column: **1.2% of top-k slots** go to centroid rows. B2 is not
"both granularities combined"; B2 **is the image view**, with the slide evidence
statistically deleted. Its numbers are the image view's numbers to 4 decimals.

What that deletion costs is exactly what D5 was introduced to buy (exp3, 5-fold
mean probabilities on the named hard patients; higher is better for the two
malignants, lower for the benign):

| | specificity | DC-12312 (y=1) | PC-9146 (y=1) | TA-16184 (y=0) |
|---|---:|---:|---:|---:|
| slide view (what you wanted) | **0.851** | 0.633 | **0.681** ✓ | **0.425** ✓ |
| B2 merged single top-k | 0.805 | 0.727 | 0.590 ✗ | 0.488 ✗ |

Both architecture-mimicry cases regress to the image view's values. The naive merge
throws away the papillary and tubular-adenoma fixes.

**Why it happens** — two independent reasons, both structural, both scale-invariant:

1. **Cardinality.** ~15 centroid rows against ~1550 image rows here; ~66 against
   ~6600 in deployment. Same 1% ratio, so this does not improve with a bigger bank.
2. **Centroids lose the nearest-neighbour race by construction.** Averaging a slide
   denoises it *away* from any individual field. Mean top-1 cosine: **0.998** for
   image rows vs **0.987** for centroid rows; a centroid out-scores the best image
   on only **0.8%** of queries (exp3). In a BreaKHis bank stuffed with 60–235
   near-duplicate fields per slide, a query's top-15 is *always* individual fields.

And **per-level score normalisation does not rescue it** — B2b z-scores each level's
similarity distribution onto a common scale and the centroid share goes to
**0.000**, i.e. slightly worse. The two levels are not measuring the same quantity;
you cannot fix that by rescaling. They must be **ranked separately and combined as
evidence**, which is what the fusion gate (D6) already does.

## 2.4 The recommended design

One table. One file. Two views.

```
MemoryBank                                       # one store, one .npz, one loader
┌────────┬─────────────────┬───────┬─────────┬────────────┬──────┐
│ level  │ key [1024] (L2) │ label │ subtype │ patient_id │ mag  │
├────────┼─────────────────┼───────┼─────────┼────────────┼──────┤
│ image  │ f(x_1)          │  1    │  DC     │ SOB_M_DC.. │ 100  │
│ image  │ f(x_2)          │  1    │  DC     │ SOB_M_DC.. │ 400  │
│  ...   │                 │       │         │            │      │
│ slide  │ centroid(p_1)   │  1    │  DC     │ SOB_M_DC.. │ ALL  │   <- derived on build
│ slide  │ centroid(p_2)   │  0    │  TA     │ SOB_B_TA.. │ ALL  │
└────────┴─────────────────┴───────┴─────────┴────────────┴──────┘

bank.query(level="image", route="same_mag", k=15, per_patient_cap=3)  -> p_img
bank.query(level="slide", route="all",      k=5,  per_patient_cap=1)  -> p_slide
                                            └── separate top-k, separate temperature
```

What unification actually buys — none of it is a metric gain, all of it is
engineering and correctness:

- **One leakage assertion instead of two.** The single highest-risk part of this
  module is "did a val/test patient get into the bank?" One `patient_id` column
  checked once is materially safer than two stores that can drift apart.
- **Centroids become derived data, not a second artefact.** `build()` computes them
  by grouping the image rows. They cannot go stale, and there is no second file to
  version.
- **One `save`/`load`, one config block, one diagnostic hook.** §10.2
  loses a module: `bank.py` + `memory.py` + `gate.py`, no separate prototype store.
- **Extensibility for free.** Future granularities (region-level from
  `fpn_features`, subtype prototypes for the deferred `exp4`) are new `level` values
  and a new view — no new module, no new file, no new leakage surface.
- **Interpretability comes out uniform.** "Which archived evidence drove this
  decision" is one query against one table, mixing fields and slides.

Two design details the unification exposes, both worth stating:

- **Per-level temperature and k are mandatory, not optional.** Centroid similarities
  are more dispersed than image similarities (std 0.236 vs 0.06-ish neighbourhood
  spread), so a shared T is wrong. The measured optimum stays T=0.07 for both, but
  the *reason* differs per level and the config must let them move independently.
- **Keep one centroid per patient — not per (patient, magnification).** This was
  tested, since same-mag routing makes per-(patient, mag) centroids look natural:

  | slide prototype definition (exp3) | img acc | AUC | pat acc | sens | spec |
  |---|---:|---:|---:|---:|---:|
  | **per patient (D5, keep this)** | 0.8901 | 0.9178 | 0.9625 | 0.905 | **0.851** |
  | per (patient, mag), all-mag | 0.8922 | 0.9113 | 0.9625 | 0.907 | 0.855 |
  | per (patient, mag), same-mag routed | 0.8915 | 0.9193 | 0.9625 | 0.908 | 0.849 |

  All within noise, and on exp1 the per-(patient, mag) variants are strictly worse
  (0.8993 → 0.8935/0.8973). The whole point of the slide level is to average *away*
  individual-field ambiguity; splitting it four ways partially undoes that. Slide
  rows therefore carry `mag = ALL` and are **not** magnification-routed.

## 2.5 A correction this analysis forced — the slide-level AUC is fragile

While validating that B3 == B1, an implementation detail surfaced that changes one
number in this document.

With `slide_k=5` on a 15-patient bank, **63% of queries retrieve five centroids that
all carry the same label**, so `p_slide` is exactly 0.0 or 1.0 for those queries.
AUC over a score with 63% mass in two tied blocks is dominated by tie-handling:
two implementations whose probabilities agree to **3e-13** produce AUCs of **0.9364
and 0.9159**. The quoted **0.9266** sits inside that band; it is not wrong, it is
**not resolvable to better than ±0.02**.

Confirmed by the `slide_k` sweep (exp3, 5-fold mean):

| slide_k | img acc | AUC | pat acc | sens | spec | unanimous fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.8951 | 0.9048 | 0.9625 | 0.912 | 0.851 | 0.881 |
| **5 (D5)** | 0.8901 | ~0.92 ±0.02 | 0.9625 | 0.905 | **0.851** | 0.631 |
| 7 | 0.8889 | 0.9230 | 0.9625 | 0.906 | 0.843 | 0.608 |
| 10 | 0.8903 | 0.9269 | 0.9750 | 0.911 | 0.837 | 0.592 |
| 15 (= whole bank) | 0.8922 | 0.9253 | 0.9750 | 0.915 | 0.833 | 0.000 |

AUC rises with `slide_k` mostly because unanimity — and therefore the tie mass —
falls. **Accuracy is flat (0.889–0.895) and specificity falls monotonically
(0.851 → 0.833).** So `slide_k=5` is still the right choice, for the reason D5 gave
(specificity, and the PC/TA rescues) — but the claim *"slide prototypes deliver the
best retrieval AUC (0.9266)"* should be withdrawn.

**Corrected claim:** slide prototypes deliver the **best specificity of any pure
retrieval rule (0.851 vs 0.805)** and rescue PC-9146 and TA-16184. Judge the slide
level on specificity and named cases, not on AUC. On the deployed per-fold bank
(52–53 patients) the unanimity fraction will fall and slide-level AUC will become
measurable — re-check it then.

## 2.6 Verdict

**Yes — build one unified `MemoryBank`.** It is bit-identical to the two-store design
(max |Δp| = 0), it removes a module and a leakage surface, and it matches the
"database" mental model. The hard rule, which is the thing worth writing down:

> **Unify the store. Never unify the ranking.** One table, one `level` column, one
> leakage check — but a separate top-k per level, separate `k`, separate
> temperature, and the two evidence terms combined only at the fusion gate. A
> single top-k over the union silently degenerates into image-only retrieval
> (centroid share 1.2%) and loses exactly the PC-9146 / TA-16184 rescues the slide
> level exists to provide.

---

# Part VI — Handbook: training and inference, step by step


Written for someone running this for the first time. PowerShell, from the repo
root, with the env active:

```powershell
conda activate rapmst
Set-Location "d:\Projects\RAP-MST-v1"
```

**Mental model before you start.** There is no end-to-end training here. You train
an encoder *once*, then you **freeze it forever**. Everything after that is
bookkeeping: turn the frozen encoder's outputs into a labelled lookup table, fit
~100 parameters to decide how much to trust that table, and then predict. Nothing
after Stage A touches a GPU for more than an encoding pass, and nothing after
Stage A can be undone by a bad training run.

```
   STAGE A            STAGE B1              STAGE B2            STAGE C
 train encoder  →  build memory bank  →   fit fusion gate  →   test & report
  (5 folds)         (5 banks)              (1 gate)            (evidence + metrics)
   ~10 h GPU         ~10 min GPU            ~1 min              ~5 min GPU
  ✅ DONE for       one per fold           one, shared         also produces the
  exp1/exp2/exp3     from that fold's      across all folds    exemplar panels
  ✅ DONE exp3n      TRAIN patients                            for the paper
   -- the base       ~2 min & 17.5 MB
      encoder (D9)   per fold (measured)
```

**All three of Stages B1/B2/C are implemented** (`scripts/build_memory_bank.py`,
`scripts/train_gate.py`, `scripts/test.py --retrieval`); the commands below are
real, not planned. Copy-paste versions also live in `docs/COMMANDS.md` §11.

---

## Stage A — the encoder *(**done**; nothing to run)*

**What you have.** `runs/exp1_swin_cls/`, `runs/exp2_swin_mag_cls/`,
`runs/exp3_swin_mag_supcon_cls/` **and `runs/exp3n_swin_supcon_cls/`** each contain
all five folds with `checkpoints/best.pt` and a `test/` directory.

**The base encoder is `exp3n`, and that is settled** — Part 4 below compares it
against exp3 on every axis and it wins or ties on all of them. Stage A is complete;
go straight to Stage B1. The commands in this section are kept only so the encoder
can be reproduced from scratch.

> **Do not mix encoders.** The bank, the gate and the test run must all come from
> exp3n. The `.npz` records which encoder and which fold produced it and the loader
> refuses a mismatch, so this is enforced rather than merely advised.

<details>
<summary>Reproducing Stage A from scratch (only if the encoder must be rebuilt)</summary>

```powershell
foreach ($k in 0..4) { python scripts/train.py --experiment exp3n --fold $k }
```

Two-view SupCon (recommended, per `docs/COMMANDS.md` §9) on 4 GB:

```powershell
foreach ($k in 0..4) {
  python scripts/train.py --experiment exp3n --fold $k `
    --set data.two_view=true --set data.batch_size=8 --set train.grad_accum_steps=4
}
```

Then score them so you can compare against `docs/results/classifier_ladder.md` §2:

```powershell
foreach ($k in 0..4) {
  $run = Get-ChildItem "runs\exp3n_swin_supcon_cls" -Filter "*train_fold$k" |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
  python scripts/test.py --checkpoint "$($run.FullName)\checkpoints\best.pt"
}
```

The full parity suite (per-patient fold diagnosis, threshold calibration, UMAP +
separation metrics, all four retrieval probes) is **`docs/COMMANDS.md` §9b**. It has
been run; its verdict is Part 4 of this file.

**The checkpoint this stage had to clear, and its outcome.** The pre-registered bar
was **AUC ≥ 0.9607** (exp3's) and no collapse in patient accuracy. exp3n posted AUC
**0.9593** — a −0.0014 wash, deep inside the ±0.025 per-fold spread — while
*raising* image accuracy to 0.8986 and patient accuracy to 0.9750. Passed.

</details>

---

## Stage B1 — build the memory bank *(one per fold)*

**What this step does, in words.** Load fold *k*'s frozen `best.pt`. Run every
**training** image of that fold through it. Take `out["features"]` — the 1024-d
vector *before* the magnification block (D1) — L2-normalise it, and write it to a
table along with the image's label, subtype, patient id and magnification. Then
group those rows by patient, average, re-normalise, and append them as `slide`
rows (Part 2). Save one `.npz` per fold.

```powershell
python scripts/build_memory_bank.py --experiment exp3n          # all five folds
```

or one fold at a time (identical result; useful if you want to watch it):

```powershell
foreach ($k in 0..4) { python scripts/build_memory_bank.py --experiment exp3n --fold $k }
```

Output: `analysis/retrieval/exp3n/bank_fold0.npz` … `bank_fold4.npz`, **17.5 MB
each** (measured; compressed `.npz`), plus `bank_summary.json` and a log. The
`exp3n` component of the path is the encoder namespace — banks from different
encoders can never collide, and the loader additionally refuses a bank whose
recorded experiment/fold/key does not match the run.

**~2 minutes per fold** on an RTX 3050 (4,541 images for fold 0), ~10 minutes for
all five. If you hit CUDA OOM: `--set data.batch_size=8`.

**The rule that matters more than anything else in this file.** Fold *k*'s bank
contains **only fold *k*'s training patients**. Not its validation patients. Never
the 16 test patients. If a validation patient is in the bank, the gate you fit in
Stage B2 is fitted on retrieval results that saw the answer, and every number after
that is worthless. The script asserts this at build time *and* at query time — if
you ever see a leakage assertion fire, do not work around it.

**Sanity checks before moving on** (the script should print them):

| check | expected | fold 0, measured |
|---|---|---|
| image rows | ≈ 4,300–4,700 (the 52–53 training patients of that fold) | **4,541** |
| slide rows | = number of training patients in that fold | **52** |
| bank ∩ val patients | **empty** | empty ✓ |
| bank ∩ test patients | **empty** | empty ✓ |
| every key's L2 norm | 1.000 | 1.000000 ✓ |
| magnification shard sizes | four non-empty shards | 1143 / 1223 / 1133 / 1042 |
| key dim | 1024 (`features`, D1 — *not* 1088) | **1024** ✓ |
| max image rows for one patient | 60–235 (why the cap exists, D4) | **158** |

The script prints exactly this table and **raises** — it does not warn — if the
patient sets do not match, if a key is not unit-norm, or if a val/test patient
appears. A missing magnification shard is a warning, not an error, because
`--set data.magnifications=[...]` is a legitimate (if unusual) restriction. The
whole block, including the config it ran with and the bank's provenance, is written
to `build_memory_bank.log` in the output directory (§10.9).

> **Corrected in this revision.** The earlier version of this table said "≈5,200–5,400
> images / ≈53 patients". The real per-fold training split is 52–53 patients and
> ~4,540 images: the 66 CV patients minus that fold's 13–14 validation patients.
> §0's "~6,600-image bank" was the same over-count and is corrected
> there too.

---

## Stage B2 — fit the fusion gate *(once, not per fold)*

**What this step does, in words.** The gate answers one question: *for this image,
how much should I trust the classifier versus the memory?* It is a 5 → 16 → 3 MLP,
147 parameters, that reads five cheap signals — the head's probability, how
confident the head is, how much the neighbourhood agrees, how similar the top match
is, and how many distinct patients contributed — and outputs three weights that sum
to 1: `p_final = w_param·p_param + w_img·p_img + w_slide·p_slide`.

**Why it is fitted once, on pooled data.** The five folds' validation sets are a
disjoint partition of all 66 CV patients, so concatenating them gives full
out-of-fold coverage — every patient scored by a model that never trained on it.
`docs/results/threshold_calibration.md` §4 is the cautionary tale here: thresholds fitted
per fold on ~13 patients swung from **0.05 to 0.91** and *hurt* test. Do not fit
this per fold.

```powershell
python scripts/train_gate.py --experiment exp3n
```

Internally, for each fold: load `best.pt`, run that fold's **validation** images,
retrieve each against **that fold's** bank (asserting the bank is disjoint from
those validation patients before a single query is issued), record
`[p_param, p_img, p_slide, agreement, top1_sim, n_distinct_patients, true_label]`.
Then pool all five folds (~1,700 rows per fold, ~7,900 total / 66 patients) and fit
by full-batch Adam on binary cross-entropy. Takes seconds.

Output: `analysis/retrieval/exp3n/gate.pt` — **one gate, used by all folds** — plus
`gate_fit.json` and a log. `gate.pt` also carries the **pooled-OOF decision
thresholds** for all four probability columns at image and patient level, computed
with the same sweep as `scripts/threshold_calibration.py`; Stage C reads them from
there, so the reported "calibrated" numbers never touch the test set.

**Sanity checks before moving on** (the script prints them, and turns the first two
into hard assertions):

| check | expected | if it fails |
|---|---|---|
| pooled rows | ~7,900 images / **66** distinct patients | a fold's val set was skipped (warning) |
| any test patient present | **no** | **raises** — stop; leakage |
| all folds agree on active terms | yes | **raises** — you changed a level mid-run |
| mean learned weights | roughly (0.4–0.6, 0.3–0.5, 0.1–0.3) | see below |
| pooled val accuracy vs `p_param` alone | higher | see below |

If the gate collapses to `w_param ≈ 1.0`, the memory is not adding anything on
validation — believe it, and report the module as not helping *on this encoder*
rather than tuning until it does. §6 predicts α≈0.5 for exp3 and
α≈0.8 for exp1, so a high `w_param` is a real possible outcome, not a bug. The
script raises a warning banner in both cases rather than letting them pass quietly.

Before it fits, the script also logs a **pooled-OOF retrieval-diagnostics** block
(mean top-1 similarity, neighbourhood size, distinct-patient count, the
same-magnification rate, and the image-vs-slide disagreement — proof the two views
really are voting differently) and runs `check_retrieval_health`, which raises a
WARNING banner for any of the failure modes in the troubleshooting table below
(§10.9). So a run that is quietly broken — a magnification-locked
key, one-patient-dominated neighbourhoods, a merged slide level — announces itself
instead of surfacing only as a disappointing final number.

---

## Stage C — test and report

```powershell
foreach ($k in 0..4) {
  $run = Get-ChildItem "runs\exp3n_swin_supcon_cls" -Filter "*train_fold$k" |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
  python scripts/test.py --checkpoint "$($run.FullName)\checkpoints\best.pt" --retrieval
}
```

The bank and gate paths come from the config (`analysis/retrieval/exp3n/...`) and
the fold is read from the checkpoint, so `--bank` / `--gate` are only needed when
they live somewhere else:

```powershell
python scripts/test.py --checkpoint "<...>\checkpoints\best.pt" --retrieval `
  --bank "analysis\retrieval\exp3n\bank_fold0.npz" --gate "analysis\retrieval\exp3n\gate.pt"
```

Each fold's `test/` directory gains the following, **alongside** the existing
`test_metrics.json` / `test_predictions.csv`, which are never overwritten — so the
parametric baseline stays exactly as `docs/results/classifier_ladder.md` reported it:

- `test_metrics_retrieval.json` — metrics for `p_param`, `p_img`, `p_slide` and
  `p_final`, each at threshold 0.5 **and** at the locked pooled-OOF threshold, so
  the numbers stay comparable with `docs/results/classifier_ladder.md` and
  `docs/results/threshold_calibration.md`. Also carries the mean gate weights and the
  retrieval diagnostics (mean top-1 similarity, neighbour agreement, distinct
  patients per query, retrieved-subtype histogram).
- `test_predictions_retrieval.csv` — per image: all four probabilities, the gate
  weights, and the retrieved neighbours' patient ids, subtypes and similarities,
  for both levels.
- `exemplars/<patient_id>.json` — for each hard patient, sampled query images with
  the top-5 retrieved **images** and the top-5 retrieved **slides**, each with
  subtype and cosine similarity. **This is the interpretability deliverable and it
  goes in the paper regardless of what happens to accuracy.**

> A real example from the smoke run (fold 0, PC-9146's 40× field): the image level
> returns three fields from one ductal patient at similarity 0.996, while the slide
> level's second-ranked centroid is **the other papillary patient in the bank**
> (PC-15687B, 0.985). That is the D5 mechanism visible in one query.

### How to read the result — in this order

**1. The named cases first, not the aggregate.** `pat✓` out of 5 folds for
DC-12312, PC-9146, TA-16184, MC-16456, DC-20636. Targets: DC-12312 and PC-9146 →
**5/5**, TA-16184 stays **5/5**. **If any of the easy 13 patients regresses, the
module has failed** regardless of what the mean says.

**2. Sensitivity at matched specificity**, versus the parametric baseline. Not
accuracy at 0.5 — `docs/results/threshold_calibration.md` showed that number is
threshold-dominated. Expect retrieval to trade a little specificity for
sensitivity (exp3: 0.876 → 0.854 at α=0.5); report that trade explicitly.

**3. Aggregate accuracy last, and with the honesty caveat attached.** The test set
is 16 patients, so patient accuracy moves in steps of 1/16 = 0.0625. The predicted
+0.05 patient gain is **under one patient**. It is a direction, not a headline.

**4. Retrieval quality independent of the classifier** — patient-blocked kNN
accuracy, neighbour subtype purity, k-occurrence skew — recomputed on the real
52-patient-per-fold bank. Several subtype-purity cells that are `None` in the probes
finally become measurable here.

### The ablation ladder

Each is a one-line `--set` on `scripts/test.py`; each has a pre-registered
prediction, which is what makes the result meaningful. Exact commands:
`docs/COMMANDS.md` §11.4.

| run | change (`--set ...`) | prediction |
|---|---|---|
| key ablation | `retrieval.key=embeddings` | ~0 subtype lift, ~100% mag-locked neighbours. **Needs its own bank** (`build_memory_bank.py --set retrieval.key=embeddings`) — the key is baked into the stored vectors |
| routing | `retrieval.levels.image.route=all` | −1.1 image points |
| cap | `retrieval.levels.image.per_patient_cap=0` (off) | −0.6 image, −2.6 specificity |
| granularity | `retrieval.levels.slide.enabled=false` | PC-9146 and TA-16184 regress |
| **merged index** | `retrieval.merge_levels=true` | **degenerates to image-only; PC/TA regress** (Part 2) |
| gate | `retrieval.gate.enabled=false` (fixed α=0.5) | ≈ equal on exp3n — judge the gate on cross-encoder transfer, not here |

> **Two of these change which evidence terms exist** (`slide.enabled=false`,
> `merge_levels=true`), so a gate fitted with three terms no longer applies. The
> loader **refuses** that combination rather than silently re-normalising. Either
> refit (`train_gate.py` with the same `--set`, then pass `--gate` at the new file)
> or add `--set retrieval.gate.enabled=false` to compare at a fixed α. Both are
> honest; pick one and say which.

---

## What can go wrong, and what to do

| symptom | most likely cause | fix |
|---|---|---|
| leakage assertion at bank build | wrong split loaded, or val patients included | never bypass it; re-check `--fold` matches the checkpoint's fold |
| `Bank belongs to experiment X, not Y` | you mixed encoders | rebuild the bank for the encoder you are testing; do not mix |
| `Bank was built on key=... but retrieval.key=...` | key ablation without rebuilding the bank | rebuild with the same `--set retrieval.key=...` |
| `Gate term mask ... != current level configuration` | an ablation removed an evidence term the gate was fitted with | refit the gate with the same `--set`, or `--set retrieval.gate.enabled=false` |
| every retrieved neighbour is the same magnification | you indexed `embeddings`, not `features` | `retrieval.key=features` (D1) |
| top-15 all from one patient | per-patient cap off | `per_patient_cap=3` (D4) |
| slide evidence never appears in the results | you merged the two levels into one top-k | query the levels separately (Part 2 §2.3) |
| `p_slide` is exactly 0.5 everywhere | the slide level is disabled | `retrieval.levels.slide.enabled=true` |
| gate weights collapse to `w_param=1` | memory genuinely not helping on this encoder | report it; check the encoder passed Stage A's checkpoint |
| patient accuracy jumps a lot | one patient flipped | 1 patient = 0.0625; check `pat✓` per patient before believing it |
| CUDA OOM during bank build | batch too large for 4 GB | `--set data.batch_size=8` |

## Time and cost

Measured on an RTX 3050 (fold 0 timed end-to-end; the rest scaled from it):

| stage | GPU | wall clock (RTX 3050) | re-run when |
|---|---|---|---|
| A train exp3n × 5 folds | yes | ~10 h — **already done** | encoder changes |
| B1 build 5 banks | yes (encode only) | **~10 min** (~2 min/fold, 4,541 images) | encoder, key or split changes |
| B2 fit gate | mostly (val inference) | **~5 min**, the fit itself is seconds | bank or encoder changes |
| C test × 5 folds | yes | **~5 min** (1,653 images/fold) | anything changes |

Total to a complete retrieval result **from the exp3n checkpoints you already
have**: about **20 minutes**, all of it in B1/B2/C. Bank size is **17.5 MB** per
fold — no FAISS, no ANN index; exact cosine over a magnification shard is the right
choice at this scale.

---

# Part VII — Why the module does not help on a shared encoder


> **Follow-up (settled): `docs/results/retrieval_key_ablation.md`.** The verdict below — *"the
> module does not help on this encoder; report it, do not tune it"* — stands and is
> now far better evidenced (43 keys measured). Two things in this document were
> **corrected** by that study and should be read with it:
>
> * §3's near-rank-1 geometry is a true *description* of the key space but a false
>   *explanation* of the failure. Whitening moves effective rank across a 338×
>   range (1.08 → 364) and retrieval AUC does not move:
>   `corr(log effective rank, p_img AUC) = +0.007`.
> * "What would actually change the result" item **1** (key on `fpn_features`) is
>   now **tested and refuted** — every spatial pooling GAP discards was measured
>   and stays as redundant as the pooled vector. Item **2** is right in spirit but
>   must mean a genuinely *different encoder*: within exp3n, `projections` is the
>   most collapsed key measured (effective rank 1.08). The binding constraint is
>   that the bank and the classifier **share an encoder**, not what is keyed on.

**Question asked:** the Stage B2 gate fit (`analysis/retrieval/exp3n/train_gate.log`)
raised the `WARNING: The memory is not helping on validation` banner
(`p_final 0.8645` does **not** beat `p_param 0.8654`). Is something wrong in the
implementation, or is this a genuine property of the encoder?

**Answer:** the implementation is correct — every guard, invariant and vote is
doing what it should, and the WARNING is an *honest* negative result, not a
symptom of a bug. The module cannot help because of **what it keys on**, not how
it was coded. Evidence below.

---

## 1. The pipeline is sound (nothing to fix in the code)

Checked end-to-end against the build/gate logs and the source:

| Check | Where | Status |
|---|---|---|
| No val/test patient in any fold's bank | `assert_disjoint` at build + `train_gate.log:21,28,35,42,49` | ✓ passed all folds |
| Query patients blocked at query time | `bank.query` `query_patient_ids` mask + `block_query_patients=True` | ✓ |
| Keys L2-normalised | `bank_summary.json` `key_norm 1.0000..1.0000` | ✓ |
| "ONE store → TWO independent rankings" (D5) | `_verify_views_once`, verified on first batch each fold | ✓ (`train_gate.log:24,31,38,45,52`) |
| Per-patient cap, softmax(sim/T) vote, routing | `bank.query` D3/D4/D7 | ✓ matches spec |
| Gate fitted once on pooled OOF, not per fold | `gate_fit.json` `fit_on=pooled_oof`, 66 patients, 6256 rows | ✓ |
| Gate honestly down-weights useless evidence | learned `w=(param 0.823, img 0.146, slide 0.030)` | ✓ correct behaviour |

The gate converging to ~0.82 weight on the parametric branch and the banner firing
is the system **working as designed** — it detected redundant evidence and refused
to lean on it.

## 2. Root cause — retrieval keys on the classifier's own input vector

In exp3n the magnification embedding is off, so in the forward dict
`embeddings == features` ([rap_mst/models/rap_mst_model.py:104](../rap_mst/models/rap_mst_model.py#L104)),
and the classifier is a **single linear layer** over that vector
(`classifier.hidden_dim: null`, [config/config.yaml:79](../config/config.yaml#L79);
[rap_mst/models/classifier.py:33-37](../rap_mst/models/classifier.py#L33-L37)):

```
features (1024-d)  ──▶ Linear ──▶ logits        = p_param   (parametric readout)
        │
        └──────────▶ cosine-kNN vote            = p_img/p_slide (retrieval readout)
```

`p_param` and the retrieval vote are **two readouts of the exact same 1024-d
vector**. Retrieval therefore has no information that is orthogonal to what the
classifier already extracts — at best it re-reads the same signal, more noisily.

## 3. That vector is near rank-1, so there is only one thing to read

A direct probe of the stored keys (all five banks, image rows,
`scripts/retrieval_probe5.py`):

| Metric (mean over folds) | Value | Meaning |
|---|---|---|
| Effective rank of `features` (participation ratio) | **1.19 / 1024** | space is essentially 1-dimensional |
| Top singular direction share of variance | 0.84 – 0.96 | one axis holds almost everything |
| ‖mean of unit keys‖ | 0.84 | all keys sit in a narrow cone around one direction |
| Random-pair cosine (mean) | 0.70 | even *unrelated* images are highly aligned |
| Nearest-neighbour cosine (mean) | **0.999** | explains the `top1_sim=0.9986` in the logs |

The `top1_sim ≈ 0.999` that looked alarming in the log is **not** a leak and not a
duplicate — it is the floor of a collapsed cone: when every key points nearly the
same way, the nearest neighbour is trivially at cosine ≈ 1. The single dominant
axis is effectively the malignancy axis. A **linear** classifier reads that axis
optimally; a cosine-kNN vote reads a noisier copy of the *same* axis. There is no
rich neighbourhood structure for retrieval to exploit beyond "which side of the one
axis," which the linear head already does.

## 4. The generalisation gap proves it is redundancy, not a coding fault

Leave-patient-out kNN vote AUC, measured two ways on the identical bank:

| Retrieval evaluated on … | AUC | vs parametric head |
|---|---|---|
| **train** patients (query train image vs *other* train patients) | **0.975** | — |
| **val** patients (the real Stage B2 setting, `gate_fit.json prob_img`) | **0.868** | parametric head = **0.885** |

Retrieval is nearly perfect *within the training distribution* (0.975) — proof the
geometry genuinely carries class, and that nothing is miswired. But on **held-out**
patients it drops to 0.868, **below** the linear head's 0.885 on the same patients.
kNN overfits to the bank's patients; on a genuinely novel slide, the linear readout
of the shared axis generalises better. So fusing them cannot beat the parametric
branch alone — exactly what the gate learned (`p_final 0.8645 ≈ p_param 0.8654`).

## 5. Conclusion

> The Retrieval Memory v1 is correctly implemented. It does not improve exp3n
> because its key — the pre-magnification `features` vector — is the **same
> near-rank-1 representation the classification head already reads linearly and
> optimally**. Retrieval contributes a noisier, more overfit re-reading of
> information the parametric branch already exploits, so an honest gate collapses
> to the parametric branch. This is a property of the encoder's representation,
> **not** a defect in the retrieval code, and should be reported as *"the module
> does not help on this encoder,"* per the D9 protocol — **do not tune it to make
> the number move.**

### What would actually change the result (for future work, not a fix now)

The module only adds value if it keys on information the linear classifier does
**not** already use. Concretely:

1. **Key on `fpn_features` (the spatial multi-scale pyramid), not the pooled
   `features`.** Dense/region-level structure is discarded by GAP before the
   classifier sees it, so a region-level bank could carry orthogonal evidence.
   This is already the intended path — the forward dict exposes `fpn_features` for
   exactly this: future modules consume `features` and/or `fpn_features`.
2. **Use an encoder whose retrieval key is *not* the classifier's input** — e.g. a
   prototype/reasoning head trained with an objective that spreads the space
   (higher effective rank) instead of collapsing it onto the single class axis.
   The exp3n `features` space having effective rank ≈ 1.2 is the real ceiling here.
3. Secondary levers (routing `same_mag`→`all`, `k`, `T`) cannot lift the ceiling —
   they only change how the *same* one axis is read, and D4/D9 already fix them.

*Reproduce §3–4:* `python scripts/retrieval_probe5.py`
(reads `analysis/retrieval/exp3n/bank_fold*.npz`; needs only numpy).
