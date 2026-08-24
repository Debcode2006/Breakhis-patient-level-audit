# The magnification embedding — audit, and the choice of base encoder

*Paper §4.2 ("Magnification Conditioning"), Fig. 3, and Table 4. This is the report
behind the paper's claim that explicit magnification conditioning acts as a four-value
per-zoom logit offset rather than as a change to the representation.*

**Two parts.**

- **Part 1 — the audit.** What the 64-d magnification block was supposed to do, what
  it structurally *can* do, what it measurably *does* under counterfactual inputs, and
  what it costs. *Verdict: it is a bias term dressed up as a change to the
  representation; retire it from the retrieval line and report it as a documented
  negative result.*
- **Part 4 — the consequence.** exp3 versus exp3n as the base encoder for the
  retrieval memory. *Settled: **exp3n**, and the code is built on it.*

(Parts 2 and 3 of the original working document were the unified-bank design and the
operator handbook; they now live in [`../retrieval.md`](../retrieval.md) Parts V and VI.
Part numbering is preserved so existing cross-references still resolve.)

**Evidence.** `scripts/retrieval_probe4.py` → `analysis/retrieval_probe/probe4.json`,
run on the saved held-out test embeddings under the same strict leave-one-patient-out
protocol as probes 1–3:

```
python scripts/retrieval_probe4.py --experiments exp1 exp2 exp3
```

Probe 4 adds one thing probes 1–3 could not do: it loads the **frozen classifier
weights** out of each `best.pt` and re-runs the decision head under counterfactual
magnification inputs. No dataset access, no re-training — the head is
`Dropout → Linear(1088 → 2)`, so it is exactly reproducible in numpy from the saved
1088-d vectors. `analysis/retrieval_probe/probe4.json` is also the data source for
**Figure 3** of the paper (`scripts/figure_panels.py`).

> Same caveat as every probe: the bank here is the other 15 test patients (~1,550
> images), not the deployed per-fold training bank (52–53 patients / ~4,540 images).
> Absolute numbers are a lower bound; the **relative** comparisons are the signal.

---

# Part 1 — The magnification embedding: what it is for, and whether it earns its place

## 1.1 What it was supposed to do

The stated intent (`models/magnification.py`, the paper) is **conditioning**: one
unified model sees 40×, 100×, 200× and 400× fields, and those are genuinely
different imaging regimes — a 40× low-power field shows architecture (gland
shapes, stromal pattern), a 400× field shows cytology (nuclear size, chromatin).
Telling the model which regime it is looking at should let it apply
regime-appropriate reasoning instead of averaging four visual grammars into one
decision rule.

That is a good hypothesis. The question is whether the *implementation* — a 64-d
`nn.Embedding` lookup concatenated onto the 1024-d fused feature — can deliver it.

## 1.2 What it structurally *can* do — the decisive fact

`model.classifier.hidden_dim` is `null`, so the classification head is **one linear
layer**. For a linear head, concatenating a value that is *constant per
magnification* cannot interact with the image feature at all. Write the head as
`W = [W_feat | W_mag]`; then

```
logit_malignant − logit_benign  =  (W_feat[1]−W_feat[0])·f  +  (W_mag[1]−W_mag[0])·E[mag]
                                   \_______ image-driven _______/   \___ 4 constants ___/
```

The entire magnification embedding — 4 × 64 = 256 parameters — collapses at the
decision into **four scalars**: one logit offset per magnification. It is a learned
**per-zoom prior**, not conditioning. It cannot change the *ranking* of images
within a magnification, only where the threshold sits for that magnification.

Measured, averaged over 5 folds:

| | 40× | 100× | 200× | 400× | spread | image-driven logit std | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp2 offset | −0.141 | +0.033 | +0.012 | +0.297 | 0.582 | 4.597 | **0.127** |
| exp3 offset | −0.322 | −0.117 | −0.044 | +0.289 | 0.674 | 4.224 | **0.160** |

The offsets are **monotone in zoom** for exp3 — the model learned "be more willing
to call malignant at high power, more conservative at low power," which is a
sensible pathology prior and a real (if modest) thing to have learned. But the
whole span of that prior is ~0.6 logits against an image-driven signal with a
standard deviation of ~4.4. It is a nudge, not a decision.

## 1.3 What it measurably *does* — the counterfactual

Frozen head, same features, magnification block replaced (5-fold mean):

| exp3 head input | image acc | **AUC** | mean \|Δp\| vs. true |
|---|---:|---:|---:|
| **true magnification (the deployed model)** | 0.8858 | **0.96068** | — |
| block zeroed out entirely | **0.8867** | 0.96050 | 0.0083 |
| block = table mean (a pure constant) | 0.8854 | 0.96050 | 0.0084 |
| every image forced to 40× | 0.8812 | 0.96050 | 0.0119 |
| every image forced to 100× | 0.8838 | 0.96050 | 0.0087 |
| every image forced to 200× | 0.8863 | 0.96050 | 0.0082 |
| every image forced to 400× | **0.8904** | 0.96050 | 0.0121 |

Read the AUC column first. **Every counterfactual gives AUC 0.96050; the real model
gives 0.96068.** The magnification embedding's total contribution to exp3's ranking
quality is **+0.00018 AUC**. exp2 is the same story: 0.95496 with, 0.95464 without
— **+0.00032**.

And it is AUC-neutral *by construction*: forcing every image to a single
magnification adds one constant to every logit, which cannot reorder anything. The
only way the block can move AUC is by making the offsets *differ* across
magnifications — and that difference is worth 0.0002.

At the 0.5 threshold, deleting the block changes exp3 accuracy by **−0.0009** (i.e.
zeroing it is very slightly *better*), and forcing everything to 400× is better
still (0.8904). The block is doing nothing the model needs.

Two things this does **not** show, stated plainly:

- It does not measure the block's effect *during training*. Gradients flow through
  it into the backbone, so exp2's features are not exp1's features. That effect is
  measured elsewhere, and it is negative — see §1.5.
- It applies to the **linear** head. Through the **projection head** (an MLP) the
  block is *not* a bias, and there it does real damage — see §1.4.

## 1.4 What it costs

**(a) It destroys the retrieval key.** From `docs/retrieval.md` Finding 1: with the block
in the key, **99.6–100%** of retrieved neighbours share the query's magnification
against a 25.1% chance rate. The block alone retrieves at kNN accuracy 0.473 and
AUC **0.361** — *anti*-correlated with the label — and its subtype lift is **−0.221**,
i.e. it actively pushes same-subtype images apart. Its 64 dimensions carry ~8% of
the vector's squared norm (‖block‖ 8.00 vs ‖feature‖ 28.19) and take only **4
distinct values** in the entire dataset.

**(b) It contaminates the space SupCon shapes.** This is the part that matters most
for the user's instinct, and it is the one place the block is *not* a harmless
bias. The projection head is `Linear → BN → ReLU → Linear` applied to the
**1088-d** vector — so SupCon optimises a space built on top of the magnification
block, and the block warps it nonlinearly. Result: exp3's `projections` are **97.6%
magnification-locked**. SupCon never asked for that. It is spending representational
capacity separating zoom levels — a nuisance factor — inside the very space whose
job is to separate benign from malignant.

`docs/results/embedding_geometry.md` §3 already isolated this with the exp2 control and
attributed it correctly: the manifold **fragmentation** and the binary-silhouette
drop 0.623 → 0.538 are the **magnification embedding's** doing, not SupCon's;
SupCon's own contribution (0.538 → 0.569, kNN 0.888 → 0.898) is partly spent
*repairing* that fragmentation. That is exactly "it breaks the embedding space into
fragments and SupCon has a hard time rebuilding it," and it is measured, not felt.

**(c) It costs accuracy end-to-end.** From `docs/results/classifier_ladder.md` §2: exp1 → exp2 is
−0.008 image accuracy, −0.025 patient accuracy, −0.019 sensitivity, with flat AUC
(0.9559 → 0.9550). Per magnification (probe 4 A4), exp2's head is worse than exp1's
at **three of four** zoom levels:

| | 40× | 100× | 200× | 400× |
|---|---:|---:|---:|---:|
| exp1 acc / AUC | **0.909** / **0.969** | **0.923** / **0.970** | 0.886 / 0.933 | **0.903** / 0.957 |
| exp2 acc / AUC | 0.893 / 0.959 | 0.906 / 0.961 | **0.890** / **0.941** | 0.900 / **0.966** |

If the mechanism were "conditioning helps the model read each zoom correctly," this
table would show a per-zoom gain. It does not.

## 1.5 Is it helping *anywhere*? An honest search

Three places it could plausibly be earning its keep. All three were checked.

**(i) Redundancy — is it at least *new* information?** Yes, and this is the one
result that goes in its favour. Magnification is only weakly recoverable from the
1024-d feature: patient-blocked kNN recovers the correct magnification at
**0.353 / 0.378 / 0.393** (exp1/exp2/exp3) against a 0.261 majority-class baseline,
and the same-magnification neighbour rate is only 0.31–0.34 vs 0.25 chance. So the
feature does **not** already encode zoom, and the embedding is genuinely
non-redundant information. The problem is not that the information is already
there — it is that **a linear head can only use it as a bias**.

**(ii) Per-magnification decision offsets — is the prior worth anything at all?**
This is the block's actual mechanism, so it deserves its best case. Compare a
single accuracy-optimal threshold against four per-magnification
accuracy-optimal thresholds, both fitted **on the test set itself** (an oracle, an
upper bound that no honest procedure can reach):

| | acc @0.5 | oracle single threshold | oracle per-magnification | headroom |
|---|---:|---:|---:|---:|
| exp1 | 0.9054 | 0.9176 | 0.9273 | **+0.0097** |
| exp2 | 0.8974 | 0.9166 | 0.9227 | **+0.0060** |
| exp3 | 0.8858 | 0.9218 | 0.9303 | **+0.0085** |

The entire theoretical value of magnification-conditioned decision-making, with
perfect hindsight, is **≤1 accuracy point** — and `docs/results/threshold_calibration.md`
§4 already showed that operating points fitted on realistic amounts of held-out
data lose most of their oracle gain. Note the internal consistency: exp2's headroom
is the **smallest** (+0.0060), because exp2 has already absorbed part of the per-mag
offset into its weights. That confirms the mechanism *and* bounds its value.

**(iii) Zoom-aware retrieval — the original motivation.** `docs/results/classifier_ladder.md` §8 wanted
retrieval to be "zoom-aware." The block delivered that implicitly and expensively.
`docs/retrieval.md` D3 delivers it explicitly and better: shard the bank by
magnification and retrieve from the query's own shard — **+1.1 image points** on
exp3 (0.8863 → 0.8974), which is *exactly* the accuracy the mag-block-in-key
achieved (0.8975), but with the 1024-d key's better subtype geometry (subtype lift
0.074 → 0.101) intact. So the user's read is right: **retrieval only needs `mag_id`
as a metadata column, and carrying it as metadata strictly dominates baking it into
the vector.**

## 1.6 Verdict

**The magnification embedding, as implemented, is a 4-value per-zoom threshold prior
worth +0.0002 AUC, purchased at the price of a magnification-locked retrieval key
and a magnification-locked SupCon space.** It is a no-op where it could help (the
linear classifier) and a real distortion where it hurts (the projection head that
SupCon optimises and the vector the memory will index).

**Recommendation — three parts.**

1. **Add `exp3n` = exp3 minus the magnification embedding** (already added to
   `rap_mst/experiments.py`; exp1–exp3 untouched) and make **that** the retrieval
   base encoder instead of exp3. This is not just hygiene — it makes the space
   SupCon shapes and the space retrieval indexes the *same 1024-d space*, which is
   currently not true.
2. **Keep exp1 → exp2 → exp3 in the paper unchanged.** exp2 is now a *good* negative
   result: the mechanism is measured (a per-zoom bias, AUC-neutral by construction),
   the cost is measured (fragmentation, key contamination), and the correct
   alternative is measured (same-mag routing). That is a stronger contribution than
   a component that quietly helped by 0.002.
3. **If you ever want real magnification conditioning**, do not concatenate before a
   linear head. Condition the *features*, e.g. FiLM (per-magnification scale/shift)
   on the FPN levels, or a magnification-conditioned gate in `FeatureFusion`. Then
   the signal multiplies the image evidence instead of adding a constant to it. This
   is a separate experiment (`exp6`), and it should be judged against the ≤0.01
   oracle headroom in §1.5(ii) before anyone invests in it.

**These were falsifiable predictions — exp3n has now been trained and measured.**
Every one of the D9 predictions was registered *before* the run. Actuals (5-fold
mean, held-out test set):

| prediction | expected | **exp3n actual** | verdict |
|---|---|---|---|
| test AUC ≥ exp3 (no drop > 0.005) | ≥ 0.9607 or within 0.005 | **0.9593** (−0.0014) | ✅ wash, within noise |
| retrieval key same-mag neighbour rate | ≈ 0.33 (not 1.00) | **0.342** | ✅ |
| `projections` same-mag neighbour rate | ≈ 0.33, down from 0.976 | **0.335** | ✅ **decisive** |
| binary silhouette of `embeddings` | ≥ 0.62, up from 0.569 | **0.680** | ✅ beats exp1 (0.623) |
| error-rescue rate (complementarity) | ≥ 0.30 | **0.242** | ⚠ below — see below |

Four of five confirmed, three of them cleanly. The full comparison and the honest
reading of the fifth row are **Part 4** below — including the train/test-accuracy
result (exp3n *beat* exp3: image acc 0.8858 → 0.8986, patient acc 0.9375 → 0.9750),
which was not part of the D9 geometry predictions but is the headline result.
**The magnification block is confirmed retired; exp3n is the retrieval base.**

---

# Part 4 — Final verdict: exp3 vs exp3n as the retrieval base encoder

exp3n has been trained on all 5 folds and put through the full suite (test metrics,
threshold calibration, UMAP + separation metrics, all four retrieval probes). This
part collects the numbers and makes the call.

> **This decision is now closed and implemented.** `exp3n` is the retrieval base
> encoder. `config/config.yaml` records it as `retrieval.base_experiment: exp3n`,
> the `exp5` preset in `rap_mst/experiments.py` is exp3n + `retrieval.enabled:
> true` (it reuses exp3n's frozen checkpoints — it does **not** re-train an
> identical encoder), and every bank file records the encoder that produced it so a
> mismatched combination raises instead of silently producing numbers. exp3 remains
> in the paper as a ladder rung; nothing downstream reads its checkpoints.

## 4.1 The headline — exp3n *beat* exp3 on the classifier itself

Contrary to the worry that removing a component would hurt, exp3n is a **better
classifier** than exp3 on the held-out 16-patient test set (5-fold mean):

| metric | exp1 | exp3 | **exp3n** | exp3n − exp3 |
|---|---:|---:|---:|---:|
| image accuracy | 0.9054 | 0.8858 | **0.8986** | **+0.0128** |
| image AUC | 0.9559 | 0.9607 | 0.9593 | −0.0014 |
| sensitivity | 0.9242 | 0.8894 | **0.9193** | **+0.0300** |
| specificity | 0.8563 | 0.8764 | 0.8445 | −0.0319 |
| **patient accuracy** | 0.9875 | 0.9375 | **0.9750** | **+0.0375** |
| patient AUC | 0.9958 | 0.9958 | **1.0000** | +0.0042 |

And the hard cases `docs/results/classifier_ladder.md` §3 named all move the right way (per-patient mean
P(malignant), 5-fold; higher is better for the malignants, lower for the benign):

| patient | y | exp3 (pat✓) | **exp3n** (pat✓) |
|---|---|---|---|
| DC-12312 low-grade ductal | 1 | 0.577 (3/5) | **0.712 (4/5)** |
| PC-9146 papillary, rare | 1 | 0.639 (3/5) | **0.726 (5/5)** ✅ |
| DC-20636 borderline | 1 | 0.916 (5/5) | **0.986 (5/5)** |
| MC-16456 mucinous | 1 | 0.852 (5/5) | 0.849 (5/5) |
| TA-16184 tubular adenoma | 0 | 0.316 (4/5) | 0.370 (4/5) |

**Why it improved — and why "we hoped the opposite" was the wrong worry.** The tell
is that **image AUC is flat** (−0.0014, deep inside the ±0.025 per-fold spread).
Removing the magnification block did **not** change how well the model *ranks*
images; it changed *where the 0.5 cut falls*. Part 1 established that the block was a
per-magnification **logit bias**, monotone in zoom, that pushed the operating point
benign-ward — which is exactly why `docs/results/threshold_calibration.md` found exp3's
accuracy-optimal cut was 0.36, the most mis-calibrated of all. On a 72%-malignant
test set, a benign-leaning bias at a fixed 0.5 threshold silently converts borderline
malignants (DC-12312, PC-9146, DC-20636) into false negatives. Delete the bias →
sensitivity recovers on exactly those patients (+0.030) → accuracy at 0.5 rises. The
−0.032 specificity is the flip side (TA-16184 drifts up 0.316 → 0.370). On this
prevalence the trade nets **positive**.

So the accuracy gain is real but is a **calibration effect, not new discrimination**
— the AUC wash is the honest core, and it says the block was never contributing
ranking power, only mis-placing the threshold. Two caveats kept in view: exp1 still
leads *raw* accuracy (0.9054), and the +0.0375 patient gain is ~0.6 patients on a
16-patient set — right direction, tiny-N, same quantisation band as everything else
in this project.

## 4.2 The geometry — why exp3n is the better *retrieval* base (the real reason we care)

The classifier win is a bonus. The reason exp3n was proposed is the embedding
geometry, and the probes settle it decisively.

**Magnification lock — gone.** Fraction of a query's 15 blocked neighbours sharing
its magnification (chance ≈ 0.25):

| space | same-mag rate | subtype lift | kNN AUC |
|---|---:|---:|---:|
| exp3 · embeddings (block in key) | 1.000 | 0.074 | 0.921 |
| exp3 · projections (block upstream) | **0.976** | 0.077 | 0.926 |
| **exp3n · embeddings** | **0.342** | **0.105** | 0.924 |
| **exp3n · projections** | **0.335** | **0.106** | **0.928** |

exp3's SupCon projection space retrieved a same-magnification neighbour 97.6% of the
time — it was spending its structure on a nuisance factor. **exp3n's projections drop
that to 33.5%** and simultaneously post the **best subtype lift and best kNN AUC of
any space measured.** This is the whole argument in two rows: the block was locking
the SupCon space to magnification, and removing it hands that capacity back to
morphology.

**Cluster structure — best of all spaces** (separation metrics, 5-fold mean):

| space | binary silhouette | subtype silhouette | kNN_blocked |
|---|---:|---:|---:|
| exp1 · embeddings | 0.623 | −0.210 | 0.889 |
| exp3 · embeddings | 0.569 | −0.158 | **0.898** |
| **exp3n · embeddings** | **0.680** | −0.184 | 0.880 |
| exp3 · projections | 0.669 | −0.348 | 0.895 |
| **exp3n · projections** | **0.701** | **−0.274** | 0.884 |

exp3n has the highest binary silhouette of any embedding space (0.680, above exp1)
and the highest of any space overall in its projections (0.701), with the subtype
collapse *relaxed* (−0.274 vs exp3's −0.348).

**Complementarity — the ⚠ row, read honestly.** The one prediction that landed below
target: error-rescue fell to 24.2% (exp3 was 31.6%). This is **not** retrieval
getting worse — it is the parametric head getting *better*:

| exp | param acc | retrieval acc | rescue % of head errors | oracle-of-two | corr |
|---|---:|---:|---:|---:|---:|
| exp1 | 0.9054 | 0.8886 | 20.8% | 0.9257 | 0.938 |
| exp3 | 0.8858 | 0.8863 | 31.6% | 0.9266 | 0.906 |
| **exp3n** | **0.8986** | 0.8801 | 24.2% | **0.9272** | 0.915 |

exp3's high rescue rate was partly an artefact of a *weaker* head (0.8858) with more
errors to rescue. exp3n's head is stronger (0.8986), so the *percentage* rescued
drops — but the **oracle-of-two ceiling actually rises to 0.9272** (the best of the
three) and the head/memory correlation stays low (0.915, well under exp1's 0.938).
Translation: retrieval on exp3n has a higher combined ceiling and is still genuinely
complementary; it just starts from a better baseline. That is a better position to
build on, not a worse one.

**The one debit, and its fix.** exp3n's *raw all-magnification* kNN dips to 0.880
(exp3 0.898). Cause: exp3's key had the block *inside* it, which acted as an implicit
same-magnification filter worth ~1 point (`docs/retrieval.md` D3). exp3n's key has no such
filter, so raw kNN mixes zooms. But that function is exactly what D3 makes explicit:
route exp3n's bank by magnification and it comes back —

| exp3n retrieval vote | image acc | AUC | patient acc |
|---|---:|---:|---:|
| image kNN, all-mag (raw) | 0.8801 | 0.9242 | 0.9750 |
| image kNN, **same-mag routed** | 0.8871 | — | 0.9750 |
| **patient-capped (≤3/patient)** | 0.8957 | 0.9123 | 0.9625 |
| **patient-level (slide centroids)** | **0.9031** | 0.9182 | 0.9750 |
| head + slide blend (α=0.5) | 0.9022 | **0.9509** | 0.9750 |

With the D3/D4/D5 machinery the module is already specified to use, exp3n retrieval
reaches **0.903 image accuracy** — its slide-level vote is the best pure-retrieval
accuracy of any experiment — and the head+memory blend hits **0.9509 AUC**. The raw
kNN dip is a routing artefact, not a geometry regression.

## 4.3 Decision and trade-off ledger

**Decision: build the retrieval module on `exp3n`.** It wins or ties on every axis
that matters for retrieval, and as a bonus it is the better standalone classifier.

| axis | exp3 | exp3n | who wins | how much it matters |
|---|---|---|---|---|
| projection mag-lock | 0.976 | **0.335** | exp3n | **decisive** — this is the whole reason |
| subtype lift (key) | 0.074 | **0.105** | exp3n | large — cleaner rare-subtype neighbourhoods |
| embedding silhouette | 0.569 | **0.680** | exp3n | large — fragmentation repaired |
| test AUC | **0.9607** | 0.9593 | tie | wash (−0.0014, within noise) |
| test image acc | 0.8858 | **0.8986** | exp3n | real, calibration-driven |
| test patient acc | 0.9375 | **0.9750** | exp3n | real but tiny-N (~0.6 patient) |
| raw all-mag kNN | **0.898** | 0.880 | exp3 | small; recovered by same-mag routing |
| error-rescue % | **31.6%** | 24.2% | exp3 nominally | illusory — exp3n's oracle ceiling is higher |
| key ≡ SupCon space ≡ index | no (1088 vs 1024) | **yes (1024)** | exp3n | architectural — one space, one code path |

**How much does the downside actually cost?** The only genuine debits are (a) a
0.0014 AUC wash — unmeasurable; (b) a 1.5-point raw-kNN dip that same-mag routing
erases; and (c) a 0.032 specificity trade that is net-positive on this prevalence and
was already flagged as an inherent property of every retrieval variant
(`docs/retrieval.md` §10.8). Against those: the SupCon space stops being magnification-
locked, subtype geometry improves, and the retrieval key becomes identical to the
space SupCon optimises. There is no axis on which exp3n is meaningfully worse.

## 4.4 What this does *not* change, and what stays open

- **exp1/exp2/exp3 remain in the paper, unchanged and load-bearing.** exp2 is the
  control that localised the fragmentation to magnification (without it, exp1-vs-exp3
  confounds the two factors); exp3 is where SupCon's machinery was proven to work at
  all, under the harder block-present condition. exp3n is exp3's mechanism with the
  confound removed — it *depends* on those rungs, it does not retire them. See
  `docs/results/embedding_geometry.md` §3.
- **The subtype tail is reduced, not solved.** exp3n's subtype silhouette is still
  negative (−0.184 embeddings, −0.274 projections). PC's blocked binary-kNN improves
  (0.618 → 0.691) but MC (0.883 → 0.838) and TA (0.544 → 0.497) slip slightly. The
  subtype-aware training term (`exp4`, D8) remains the open follow-up — and the
  retrieval module is explicitly designed **not** to depend on it.
- **All numbers are still the 15-patient probe bank.** The deployed bank is fold
  *k*'s own 52–53 training patients (~4,540 images, measured), which should improve
  absolute retrieval accuracy; re-check `k`, the cap, and the slide-level AUC on it
  before locking them (`docs/retrieval.md` §10.8).

---
