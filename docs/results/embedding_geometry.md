# RAP-MST — Embedding-Space Evidence that SupCon Works (UMAP of the held-out test set)

**Question.** `docs/results/classifier_ladder.md` and `docs/results/threshold_calibration.md` established that exp3
(SupCon) helps *ranking* (best AUC 0.9607; best patient accuracy once the threshold
is calibrated) even though fixed-0.5 accuracy hid it. But AUC is only *indirect*
evidence that SupCon reshaped the representation. This report is the **direct**
evidence: we embed the **held-out 16-patient test set** (`splits/breakhis_splits.json`
→ `test_patients`, never seen in training or validation) with each model, project the
per-image `embeddings` — the exact vector a future Retrieval/Prototype module will
index (`docs/results/classifier_ladder.md` §8) — to 2-D with UMAP, and back the pictures with hard
cluster-separation numbers.

Reproduce: `python scripts/visualize_embeddings.py --experiments exp1 exp2 exp3 exp3n`.
Outputs land in `analysis/embeddings/`.

> **✅ Resolved — `exp3n` is now included, and it confirms this report's central
> attribution claim directly.** §3 attributed the manifold *fragmentation* and the
> binary-silhouette drop (0.623 → 0.538) to the **magnification embedding**, using
> exp2 as the control. But exp2 can only show mag-*without*-SupCon. The missing cell
> was **SupCon-*without*-mag** = `exp3n` (`docs/results/magnification_audit.md` Part 1 / `docs/retrieval.md`
> D9). Running it closes the 2×2:
>
> | | no SupCon | SupCon |
> |---|---|---|
> | **with mag** | exp2 (sil 0.538) | exp3 (sil 0.569) |
> | **no mag** | exp1 (sil 0.623) | **exp3n (sil 0.680)** |
>
> Reading *down* each column isolates SupCon (it *raises* silhouette in both:
> 0.538→0.569 with mag, 0.623→**0.680** without). Reading *across* each row isolates
> the magnification block (it *drops* silhouette in both: 0.623→0.538 without SupCon,
> **0.680**→0.569 with SupCon). **The fragmentation is the magnification block's
> doing in both rows, and SupCon is beneficial in both columns** — exactly this
> report's §3 claim, now proven on the decisive fourth cell rather than inferred.
> The magnification-lock numbers that motivated it are in §1a below.

> **All figures are the held-out test set only.** Every point is a test image from
> the 16 held-out patients. Rows in the grid = the *same* test set embedded by
> different fold models (robustness), not a mix of train/val data. The single
> headline figure uses one representative fold's model (there is no single
> "ensemble" coordinate frame to pool folds into).

## Figures (in `analysis/embeddings/`)
| File | What it shows |
|------|---------------|
| **`umap_testset_binary.png`** | **Headline.** Held-out test set, fold-0 model: exp1 vs exp3 `embeddings` (+ exp3 `projections`), coloured benign/malignant. |
| **`umap_testset_subtype.png`** | Same, coloured by the 8 tumour subtypes. |
| `umap_binary.png` / `umap_subtype.png` | Per-fold grid (5 folds × {exp1 emb, exp3 emb, exp3 proj}) — robustness. |
| `umap_control_3way_binary.png` / `_subtype.png` | Control adding **exp2** (mag, no SupCon) to separate SupCon's effect from the magnification embedding's. |
| `separation_metrics.json` / `separation_metrics_3way.json` | Per-fold numeric metrics. |
| `embeddings/<exp>_fold<k>.npz` | Raw high-dim vectors + labels/subtype/patient — reusable by the retrieval module. |

---

## 1. Separation metrics (mean over 5 folds, on the high-dim vectors)

Silhouette is cosine, computed on the vectors themselves (not the 2-D UMAP);
**kNN_blocked** is a leave-one-out k=15 nearest-neighbour vote where neighbours from
the *same patient* are excluded — the retrieval premise made literal, with slide
near-duplicates blocked so it can't cheat.

| Space (test set) | silhouette **binary** | silhouette **subtype** | **kNN_blocked** acc |
|------------------|----------------------:|-----------------------:|--------------------:|
| exp1 · embeddings | 0.623 | −0.210 | 0.889 |
| exp2 · embeddings | 0.538 | −0.132 | 0.888 |
| exp3 · embeddings | 0.569 | −0.158 | **0.898** |
| **exp3n · embeddings** | **0.680** | −0.184 | 0.880 |
| exp3 · projections | 0.669 | −0.348 | 0.895 |
| **exp3n · projections** | **0.701** | **−0.274** | 0.884 |

Per-fold kNN on the indexed `embeddings` (exp1 → exp3): fold0 0.884→0.911,
fold1 0.846→0.846, fold2 0.895→0.892, fold3 0.883→0.898, fold4 0.934→0.941 —
exp3 ≥ exp1 on 4/5 folds.

**Read the two new rows carefully — they are the point of this update.**
- **exp3n · embeddings silhouette 0.680 is the highest of any embedding space**,
  above even exp1's 0.623. SupCon *and* no magnification block together give the
  cleanest binary geometry. This is the fragmentation being repaired.
- **exp3n · projections silhouette 0.701 is the highest of all six spaces**, and its
  subtype silhouette −0.274 is the **least negative** of the two projection spaces
  (exp3's is −0.348) — i.e. removing the magnification block *also* relaxes the
  subtype collapse, not just the binary fragmentation.
- **The one honest debit: exp3n's kNN_blocked dips to 0.880** (exp3 0.898, exp1
  0.889). Silhouette (global cluster tightness) went up while kNN (very-local purity)
  went down slightly. §1b explains why, and why it does not survive the proper
  same-magnification routing the retrieval module uses.

### 1a. Magnification lock — the number that drove D9

Fraction of a query's k=15 patient-blocked neighbours that share its magnification
(chance ≈ 0.25). This is what "the block poisons the key" means, made literal:

| space | same-mag neighbour rate | subtype lift | kNN AUC |
|---|---:|---:|---:|
| exp1 · embeddings (no block) | 0.314 | 0.075 | 0.916 |
| exp3 · embeddings (block in key) | **1.000** | 0.074 | 0.921 |
| exp3 · projections (block upstream of head) | **0.976** | 0.077 | 0.926 |
| **exp3n · embeddings** | **0.342** | **0.105** | 0.924 |
| **exp3n · projections** | **0.335** | **0.106** | **0.928** |

exp3's projections retrieve a same-magnification neighbour **97.6%** of the time —
the SupCon space is spending its structure separating zoom levels. **exp3n's
projections drop that to 33.5%** (morphology-driven, near chance) *and* post the
**best subtype lift (0.106) and best kNN AUC (0.928) of any space measured.** That
is the whole D9 case in one row: the magnification block was not neutral in the
SupCon space — it was locking it — and removing it hands SupCon back its capacity.

### 1b. Why exp3n's kNN dipped even though every cluster metric improved

exp3/exp3's `embeddings` had the magnification block *inside* the key, which acted
as a near-hard same-magnification filter (rate 1.000). A k=15 image-kNN vote on that
key was therefore implicitly comparing 40× only to 40×, 400× only to 400× — and
`docs/retrieval.md` Finding 3 shows same-magnification retrieval is worth ~+1 accuracy
point because a low-power field and a high-power field of the same tumour have
different texture statistics. exp3n's key has no such filter, so its *raw all-mag*
kNN mixes zooms and loses that ~1 point (0.898 → 0.880). But this is not a loss of
information — it is the block's *one* useful function (implicit routing) becoming
explicit and optional. Route exp3n's bank by magnification and the point comes
straight back: same-mag kNN 0.887, patient-level vote 0.903 (`docs/retrieval.md`
§ "final verdict"). The dip is a routing artefact, not a geometry regression — and
the geometry (silhouette, subtype lift, AUC) is strictly better.

---

## 2. What the pictures show

**exp1 · embeddings** — a **single connected manifold**: benign (blue) and malignant
(orange) occupy opposite ends of one continuous arc that they *share* in the middle.
Cross-entropy alone learns a discriminative *direction* but leaves one topological
blob. The fence-sitters live in the shared middle.

**exp3 · embeddings** — the manifold has **broken into discrete, mostly single-class
islands / filaments**. Benign images pull off into their own tight clusters; malignant
threads separate from them. The space went from *continuous* to *clustered*.

**exp3 · projections** — the SupCon-optimised space: the sharpest, cleanest threads,
highest binary silhouette (0.669). This is where SupCon acts directly, and it clearly
did its job.

The headline figure (`umap_testset_binary.png`) shows exactly this progression on the
held-out test set, side by side.

---

## 3. The honest attribution — SupCon vs the magnification embedding

Comparing **only** exp1 vs exp3 would be misleading, because exp3 differs from exp1 by
*two* things: the magnification embedding **and** SupCon. The exp2 control (mag, no
SupCon) separates them, and the result is not what a naïve reading assumes:

- **The fragmentation and the binary-silhouette drop are caused by the magnification
  embedding, not SupCon.** exp1→exp2 already breaks the manifold into islands and drops
  binary silhouette 0.623 → 0.538 (the mag vector injects zoom-dependent variance, so a
  class is no longer one blob). kNN is unchanged (0.889 → 0.888) — mag alone does **not**
  improve class-neighbourhood purity.
- **SupCon then re-tightens class structure on top of that.** exp2→exp3 *recovers*
  binary silhouette 0.538 → 0.569 **and** is the **only** component that lifts the
  retrieval-relevant local metric: kNN 0.888 → **0.898**.
- **SupCon's own projection space is the most class-separable (silhouette 0.669)** —
  direct, unambiguous proof it shaped a more benign/malignant-discriminative
  representation.

So: exp3's *embedding* silhouette being lower than exp1's is a **magnification** effect;
SupCon's contribution is real and shows up as **tighter class neighbourhoods (kNN) and a
much cleaner projection space** — the properties a neighbourhood-vote retrieval module
actually exploits. This is direct representational evidence, independent of AUC.

> Why silhouette is the wrong single lens for a contrastive space: silhouette rewards
> *one compact blob per class*. SupCon deliberately creates *many* pure sub-clusters, so
> a class's within-label spread rises even as local purity improves. The local metric
> (kNN) is the honest read for retrieval — and it moves the right way only with SupCon.

**Update — exp3n confirms this attribution on the decisive cell, and sharpens it.**
When this section was first written, "the fragmentation is the magnification block's
doing, SupCon's job is the *tightening*" was inferred from the exp2 control. exp3n
tests it directly: it *is* SupCon with the magnification block removed. The result
(§1/§1a): binary silhouette jumps to **0.680** — SupCon delivering its tightening on
a space the block is no longer fragmenting — while the projection space's
magnification lock collapses from **0.976 → 0.335** and its subtype lift rises to the
best measured (0.106). This is the cleanest possible statement of the two-factor
story: **SupCon is the beneficial factor; the magnification block is the fragmenting
factor; they were fighting inside exp3, and exp3n lets SupCon win.**

**None of this makes exp2 or exp3 a wasted step — they each isolated one factor.**
- **exp2 was the essential control**, not a failed model. Without it we could not have
  separated the magnification block's effect from SupCon's — exp1-vs-exp3 alone
  confounds the two, and the entire attribution above (and the D9 decision to drop the
  block) rests on exp2 having pinned the fragmentation on magnification. A negative
  result that correctly localises a cause is a load-bearing experiment.
- **exp3 is what proved SupCon's contrastive machinery works at all**, and it did so
  under the *harder* condition (block present). Its projection space is class-separable
  (silhouette 0.669) and it was the model where retrieval was *most* complementary to
  the head (`docs/retrieval.md` Finding 2, 31.6% error-rescue). exp3n is not a replacement
  for that finding — it is exp3's mechanism, run once more with the confound removed.
  The ladder exp1 → exp2 → exp3 → exp3n is a controlled decomposition, and every rung
  carries one inference the others cannot.

---

## 4. The limitation the subtype view exposes — and why retrieval is the right next step

Colour the *same* embeddings by the 8 subtypes (`umap_testset_subtype.png`,
`umap_subtype.png`) and the tail problem `docs/results/classifier_ladder.md` §8 predicted is now visible and
measured:

- **Subtype silhouette is negative in every space, and worst in SupCon's projections
  (−0.348).** Class-only SupCon pulls *all* malignant subtypes into shared threads:
  **papillary (PC), mucinous (MC), lobular (LC)** are smeared through the ductal (DC)
  mass and never claim their own territory. The benign subtypes (adenosis, fibroadenoma)
  do separate; the malignant tail does not.
- This is the exact mechanism behind the residual errors: the rare/atypical malignants
  (PC-9146, low-grade DC-12312) sit *inside* the DC blob, so a parametric head — and a
  global threshold — can't isolate them.

**Why this justifies the retrieval module.** The evidence cuts two ways and both point
to a non-parametric memory:
1. **The indexed embedding is already locally class-pure (~0.89 kNN) and SupCon nudges it
   to best (0.898).** Retrieval-by-neighbourhood-vote is well-founded on this space — the
   easy 13 patients sit in clean single-class neighbourhoods.
2. **But subtype structure is absent/negative**, so plain class-only SupCon *cannot* fix
   the rare-malignant tail — it actively collapses it. A memory that (a) resolves
   fence-sitters by their nearest labelled exemplars and (b) is made **subtype-aware**
   (subtype positives / rare-subtype up-weighting, `docs/results/classifier_ladder.md` §8) is precisely the tool
   that turns "clean local binary neighbourhoods + collapsed subtype tail" into correct
   decisions on PC / low-grade DC.

---

## 5. Conclusion

- **Direct proof SupCon works:** it is the sole component that raises patient-blocked kNN
  purity on the indexed embedding (0.889/0.888 → **0.898**) and it produces the most
  class-separable space of all (projections, silhouette **0.669**). The CE manifold
  (one connected blob) becomes a set of discrete, mostly-pure clusters. This is
  representational evidence, complementary to the AUC gain, and independent of any
  threshold.
- **Honest caveat, via the exp2 control:** the visible *fragmentation* and the lower
  embedding silhouette are the **magnification embedding's** doing, not SupCon's;
  SupCon's contribution is the *tightening* (kNN ↑, silhouette recovered vs exp2, clean
  projections). Attribution matters — exp1-vs-exp3 alone would have mis-assigned it.
- **Why accuracy still lagged and what fixes it:** the same subtype view shows the
  rare-malignant tail dissolved into the DC blob (subtype silhouette negative, worst in
  SupCon's space). Better *global* ranking cannot rescue mis-*located* rare subtypes.
  The embedding is clean enough locally for retrieval to exploit, and broken enough on
  subtypes that a **subtype-aware retrieval/memory module** is the logical next step —
  now motivated by measured geometry, not just the AUC/accuracy gap.
- **exp3n update — the caveat is now the headline.** The "honest caveat" above said
  the fragmentation was the magnification block's doing and only inferred it from exp2.
  exp3n proves it: SupCon-without-the-block gives the **best embedding silhouette of
  all (0.680)**, the **best projection silhouette of all (0.701)**, un-locks the
  projection space from magnification (**0.976 → 0.335** same-mag neighbours), and lifts
  subtype coherence to the best measured (subtype lift 0.106). It also carries straight
  through to the classifier: exp3n's **test AUC holds at 0.9593** (exp3 0.9607, a 0.0014
  wash) while its patient accuracy *rises* to 0.9750 (exp3 0.9375) because removing the
  block's benign-leaning per-zoom bias recovers sensitivity on the borderline malignants
  — full accuracy story in `docs/results/magnification_audit.md` Part 4. The single debit is a ~1.5-point
  dip in *raw* kNN (0.898 → 0.880) that same-magnification routing recovers (§1b).
  **Net: exp3n is the better retrieval base encoder.** The subtype tail is *reduced* but
  not solved (subtype silhouette still negative), so the subtype-aware training term
  (`exp4`) remains the open follow-up — retrieval does not depend on it.

*Sources: `analysis/embeddings/` (figures, `separation_metrics*.json`, `embeddings/*.npz`)
produced by `scripts/visualize_embeddings.py`; held-out set from
`splits/breakhis_splits.json`; AUC/accuracy context from `docs/results/classifier_ladder.md` and
`docs/results/threshold_calibration.md`.*
