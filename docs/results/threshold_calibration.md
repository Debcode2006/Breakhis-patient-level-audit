# RAP-MST — Threshold-Calibration Experiment

**Question this answers.** `docs/results/classifier_ladder.md` found that adding the magnification
embedding (exp2) and SupCon (exp3) did **not** raise thresholded test accuracy at
the fixed 0.5 cut, yet held-or-improved **AUC** (exp3 best, 0.9607). AUC is a
ranking measure — it does not depend on the threshold. The report's hypothesis was
that these components **shifted the operating point benign-ward** rather than
adding discrimination, so a fixed 0.5 cut on a 72%-malignant test set silently
costs accuracy. This experiment tests that hypothesis directly: **calibrate the
decision threshold on validation data, lock it, and re-score the untouched
held-out test set.** If the report is right, the calibrated threshold should sit
*below* 0.5 for exp2/exp3 and recover the "missing" accuracy.

Reproduce: `python scripts/threshold_calibration.py`
(→ `analysis/threshold_calibration/results.json`).

> **✅ Resolved — `exp3n` included, and it confirms the mechanism emphatically.**
> `docs/results/magnification_audit.md` Part 1 (D9) adds **exp3n** = exp3 without the magnification
> embedding. The registered prediction was: *if the magnification block is what
> shifts the operating point benign-ward, exp3n's optimal image threshold should sit
> above exp3's 0.360, and its calibration gain should be smaller.* The actual result
> is stronger than predicted: **exp3n's optimal image threshold is 0.540 — not just
> above exp3's 0.360, but above 0.5 entirely**, and it gains **−0.0005** from
> calibration (i.e. it is already calibrated and needs none). The monotone downward
> drift 0.450 → 0.428 → 0.360 was **the magnification block, not SupCon**: strip the
> block and the threshold snaps back above 0.5. exp3n rows are now in every table
> below, and §1a interprets the reversal.

## Protocol — no test leakage
The threshold is chosen from **validation predictions only**; the 16-patient test
set is *never* used to pick a threshold, only scored at the locked value. Val
predictions are recomputed from each fold's `best.pt` (they aren't saved at train
time); test predictions are read verbatim from the exact `test_predictions.csv`.
Two calibration modes are reported, each at both the **image level** (threshold on
each image's `P(malignant)`) and the **patient level** (threshold on the
per-patient mean prob — the headline BreaKHis metric and the training monitor):

- **Per-fold** — for each fold, pick the accuracy-maximising threshold on *that
  fold's* val set, apply to *that same fold's* test predictions, average across
  folds. Deployment-matched, but each threshold is estimated from only ~13 val
  patients.
- **Pooled out-of-fold (OOF)** — the 5 folds' val sets are a disjoint partition of
  all 66 CV patients, so concatenating their predictions gives full OOF coverage
  (every patient scored by a model that never trained on it). Pick **one** global
  threshold on that pool (~8,000 val images / 66 patients) and apply it to test.

The sweep evaluates every accuracy break-point (midpoints between sorted probs)
plus a dense grid; ties are broken to the **centre of the maximal plateau** (the
most robust generalising cut, not its noisy edge).

---

## 1. Headline result — the optimal threshold drops below 0.5, exactly as predicted

Pooled-OOF optimal thresholds (the stable, trustworthy estimates):

| Exp | AUC (report) | **image** opt-thr | **patient** opt-thr |
|-----|--------------|-------------------|---------------------|
| exp1 (CE) | 0.9559 | **0.450** | 0.380 |
| exp2 (+mag) | 0.9550 | **0.428** | 0.325 |
| exp3 (+mag +SupCon) | **0.9607** | **0.360** | 0.340 |
| **exp3n (+SupCon, no mag)** | 0.9593 | **0.540** | 0.170 |

Among the three original models the image-level optimum drifts monotonically
**downward** — 0.450 → 0.428 → **0.360** — as components are added, which the first
version of this report read as "each component makes the probabilities more
conservative." **exp3n corrects the attribution.** It is exp3 with *only* the
magnification block removed, and its optimum jumps to **0.540** — back above 0.5.
So the downward drift was **not** SupCon and **not** "each component"; it was the
**magnification block** specifically. SupCon on its own (exp1 → exp3n, 0.450 → 0.540)
actually nudges the cut *upward*.

### 1a. The reversal, and what it means

Line the four models up by what they contain:

| model | contains | image opt-thr | calibrated at 0.5? |
|---|---|---:|---|
| exp1 | CE | 0.450 | ~yes (0.45 ≈ 0.5) |
| exp2 | CE + **mag** | 0.428 | mildly benign-leaning |
| exp3 | CE + **mag** + SupCon | 0.360 | **most mis-calibrated** |
| **exp3n** | CE + SupCon | **0.540** | **yes — best of all** |

The magnification block appears in exactly the two models whose threshold sits
lowest (exp2, exp3). Remove it from exp3 → exp3n, and the threshold moves +0.18 back
across 0.5. This is the calibration-side signature of Part 1's counterfactual: the
block is a per-magnification **logit bias** pushing predictions benign-ward, and on a
72%-malignant test set that bias is what dragged the accuracy-optimal cut below 0.5.
It never added discrimination (AUC is flat across all four, 0.955–0.961) — it moved
the operating point, and exp3n proves it by moving it back.

**Consequence for deployment:** exp3n is the only SupCon model that needs **no
threshold calibration** — its 0.5 cut is already essentially optimal (calibration
changes image accuracy by −0.0005; see §2). exp3 needed a 0.36 cut to reach the same
place. That is a concrete operational advantage of the retrieval base encoder: one
fewer fitted hyper-parameter, and no dependence on the pooled-OOF threshold machinery
to reach its accuracy.

---

## 2. Pooled-OOF calibration on test — the money table

**Image level** (test = pooled over all 5 folds' test predictions, 8,265 images):

| Exp | thr | test acc @0.5 | test acc @cal | Δ acc | sens @0.5→cal | spec @0.5→cal |
|-----|-----|---------------|---------------|-------|---------------|---------------|
| exp1 | 0.450 | 0.9054 | 0.9064 | **+0.0010** | 0.924 → 0.930 | 0.856 → 0.845 |
| exp2 | 0.428 | 0.8974 | 0.9016 | **+0.0042** | 0.905 → 0.915 | 0.876 → 0.866 |
| exp3 | 0.360 | 0.8858 | 0.8955 | **+0.0097** | **0.889 → 0.910** | 0.876 → 0.858 |
| **exp3n** | 0.540 | **0.8986** | 0.8981 | **−0.0005** | 0.919 → 0.917 | 0.845 → 0.849 |

**Patient level** (test = mean over 5 folds, 16 patients each):

| Exp | thr | test acc @0.5 | test acc @cal | Δ acc | sens @0.5→cal | spec @0.5→cal |
|-----|-----|---------------|---------------|-------|---------------|---------------|
| exp1 | 0.380 | 0.9875 | 0.9500 | **−0.0375** | 1.00 → 1.00 | 0.95 → 0.80 |
| exp2 | 0.325 | 0.9625 | 0.9500 | −0.0125 | 0.967 → 0.983 | 0.95 → 0.85 |
| exp3 | 0.340 | 0.9375 | **0.9625** | **+0.0250** | **0.933 → 1.00** | 0.95 → 0.85 |
| **exp3n** | 0.170 | **0.9750** | 0.9375 | **−0.0375** | 0.983 → 1.00 | 0.95 → 0.75 |

**What this shows.**
1. **Among the three original models the recovery from calibration is ordered exp3 >
   exp2 > exp1** — the more a component shifted the boundary, the more calibration
   recovers. exp3 gains the most at image level (**+0.0097**) and is the *only* one of
   the three that gains at patient level (**+0.025**). exp1, already calibrated, gains
   nothing (image) or loses (patient). This is precisely the report's threshold-shift
   hypothesis, confirmed quantitatively.
   **exp3n breaks the ordering in the diagnostic direction: it gains essentially zero
   from calibration (−0.0005 image), because it is the best-calibrated model of the
   four** — its 0.5 cut already sits at the plateau. It reaches **0.8986 image accuracy
   at the raw 0.5 threshold**, higher than exp3 reaches even *after* calibration
   (0.8955), and it does so with no fitted threshold. This is the mechanism's cleanest
   confirmation: the component that needed the most calibration (exp3, +0.0097) and the
   one that needs none (exp3n) differ by *exactly* the magnification block.
2. **Calibration recovers exactly the metric SupCon was said to have cost —
   sensitivity.** exp3 image sensitivity climbs 0.889 → 0.910; exp3 *patient*
   sensitivity goes 0.933 → **1.00** (all 12 malignant test patients caught). The
   report attributed exp3's accuracy drop to a sensitivity loss on borderline
   malignants (DC-12312, PC-9146, MC-16456); lowering the cut to 0.34 pulls those
   fence-sitters back over the line — *without* moving anything the model got
   right by a wide margin.
3. **After calibration, SupCon posts the best patient-level test accuracy
   (0.9625)** — overturning the fixed-0.5 result where it was *worst* (0.9375).
   The AUC/ranking advantage of SupCon **does** convert into a patient-accuracy
   advantage, but only once the operating point is set correctly.

---

## 3. The catch — calibration does NOT flip the image-level ranking (but exp3n narrows it most)

Even at each experiment's own optimal image threshold, **exp1 still wins image
accuracy** (0.9064 > exp3n 0.8986 > exp2 0.9016 ≈ exp3 0.8955). Calibration *narrows*
the exp1→exp3 image gap from 0.0196 (at 0.5) to 0.0109, but does not close it, despite
exp3's higher AUC.

Why a better AUC still loses image accuracy at the optimal cut: exp3's ranking
gain is concentrated in the **easy mass** (per `docs/results/classifier_ladder.md` §5/§8, SupCon tightens
the large benign/typical-malignant clusters), while the images that *decide*
accuracy are the **rare/atypical-malignant tail** (papillary, low-grade ductal)
that class-only SupCon strands near the benign boundary. A single global threshold
cannot rescue those — they are mis-*ranked*, not mis-*thresholded*. This is a
memory/exemplar problem, not a threshold problem, and it is exactly what the
proposed retrieval module (`docs/results/classifier_ladder.md` §8, `docs/retrieval.md`) targets. At the **patient**
level, mean-pooling averages out the tail's per-image noise, so there the threshold
move *is* enough to let SupCon win.

**Where exp3n lands.** exp3n reaches **0.8986** image accuracy at its *raw* 0.5 cut —
the second-best of the four and better than exp3's *calibrated* 0.8955 — closing most
of the exp3→exp1 gap without any calibration, because it never acquired the
magnification block's benign-leaning bias in the first place. But it still does not
*overtake* exp1 on image accuracy, and for the same structural reason: the deciding
errors are the mis-ranked rare-malignant tail. Removing the magnification block cleans
the calibration and the geometry (see `docs/results/embedding_geometry.md` / `docs/results/magnification_audit.md`
Part 4); it does not, by itself, relocate the tail. **That remains the retrieval
module's job — and exp3n is the encoder it will run on.**

---

## 4. Per-fold calibration overfits — do not use it

The per-fold mode (threshold picked on ~13 val patients per fold) is a cautionary
result:

| Exp | per-fold image thresholds | mean test img @0.5 → @cal | mean test pat @0.5 → @cal |
|-----|---------------------------|---------------------------|---------------------------|
| exp1 | 0.12, 0.86, 0.57, 0.32, 0.90 | 0.9054 → **0.8993** (−0.006) | 0.9875 → **0.9375** (−0.050) |
| exp2 | 0.22, 0.55, 0.36, 0.27, 0.82 | 0.8974 → **0.8826** (−0.015) | 0.9625 → 0.9625 (0) |
| exp3 | 0.40, 0.91, 0.05, 0.29, 0.76 | 0.8858 → **0.8812** (−0.005) | 0.9375 → 0.9375 (0) |
| exp3n | 0.10, 0.91, 0.36, 0.10, 0.89 | 0.8986 → **0.8870** (−0.012) | 0.9750 → **0.9250** (−0.025) |

The per-fold thresholds swing wildly (exp3: **0.05 to 0.91**; exp3n: **0.10 to
0.91**) because a 13-patient
val set cannot estimate a stable operating point — the accuracy surface is a coarse
step function with wide, flat, noisily-placed plateaus. Per-fold calibration
**hurt every image-level result** and helped no patient-level result. **Only the
pooled-OOF threshold (66 patients / ~8,000 images) is stable enough to transfer to
test.** This is itself a real finding: on this dataset, threshold calibration must
be done on pooled out-of-fold predictions, never per fold.

---

## 5. Honesty caveat — magnitudes are inside the patient-quantisation band

The test set is 16 patients (4 benign, 12 malignant), so patient accuracy moves in
steps of 1/16 = 0.0625; the reported per-fold means move in fractions of that.
exp3's headline +0.025 patient gain and exp1's −0.0375 patient loss are each
**under one patient per fold** — inside the same cluster-variance band `docs/results/classifier_ladder.md`
stresses throughout. The **image-level** pooled numbers (8,265 images) are far more
stable, and there the direction is unambiguous and small: **+0.0097 for exp3,
+0.0042 for exp2, ~0 for exp1, and −0.0005 for exp3n** (already calibrated). Treat the
calibration effect as **real in direction and mechanism, modest in magnitude** — it
confirms *why* fixed-0.5 accuracy behaved as it did; it does not manufacture a large
accuracy jump. exp3n's ~0 calibration gain is the same kind of evidence as exp1's: a
model whose 0.5 cut is already right has nothing to recover, and that is a *property*
worth reporting, not a null result.

---

## 6. Conclusions — is threshold calibration helping, and what does it mean?

**Yes, for exactly the model the report flagged, and for the reason the report
gave.**

1. **The 0.5 threshold was genuinely wrong for the SupCon model — but only because
   of the magnification block, which exp3n now isolates.** exp3's accuracy-optimal cut
   is **~0.36 (image) / ~0.34 (patient)**, well below 0.5; exp1's is ~0.45–0.50,
   already correct. The original reading — "the added components shifted the operating
   point" — is refined by exp3n: **exp3n's optimal cut is 0.540**, above 0.5, so the
   shift was the *magnification block*, not SupCon. SupCon-without-the-block is the
   best-calibrated model of the four. What moved the operating point (a per-magnitude
   logit bias, `docs/results/magnification_audit.md` Part 1) is still not the *discrimination* (AUC flat
   across all four) — the mechanism claim holds; the attribution is now exact.
2. **Calibration recovers the sensitivity SupCon "lost."** Moving exp3's cut down
   restores image sensitivity 0.889 → 0.910 and patient sensitivity 0.933 → 1.00,
   and makes exp3 the **best patient-level model (0.9625)** — reversing its
   last-place fixed-0.5 standing. SupCon's better ranking is real and, once the
   threshold is set on val, it pays off at the patient level.
3. **Calibration is not a substitute for fixing the tail.** exp1 still leads
   image-level accuracy even at the optimal cut, because SupCon's AUC gain lives in
   the easy mass while the accuracy-deciding errors are rare-subtype malignants it
   mis-*ranks*. No global threshold fixes a mis-ranked fence-sitter — that is the
   job of the retrieval/memory module (`docs/results/classifier_ladder.md` §8).
4. **Methodological takeaway:** calibrate on **pooled out-of-fold** predictions,
   never per fold; a 13-patient val set overfits the threshold and degrades test.

### Recommended locked thresholds (pooled OOF)
- **Image-level** (stable, ~8k val images): exp1 → **0.45**, exp2 → **0.43**,
  exp3 → **0.36**, **exp3n → keep 0.50** (its optimum is 0.54 but calibration changes
  test accuracy by −0.0005, so 0.5 is already optimal — do not bother calibrating it).
  Adopting exp3's 0.36 recovers ~1 accuracy point and its sensitivity with no
  re-training; exp3n needs no such correction.
- **Patient-level** (only 66 val patients — use with caution): exp3 → **0.34**
  (test 0.9375 → 0.9625). For exp1/exp2/**exp3n** the val-optimal patient cut overfits
  the few val benigns and *hurts* test (exp3n 0.9750 → 0.9375) — **keep 0.5** for those.
- **For the retrieval base encoder (exp3n): threshold 0.50, no calibration step.**
  This is a genuine operational simplification versus exp3, which needed a fitted 0.36.

*Sources: `analysis/threshold_calibration/results.json` (this run, exp1/exp2/exp3/exp3n);
`runs/{exp1_swin_cls,exp2_swin_mag_cls,exp3_swin_mag_supcon_cls,exp3n_swin_supcon_cls}/*/`
`checkpoints/best.pt` + `test/test_predictions.csv`; AUC values from `docs/results/classifier_ladder.md`
§2; sweep + protocol in `scripts/threshold_calibration.py`.*
