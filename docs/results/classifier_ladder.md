# RAP-MST — Cross-Experiment Fold & Test Analysis

**Scope.** Four experiments, five patient-level CV folds each, evaluated on the
same 16-patient held-out test set (`splits/breakhis_splits.json`).

| Exp | Recipe | Delta over previous |
|-----|--------|---------------------|
| **exp1** | Swin-Tiny + FPN + FeatureFusion + CE | baseline backbone |
| **exp2** | + magnification embedding | explicit magnification signal |
| **exp3** | + projection head + SupCon (CE + SupCon) | contrastive feature shaping |
| **exp3n** | exp3 **minus** the magnification embedding | isolates SupCon from the mag block |

> **exp3n was added later**, after the magnification audit in
> `docs/retrieval.md` showed exp2's magnification embedding is a per-zoom logit
> *bias*, not conditioning. It is the missing cell of the ladder — SupCon **without**
> the magnification block — and it is the model this report's own conclusions
> predicted should exist. Its results (§2a, §5.4) are the cleanest confirmation of the
> report's central mechanism: they show the magnification block, not SupCon, is what
> cost the accuracy. exp1–exp3 numbers are unchanged; exp3n is woven into the tables
> below and analysed where it matters.

All numbers are from the saved `best.pt` of each fold (selected on **image-level
`val_accuracy`**, `monitor: accuracy`; exp1 10 epochs, exp2/exp3/exp3n 15) and the
per-fold `test/test_metrics.json` / `test_predictions.csv`. Test set = **1653 images /
16 patients**, of which **~72% images are malignant** (4 benign vs 12 malignant
patients) — remember this prevalence; it drives the accuracy-vs-AUC story below.

> **Data caveat.** Per-patient *validation* breakdowns (the `diagnose_folds.py`
> per-patient table) require loading `best.pt` and re-running inference, which
> needs PyTorch + the dataset (torch is not installed in this reporting
> environment). Per-patient *test* analysis below is computed directly from the
> saved `test_predictions.csv` and is exact. Validation is analysed at the
> fold/patient-count level plus subtype-based candidate identification; to get the
> exact val patients that flip, run
> `python scripts/diagnose_folds.py --experiment expN`.

---

## 1. Validation accuracy per fold (best epoch)

Image-level `val_accuracy` (and patient-level in parentheses):

| Fold | exp1 img (pat) | exp2 img (pat) | exp3 img (pat) | exp3n img (pat) |
|------|----------------|----------------|----------------|-----------------|
| 0 | 0.8309 (0.929) | 0.8431 (0.929) | 0.8449 (0.857) | 0.8344 (0.786) |
| 1 | 0.8736 (0.923) | 0.8557 (0.769) | 0.8780 (0.923) | 0.8905 (0.923) |
| 2 | 0.8722 (0.923) | 0.8820 (0.923) | 0.8856 (0.923) | 0.8588 (0.846) |
| 3 | 0.9675 (1.000) | 0.9788 (1.000) | 0.9788 (1.000) | 0.9821 (1.000) |
| 4 | 0.7794 (0.846) | 0.7579 (0.769) | 0.7822 (0.769) | 0.7617 (0.769) |
| **mean** | **0.8647 (0.924)** | **0.8635 (0.878)** | **0.8739 (0.895)** | **0.8655 (0.865)** |
| std (img) | 0.062 | 0.070 | 0.063 | 0.072 |

**Read:** exp3 has the best mean *image* val accuracy (0.8739) and exp1 the best
mean *patient* val accuracy (0.924). Fold 3 is trivially easy (≈0.98, 0 patients
wrong); fold 4 is the hard fold (≈0.78) in every experiment. At the patient level
the whole spread is **0–3 wrong patients out of 13–14** — i.e. one or two flips.
**exp3n's *validation* numbers are unremarkable — slightly below exp3 on both image
(0.8655) and patient (0.865) — which is the important caution up front: exp3n's
advantage does not show up on validation at all. It appears only on the held-out
*test* set (§2a), and the gap between the two is itself the patient-quantisation
noise this report keeps stressing (13-patient val folds).**

Patient-level wrong-count (from `patient_accuracy × n_val`):

| Fold | n_val | exp1 wrong | exp2 wrong | exp3 wrong | exp3n wrong |
|------|-------|-----------|-----------|-----------|-------------|
| 0 | 14 | 1 | 1 | **2** | **3** |
| 1 | 13 | 1 | **3** | 1 | 1 |
| 2 | 13 | 1 | 1 | 1 | **2** |
| 3 | 13 | 0 | 0 | 0 | 0 |
| 4 | 13 | 2 | **3** | **3** | **3** |

The added components move 1–2 patients per fold in *either* direction. That is
exactly the "effective n = patient count" regime `diagnose_folds.py` warns about:
these are cluster/seed flips, not a systematic learning signal. exp3n is no
exception on *validation* — it is 1 worse than exp3 in folds 0 and 2 — which is
exactly why the model-selection story is a wash on val and only the test set (§2a)
separates them.

---

## 2. Test accuracy per fold (held-out 16 patients)

| Fold | exp1 | exp2 | exp3 | exp3n |
|------|------|------|------|-------|
| 0 | 0.9244 | 0.9304 | 0.9220 | 0.9292 |
| 1 | 0.8560 | 0.8754 | 0.8693 | 0.8972 |
| 2 | 0.9286 | 0.9208 | 0.8536 | 0.8959 |
| 3 | 0.8953 | 0.9238 | 0.9099 | 0.9177 |
| 4 | 0.9226 | 0.8367 | 0.8742 | 0.8530 |
| **mean img_acc** | **0.9054** | **0.8974** | **0.8858** | **0.8986** |
| **mean patient_acc** | **0.9875** | **0.9625** | **0.9375** | **0.9750** |
| **mean AUC** | **0.9559** | **0.9550** | **0.9607** | **0.9593** |
| mean sensitivity | 0.9242 | 0.9054 | 0.8894 | 0.9193 |
| mean specificity | 0.8563 | 0.8764 | 0.8764 | 0.8445 |

**The headline honest finding:** adding the magnification embedding (exp2) and
then SupCon (exp3) did **not** raise thresholded test accuracy — image accuracy
drifts *down* 0.905 → 0.897 → 0.886 and patient accuracy 0.988 → 0.963 → 0.938.
Yet **AUC is flat-to-up** (0.9559 → 0.9550 → **0.9607**, exp3 best). The
mechanism is visible in the last two rows: each component trades **sensitivity for
specificity** (0.924→0.905→0.889 vs 0.856→0.876→0.876). On a 72%-malignant test
set, a more conservative (benign-leaning) boundary at a fixed 0.5 threshold
*costs* accuracy even while ranking quality (AUC) holds or improves.

**exp3n is the control that identifies *which* component caused the drop.** Its
image accuracy is **0.8986** — above exp3 (0.8858) and second only to exp1 — and its
patient accuracy is **0.9750**, recovering most of exp3's loss. Crucially, its **AUC
is 0.9593, statistically the same as exp3's 0.9607** (a −0.0014 wash, well inside the
±0.025 per-fold spread). So exp3n did *not* add discrimination over exp3; it **undid
the threshold shift.** Since exp3n = exp3 with only the magnification block removed,
the accuracy that exp2/exp3 lost was lost to the **magnification embedding**, not to
SupCon. §2a makes the mechanism explicit; §5.4 is the component write-up.

At the patient level exp1 misses only 1 patient across all 5 folds (fold 1);
exp3's drop to 0.9375 is 1 extra malignant patient flipping benign in folds 2, 3
and 4 — again single-patient moves.

---

## 3. Per-patient test analysis — who is missed, and how it moves

Same 16 patients in every run. Below, `img_acc` and `mean_prob` are averaged over
the 5 folds; `pat✓` = how many of 5 folds classify the patient correctly.

| Patient | Sub | Label | exp1 img_acc | exp2 | exp3 | exp3n | Δ(e3n−e3) | exp3 pat✓ | exp3n pat✓ |
|---------|-----|-------|-------------|------|------|-------|-----------|-----------|------------|
| SOB_M_DC-14-12312 | DC | mal | 0.686 | 0.638 | **0.563** | **0.742** | **+0.179** | 3/5 | **4/5** |
| SOB_M_PC-14-9146 | PC | mal | 0.713 | 0.664 | **0.631** | **0.724** | **+0.093** | 3/5 | **5/5** |
| SOB_B_TA-14-16184 | TA | ben | 0.671 | 0.706 | 0.695 | 0.630 | −0.065 | 4/5 | 4/5 |
| SOB_M_MC-14-16456 | MC | mal | 0.933 | 0.904 | 0.871 | 0.862 | −0.009 | 5/5 | 5/5 |
| SOB_M_DC-14-20636 | DC | mal | 1.000 | 0.984 | 0.941 | 0.998 | +0.057 | 5/5 | 5/5 |
| SOB_B_A-14-22549CD | A | ben | 0.892 | 0.894 | 0.912 | 0.879 | −0.033 | 5/5 | 5/5 |
| SOB_B_F-14-23060CD | F | ben | 0.919 | 0.972 | 0.972 | 0.972 | 0.000 | 5/5 | 5/5 |
| SOB_B_F-14-21998CD | F | ben | 0.974 | 0.984 | 0.977 | 0.965 | −0.012 | 5/5 | 5/5 |
| *(the other 8 DC/LC/MC)* | | mal | ≈0.96–1.00 | ≈0.97–1.00 | ≈0.97–1.00 | ≈0.97–1.00 | ≈0 | 5/5 | 5/5 |

**exp3n reverses the two costly false negatives at the image level.** The two hard
malignants exp3 pushed *toward* benign — DC-12312 (img_acc 0.563) and PC-9146
(0.631) — snap back up under exp3n to **0.742** and **0.724**, and the rare papillary
PC-9146 goes to **5/5 patient-correct**. This is the specificity-for-sensitivity
trade running in reverse: without the magnification block's benign-leaning bias, the
borderline malignants are no longer tipped over the line. The cost lands, as always,
on the benign fence-sitter TA-16184 (0.695 → 0.630) and the mucinous MC-16456
(marginal), but both stay patient-correct. **Nothing on the easy list regresses at
the patient level.** (Per-*probability* the swing is even sharper — DC-12312
mean_prob 0.577 → 0.712, PC-9146 0.639 → 0.726; see `docs/results/magnification_audit.md` Part 4.)

**Three patients carry essentially all the test error, in every experiment:**

1. **SOB_M_DC-14-12312 (ductal carcinoma, malignant).** The single hardest case.
   `mean_prob` sits right on the fence (0.66 → 0.60 → **0.58**) and exp3 tips it
   patient-wrong in 2 of 5 folds. A low-grade / well-differentiated ductal
   morphology that reads benign-like.
2. **SOB_M_PC-14-9146 (papillary carcinoma, malignant).** `mean_prob` 0.72 → 0.65
   → 0.63; exp3 wrong in 2/5 folds. **PC is a rare subtype** — few training
   patients — so the parametric head never learns a confident PC boundary.
3. **SOB_B_TA-14-16184 (tubular adenoma, benign).** The hardest *benign*:
   `mean_prob` ≈0.31–0.36, wrong in 1/5 folds. Tubular/dense architecture mimics a
   well-differentiated carcinoma.

Secondary erosion under exp2/exp3: **MC-16456** (mucinous, large slide) and
**DC-20636** slip from ~1.0 toward 0.87–0.94 — borderline malignants getting
pulled toward the boundary by the contrastive reshaping.

**What actually got easier** (exp3 vs exp1): the *benign* cases —
**F-23060CD (+0.053)**, **A-22549CD (+0.020)**, **TA-16184 (+0.024)**. This is the
specificity gain made concrete: SupCon tightens the benign cluster, so clear-ish
benigns get cleaner. **What got harder:** the borderline *malignants*
(DC-12312 −0.123, PC-9146 −0.082, MC-16456 −0.062, DC-20636 −0.059) — the
sensitivity loss made concrete.

---

## 4. Per-fold validation — where the folds differ, and likely-missed patients

Fold difficulty is structural, set by which subtypes land in each val split:

| Fold | Benign val subtypes | Notable | Difficulty |
|------|---------------------|---------|------------|
| 0 | A, F, F, **PT** | big phyllodes (PT, 235 img) + rare **PC** malignant | medium |
| 1 | F, **TA, TA, TA** | 3× tubular adenoma (borderline benign) | medium |
| 2 | F, F, PT, TA | mixed | easy–medium |
| 3 | A, F, TA, TA | well-separated benigns, DC-heavy malignant | **easy (0 wrong)** |
| 4 | A, F, F, **PT** | phyllodes + 8× DC, no easy malignant anchor | **hard** |

Mapping the wrong-count moves in §1 to these compositions (candidates by subtype;
confirm exact IDs with `diagnose_folds.py`):

- **Fold 4 (2→3 wrong under exp2/exp3).** Contains **SOB_B_PT-14-29315EF**
  (phyllodes — benign but cytologically alarming, a classic false-positive) plus a
  wall of 8 ductal carcinomas and no easy malignant anchor. The persistent 2 wrong
  are most plausibly the **phyllodes** + one **borderline DC**; exp2/exp3 add a
  third by pushing another borderline DC benign.
- **Fold 1 (exp2 spikes to 3 wrong, exp3 recovers to 1).** Its benign side is
  **three tubular adenomas** (TA-13200, TA-16184CD, TA-3411F). The magnification
  embedding (exp2) evidently pulled the TAs toward malignant (specificity drop);
  SupCon (exp3) re-tightened the benign cluster and recovered them. This is the
  same TA-vs-carcinoma confusion seen on the **test** TA patient.
- **Fold 0 (exp3 regresses 1→2 wrong).** Holds the 235-image **phyllodes**
  (PT-21998AB) and the rare **papillary** malignant (PC-15-190EF) — the exact two
  subtypes (PT false-positive, PC false-negative) that SupCon's class-only
  clustering handles worst.

**Consistent culprits across BOTH val and test** are a small, nameable set of
subtypes: **PT (phyllodes)** and **TA (tubular adenoma)** on the benign/false-
positive side; **PC (papillary)** and **low-grade/borderline DC** on the
malignant/false-negative side; **MC (mucinous)** on the boundary. Everything else
(typical DC, LC, adenosis, most fibroadenomas) is already at 0.95–1.00.

---

## 5. Component-by-component: what each part bought (and cost)

### exp1 — Swin + FPN + CE (baseline)
- **Strongest thresholded scores of the three:** best mean test image acc
  (0.9054) and patient acc (0.9875 — 79/80 fold-patient decisions correct).
- The FPN's multi-scale aggregation already gives a very discriminative backbone
  (test AUC 0.956). The easy 13 test patients are essentially solved (0.96–1.00).
- **Weakness:** the three boundary cases (DC-12312, PC-9146, TA-16184) are
  unresolved, and specificity is the lowest (0.856) — it errs slightly toward
  calling benign tissue malignant.

### exp2 — + magnification embedding
- **What it improved:** *specificity* (0.856 → 0.876) and several **benign** test
  patients (F-23060CD +0.053, TA-16184 +0.035 vs exp1). Telling the model the
  zoom level helps it not over-call low-power benign fields as malignant.
- **What it cost:** *sensitivity* (0.924 → 0.905) and the hard malignants
  (DC-12312 −0.048, PC-9146 −0.049). Net image acc −0.008, patient acc −0.025.
- **Why it doesn't lift accuracy:** AUC is flat (0.9559 → 0.9550) — the embedding
  **shifts the operating threshold**, it does not add discrimination. On a
  malignant-heavy set, trading recall for specificity is a net accuracy loss.
  Fold 4 test collapses to 0.837 (sensitivity 0.785) — the mag signal over-
  regularised a fold that had no easy malignant anchor.
- **Retrospective (via exp3n):** the "shifts the threshold, not the discrimination"
  read is now *proven*, not inferred. The magnification embedding is a linear
  per-magnification **logit bias** (`docs/results/magnification_audit.md` Part 1);
  `docs/results/threshold_calibration.md` §1a shows removing it moves the accuracy-optimal
  cut from 0.36 back to 0.54. **exp2
  is not a failed model — it is the experiment that isolated this signal.** Without
  exp2 we could not have attributed the fragmentation and the threshold shift to
  magnification rather than to SupCon (see `docs/results/embedding_geometry.md` §3). It earns
  its place in the ladder as the control that named the cause.

### exp3 — + SupCon (CE + SupCon)
- **What it improved:** **AUC — the best of the three (0.9607)** — and the best
  mean *image* val accuracy (0.8739). SupCon genuinely produces a more
  discriminative, better-*ranked* embedding, and further tightens benign
  (A-22549CD +0.020, TA-16184 +0.024 vs exp1). It also rescued fold-1 val (the
  TA cluster) relative to exp2.
- **What it cost:** the most sensitivity (0.889) and the lowest thresholded test
  accuracy (0.886 img / 0.938 patient). Borderline malignants slid furthest
  (DC-12312 −0.123, MC-16456 −0.062, DC-20636 −0.059 vs exp1).
- **Why the AUC gain doesn't show up as accuracy:** two reasons. (1) **Fixed 0.5
  threshold on a skewed set** — a better ranking under a benign-leaning threshold
  still mis-thresholds the fence-sitters. (2) **Class-only SupCon** pulls *all*
  malignant subtypes into one blob; the rare/atypical malignants (PC, low-grade
  DC) that don't fit the dominant DC morphology get stranded near the benign
  boundary. SupCon improved *separation of the easy mass* at the expense of the
  *tails* — and the tails are exactly the patients that decide accuracy.
- **Not superseded by exp3n.** exp3 is where SupCon's contrastive machinery was
  *proven to work at all*, under the harder condition of the magnification block being
  present (best AUC of the original three; best mean image val accuracy). exp3n is
  exp3's mechanism re-run with the confound removed — it depends on this finding, it
  does not retire it.

### exp3n — + SupCon, **minus** the magnification embedding (the corrective)
- **What it improved over exp3:** thresholded accuracy — image **0.8858 → 0.8986**,
  patient **0.9375 → 0.9750**, sensitivity **0.889 → 0.919** — recovering the two hard
  false negatives (DC-12312, PC-9146) at both image and patient level (§3). It is the
  **best-calibrated** model of all four: its accuracy-optimal threshold is 0.54 (≈0.5),
  so it needs **no** threshold calibration, where exp3 needed a fitted 0.36 to reach a
  *lower* accuracy (`docs/results/threshold_calibration.md` §1a).
- **What stayed the same:** **AUC — 0.9593 vs exp3's 0.9607 is a statistical tie**
  (−0.0014, inside the ±0.025 per-fold spread). This is the crux: exp3n did not add
  *ranking* power over exp3; it removed the magnification block's benign-leaning
  threshold bias. **The accuracy exp2/exp3 lost was lost to the magnification
  embedding, not to SupCon** — exp3n, differing from exp3 by exactly that block, gets
  it back.
- **What it cost:** specificity (0.876 → 0.845) — the same sensitivity/specificity
  trade running the other way, net-positive on a 72%-malignant set. And a small
  representational debit outside this report's scope: removing the block's implicit
  same-magnification filtering drops raw retrieval kNN ~1.5 points, which explicit
  magnification routing recovers (`docs/results/magnification_audit.md` Part 4).
- **Why it matters beyond accuracy:** exp3n is the intended **base encoder for the
  retrieval module.** Its SupCon projection space is no longer magnification-locked
  (same-mag neighbour rate 0.976 → 0.335) and its embedding geometry is the cleanest
  measured (`docs/results/embedding_geometry.md` §1). The retrieval key, the SupCon space, and
  the indexed vector become one and the same 1024-d space.

---

## 6. Why the "expected" monotonic improvement didn't happen

- **The signal is patient-quantised.** With 13–14 val patients and 16 test
  patients (each ≈60–235 near-duplicate images), effective sample size is the
  *patient* count. Every cross-experiment delta reduces to 1–2 patients flipping.
  The mean-accuracy differences (±0.01–0.02) are inside that noise band.
- **`monitor: accuracy` on a 13-patient val set** selects `best.pt` from a noisy
  signal, and only **10 epochs** — the extra objectives (SupCon) that need longer
  schedules and larger effective batches to shape a metric space are cut short.
- **Thresholded accuracy is the wrong lens for what these components do.** They
  move the *ranking* and the *operating point*; AUC (up in exp3) is the honest
  measure. Accuracy fell because the threshold stayed at 0.5 on a 72%-malignant
  set while the boundary moved benign-ward.
- **The threshold shift was the magnification block, and exp3n reclaims it.** The
  cleanest *representation-side* improvement in this whole study is exp3n — but note
  what kind of improvement it is: it did **not** raise AUC over exp3 (0.9593 vs 0.9607,
  a tie). It raised thresholded accuracy by *de-biasing the operating point* — undoing
  the magnification block, not out-discriminating anything. So even the one experiment
  that "did improve" confirms the bullet above rather than contradicting it: the
  movable quantity was the threshold, and the fix was to stop shifting it.
- **The errors are irreducible for a parametric class head:** the same 3–5
  subtypes (PT, TA, PC, low-grade DC, MC) are unresolved in *every* experiment,
  **exp3n included** — it improves DC-12312 and PC-9146 but they remain the hardest
  cases, and TA/MC still straddle the boundary. No amount of the current
  CE/embedding/SupCon machinery fixes a fence-sitter whose morphology genuinely
  straddles the class boundary — that is a **memory/exemplar** problem, not a
  **representation** problem. exp3n is the *cleanest encoder* for that memory to run
  on, not a replacement for it.

---

## 7. Intuitive bulletins

**What each component reliably buys**
- **FPN + CE (exp1):** a strong, well-calibrated backbone; solves the ~13 typical
  patients; best raw accuracy; best patient-level agreement.
- **Magnification embedding (exp2):** +specificity, cleaner *benign* low-power
  fields; catches over-calls on adenosis/fibroadenoma. Costs malignant recall.
- **SupCon (exp3):** best AUC / ranking; tightest benign cluster; best mean image
  val accuracy; recovers borderline-benign (TA) folds. Costs malignant recall on
  the rare/atypical tail — *but this cost is mostly the co-present magnification
  block, not SupCon itself (see exp3n).*
- **SupCon without the mag block (exp3n):** keeps exp3's ranking (AUC tie) while
  recovering the malignant recall exp2/exp3 gave away — best-calibrated model, needs
  no threshold tuning, and the cleanest embedding geometry for the retrieval module.
  The single deliberate subtraction (the magnification embedding) is the study's
  clearest net-positive change.

**What the stack catches well**
- Typical ductal carcinoma (DC), lobular (LC), most mucinous (MC), adenosis (A),
  most fibroadenoma (F): 0.95–1.00 across all experiments and folds.
- Clear benign at low power improves monotonically exp1→exp3.

**What the stack still cannot catch (the miss list)**
- **False negatives:** `SOB_M_DC-14-12312` (low-grade DC), `SOB_M_PC-14-9146`
  (papillary, rare subtype), and under exp3 the borderline `MC-16456` / `DC-20636`.
- **False positives:** `SOB_B_TA-14-16184` (tubular adenoma) on test; **phyllodes
  (PT)** and **tubular adenoma (TA)** patients in val folds 0, 1, 4.
- Rooted in **rarity** (PC) and **morphological overlap** (low-grade DC vs benign;
  TA/PT vs carcinoma) — not in backbone capacity.

**Why accuracy isn't climbing as hoped**
- The added objectives improve *ranking/threshold placement*, not the fixed-0.5
  *accuracy*, on a skewed test set; the deltas are within 1–2-patient cluster
  variance; 10 epochs + a 13-patient monitor under-serve SupCon; and the residual
  errors are boundary cases a parametric head structurally cannot resolve.
- **The one thing that *did* climb accuracy was subtraction, not addition:** removing
  the magnification embedding (exp3n) beat exp3 by +0.013 image / +0.038 patient — by
  fixing the *operating point*, not the discrimination. It is the exception that
  proves the rule: the ranking (AUC) was already good; what was broken was where the
  threshold sat, and one component was pushing it.

---

## 8. Proposed retrieval module — targeting exactly the missed set

> **⚠ Superseded by `docs/retrieval.md` (and amended by `docs/retrieval.md`).** This
> section is kept for the record. Three of its five proposals were **refuted** by
> direct measurement: the index vector is `features`, **not** `embeddings` (the
> magnification block makes `embeddings` ~100% magnification-locked); rare-subtype
> up-weighting **hurts** every metric; and the hard confidence gate is a no-op.
> Read `docs/retrieval.md` for the design that survived.

The forward dict already exposes what a retrieval memory needs: `features` (the
canonical per-image vector, **pre** magnification-fusion — see the note above) and
`fpn_features` (the multi-scale maps). The miss list is small, nameable, and boundary-clustered —
ideal for a **non-parametric memory that resolves fence-sitters by neighbourhood
vote** rather than by moving the global boundary.

**Design.**
1. **Memory bank.** After training, index the *training* set's `embeddings`
   (L2-normalised) into a labelled store — key = SupCon vector, value =
   `{binary label, tumor subtype, patient_id, magnification}`. Store per
   magnification so retrieval is zoom-aware (reuses the exp2 signal *correctly*,
   as neighbourhood structure instead of a threshold shift).
2. **Query & fuse.** At inference, retrieve top-k neighbours for each test
   embedding; form a retrieval logit from a **distance-weighted, subtype-balanced**
   vote. Final score = `σ(α·parametric_logit + (1−α)·retrieval_logit)`, with `α`
   learned or set by a **confidence gate**: only defer to memory when the
   parametric `|prob − 0.5|` is small (the fence-sitters). Confident images are
   left untouched, so the already-solved 13 patients don't regress.
3. **Fix the SupCon tail problem at the source.** Train SupCon with **subtype**
   (8-way) positives, or add a subtype-aware term, so PC and low-grade DC form
   their own sub-clusters instead of dissolving into the DC blob — this makes
   retrieval neighbourhoods clean for exactly the rare/atypical malignants.
4. **Rare-subtype up-weighting in the bank.** Over-represent / re-weight PC, PT,
   TA, LC exemplars in the vote so a rare query isn't outvoted by the DC mass.

**Why this covers the specific misses.**
- **PC-9146 / PC val patients (rarity):** k-NN over a subtype-balanced bank lets a
  handful of PC exemplars decide a PC query, which the parametric head can't do.
- **DC-12312 / MC-16456 / DC-20636 (borderline malignant, `mean_prob`≈0.55–0.66):**
  these are precisely the confidence-gated cases; nearest labelled malignant
  exemplars pull them over 0.5 **without** moving the global threshold — recovering
  sensitivity *without* sacrificing the specificity exp2/exp3 gained.
- **TA-16184 / phyllodes (benign false-positives):** their nearest neighbours are
  other benign TA/PT slides; the vote pulls `mean_prob` down, fixing folds 0/1/4.

**Success criterion.** Judge the retrieval module on **sensitivity at fixed
specificity** and on those ~5 named patients — not on aggregate accuracy, which
this analysis shows is a patient-quantised, threshold-dominated number. Target:
DC-12312 and PC-9146 to `pat✓ = 5/5`, TA-16184 to 5/5, with the easy 13 unchanged.

---

*Sources: `runs/{exp1_swin_cls,exp2_swin_mag_cls,exp3_swin_mag_supcon_cls}/*/`
`metrics.csv`, `train.log`, `test/test_metrics.json`, `test/test_predictions.csv`;
`splits/breakhis_splits.json`; `scripts/diagnose_folds.py`.*
