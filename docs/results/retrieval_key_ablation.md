# What should the Retrieval Memory key on? — the measured answer

**Question asked.** `docs/retrieval.md` Part VII concluded that the memory module does
not help exp3n because its key — the pooled 1024-d `features` vector — is the
classifier's own input, and listed the untested alternatives: the SupCon
`projections`, the spatial `fpn_features`, and an encoder whose key is *not* the
classifier's input. This report tests all of them.

**Answer in one paragraph.** The key is not the problem. Across **43
configurations** — every forward-dict vector, four spatial poolings of the FPN
pyramid, every individual pyramid level, composites, and eight fitted key-space
transforms including whitening and deleting the classifier's exact decision
direction — the image-level retrieval AUC moves within a **0.017 band
(0.858–0.876)** and **not one configuration reaches the parametric head's 0.8855**.
The property that actually predicts whether retrieval can contribute — how
*decorrelated* its errors are from the head's — is essentially constant across
every key drawn from the same encoder, and **nearly doubles (×1.9) the moment the
key comes from a different encoder**. The ceiling is **encoder sharing**, not pooling, not
dimensionality, and not geometry. Recommendation: **keep D1 (`features`,
`key_transform: none`) unchanged**, close the key question, and spend the next
increment on encoder diversity (§7), for which exp1's checkpoints already exist at
zero GPU cost.

---

## 1. What was built

The key is now a configurable, first-class part of the module rather than a dict
lookup, so this ablation is a config line rather than a code change.

| Piece | File | What it adds |
|---|---|---|
| Key spec language | [rap_mst/retrieval/keys.py](../../rap_mst/retrieval/keys.py) | `features` · `embeddings` · `projections` · `fpn.{gap,max,std,gem}` (+ `@level`, `.ln`) · `a+b` composition |
| Fitted key transform | [rap_mst/retrieval/transform.py](../../rap_mst/retrieval/transform.py) | `none` · `center` · `pca_drop:n` · `whiten:n` · `drop_dirs` (the classifier's decision direction) |
| Bank support | [rap_mst/retrieval/bank.py](../../rap_mst/retrieval/bank.py) | `key_transform` fitted on the bank's **train rows only**, stored in the `.npz`, applied identically to every query; format v2, v1 banks still load |
| Config | [config/config.yaml:136](../../config/config.yaml#L136) | `retrieval.key`, `retrieval.key_transform` |
| Study driver | [scripts/retrieval_key_ablation.py](../../scripts/retrieval_key_ablation.py) | two-stage sweep (§2) |

Every consumer reads the key through that one switch — `build_memory_bank.py`
(what the bank stores), `evaluator.py` (what Stage C queries with), `builder.py`
(load-time compatibility) — so a winning configuration transfers to production by
editing the two config values and rebuilding, with no code change. Mixing a bank
and a config that disagree on either value **raises**: the transform is fitted
*into* the stored keys, so a silent mismatch would be invisible in the metrics.

```powershell
python scripts/retrieval_key_ablation.py --stage cache                    # ~15 min GPU
python scripts/retrieval_key_ablation.py --stage eval --cross-encoder exp1 # ~30 min
```

## 2. Protocol

**Two stages, so 43 configurations cost one forward pass per fold.** Stage `cache`
encodes each fold's bank split (its TRAIN patients, eval transforms, in order) and
its validation split once with the frozen exp3n `best.pt`, storing every *pooled
component* of the forward dict — per-pyramid-level GAP / max / std / GeM, the
SupCon projections, the logits, and the linear head's decision direction. Stage
`eval` re-keys those cached vectors offline.

**The classes are the production ones.** `MemoryBank`, `RetrievalMemory` (two
independent rankings over one store), the per-patient cap, the softmax vote and
`FusionGate` are the same objects Stages B1/B2 use, fitted with the same protocol,
seed, epochs and learning rate.

**Held fixed.** Route, `k`, per-patient cap, temperature, the two-level ranking
(D3–D7) and the gate architecture. **Varied: the key only.**

**Leakage.** Fold *k*'s bank holds only fold *k*'s train patients — `assert_disjoint`
re-runs for every configuration — queries block their own patient, and the key
transform is fitted on bank rows alone, so it never sees a validation image.

**Two fused numbers, and only one of them is honest.** `p_final` is exactly what
Stage B2 produces: one gate fitted on all pooled OOF rows. But it is fitted *and*
scored on those rows. `p_final_loso` refits the gate on four folds and applies it
to the fifth, so no row is scored by a gate that saw it. The gap between them is
the gate's in-sample optimism: **+0.0039 AUC on average, up to +0.0217**. Every
comparison below uses LOSO.

> **Harness validation.** The D1 baseline reproduces `gate_fit.json` exactly —
> `p_img` AUC 0.8678 / acc 0.8496, `p_final` AUC 0.8862, mean gate weights
> (0.823, 0.147, 0.031). The offline path is the production path.

**Reference (identical for every row): `p_param`** — image acc **0.8654**, image
AUC **0.8855**, patient acc **0.8636**, patient AUC **0.9598**.

---

## 3. Result — the key does not matter

Pooled out-of-fold validation, 6,256 images / 66 patients. `dAUC`/`dAcc` are
LOSO-fused minus `p_param`. `corr` is the correlation between `p_img` and `p_param`;
`awrong` is the AUC of `p_img` restricted to the rows the head gets wrong (0.5 =
uninformative, **near 0 = wrong in exactly the same places**).

| enc | key / transform | dim | eff. rank | p_img AUC | LOSO AUC | dAUC | dAcc | corr | awrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3n | `features` / `pca_drop:20` | 1024 | 33.3 | 0.8593 | 0.8897 | +0.0042 | −0.0013 | 0.899 | 0.080 |
| **exp1** | **`fpn.std` / none** | 1024 | 2.9 | **0.8723** | 0.8883 | +0.0028 | −0.0002 | 0.906 | **0.128** |
| 3n | `fpn.gap@0` / `whiten:128` | 128 | 106.9 | 0.8647 | 0.8874 | +0.0019 | −0.0065 | 0.903 | 0.069 |
| **exp1** | **`features` / none** | 1024 | 1.3 | 0.8718 | 0.8872 | +0.0017 | **+0.0005** | 0.908 | **0.133** |
| 3n | `features` / `whiten:512` | 512 | 364.1 | 0.8689 | 0.8863 | +0.0008 | −0.0038 | 0.926 | 0.059 |
| 3n | `features+fpn.std` / none | 2048 | 1.4 | 0.8686 | 0.8848 | −0.0007 | −0.0041 | 0.946 | 0.057 |
| 3n | `fpn.std` / none | 1024 | 2.7 | 0.8699 | 0.8846 | −0.0009 | −0.0017 | 0.937 | 0.057 |
| 3n | `fpn.gem` / none | 1024 | 1.3 | 0.8675 | 0.8841 | −0.0014 | −0.0024 | 0.946 | 0.054 |
| 3n | `fpn.max` / none | 1024 | 1.6 | 0.8682 | 0.8838 | −0.0017 | −0.0025 | 0.941 | 0.055 |
| 3n | `fpn.gap.ln` / none | 1024 | 1.2 | 0.8667 | 0.8834 | −0.0021 | −0.0048 | 0.948 | 0.055 |
| 3n | **`features` / none  (D1)** | 1024 | 1.2 | 0.8678 | 0.8833 | −0.0022 | −0.0027 | 0.948 | 0.056 |
| 3n | `projections` / none | 128 | 1.1 | 0.8660 | 0.8832 | −0.0023 | −0.0009 | 0.944 | 0.060 |
| 3n | `fpn.gap@0` / none | 256 | 1.2 | 0.8655 | 0.8831 | −0.0024 | −0.0048 | 0.945 | 0.051 |
| 3n | `features` / `center,drop_dirs` | 1024 | 1.5 | 0.8657 | 0.8829 | −0.0026 | −0.0033 | 0.941 | 0.064 |
| 3n | `features` / `pca_drop:100` | 1024 | 117.5 | **0.8758** | 0.8706 | −0.0149 | −0.0056 | 0.906 | 0.084 |
| 3n | `features` / `pca_drop:200` | 1024 | 168.2 | 0.8691 | 0.8613 | −0.0242 | −0.0080 | 0.902 | 0.078 |

*(Full 43-row table with per-level, per-transform and slide-level columns:
`analysis/retrieval_keys/exp3n/key_ablation.json` and `key_ablation.log`.)*

Four things read straight off this table.

**3.1 Nothing clears the head.** `p_img` AUC ranges 0.8584–0.8758 across all 39
same-encoder keys; `p_param` is 0.8855. **0 / 39** beat it. That is the same result
`docs/retrieval.md` Part VII §4 reported for the single D1 key, now shown to hold for
every key the encoder can produce.

**3.2 The fused gain is largely a gate artefact.** Under the Stage B2 protocol
**26 / 39** same-encoder configurations "beat" `p_param` on AUC. Under LOSO only
**8 / 39**, by at most +0.0042 — and **0 / 39 beat it on image accuracy** (best
−0.0009). Most of the in-sample column is the gate fitting the rows it is scored on.

**3.3 Nothing is significant.** Patient-clustered bootstrap, 2,000 resamples of the
66 CV patients:

| config | Δ AUC (LOSO − param) | 95% CI | P(Δ>0) | Δ acc | 95% CI |
|---|---:|---|---:|---:|---|
| `features` / none (D1) | −0.0023 | [−0.0082, +0.0031] | 0.22 | −0.0027 | [−0.0089, +0.0028] |
| `features` / `pca_drop:20` | +0.0036 | [−0.0138, +0.0231] | 0.64 | −0.0013 | [−0.0072, +0.0047] |
| [exp1] `features` / none | +0.0016 | [−0.0071, +0.0099] | 0.65 | +0.0005 | [−0.0100, +0.0113] |
| [exp1] `fpn.std` / none | +0.0025 | [−0.0064, +0.0105] | 0.73 | −0.0000 | [−0.0115, +0.0115] |

Every interval straddles zero. **No configuration in this study is a defensible
improvement**, including the ones at the top of the ranking.

**3.4 The top of the ranking is selection noise.** 43 configurations were scored on
the surface the gate is fitted on. The best-ranked row (`pca_drop:20`, +0.0042
AUC) sits inside a bootstrap interval eleven times its own width and is *negative*
on accuracy. Treat the ranking as a null result with a spread, not as a leaderboard.

---

## 4. Why geometry was the wrong suspect

`docs/retrieval.md` Part VII §3 measured the D1 key at effective rank **1.19 / 1024**
and named the collapse as the reason retrieval has nothing to read. That
measurement is correct; the causal reading of it is not.

The transforms move effective rank across a **338× range** — 1.08 (`projections`)
to 364.1 (`features`/`whiten:512`) — while `p_img` AUC moves within 0.017. Across
the 39 same-encoder configurations:

```
corr(log effective rank, p_img AUC)   = +0.007        (i.e. none)
corr(log effective rank, LOSO ΔAUC)   = −0.162
```

Whitening does exactly what it promises geometrically — and retrieval quality does
not move:

| `features` / transform | eff. rank | random-pair cos | mean top-1 sim | ‖mean unit key‖ | **p_img AUC** |
|---|---:|---:|---:|---:|---:|
| `none` (D1) | 1.19 | 0.700 | **0.9986** | 0.836 | **0.8678** |
| `pca_drop:20` | 33.3 | 0.000 | 0.670 | 0.020 | 0.8593 |
| `whiten:128` | 103.1 | 0.001 | 0.596 | 0.032 | 0.8663 |
| `whiten:512` | 364.1 | 0.000 | **0.386** | 0.019 | 0.8689 |

The collapsed cone is gone — random-pair cosine 0.700 → 0.000, the alarming
nearest-neighbour floor 0.9986 → 0.386, the mean unit key 0.836 → 0.019 — and
`p_img` AUC moves by **+0.0011**. The near-rank-1 cone is a true *description*
of the space and a false *explanation* of the failure.

Two further measurements say why the "delete what the classifier reads" idea also
fails. Per fold, between the top principal direction of the key space and the
linear head's decision direction `w = W[1] − W[0]`:

| fold | \|cos(PC₁, w)\| | share of \|w\|² inside the top 1 / 20 / 100 PCs |
|---|---:|---|
| 0 | 0.601 | 0.36 / 0.45 / 0.56 |
| 1 | 0.402 | 0.16 / 0.21 / 0.28 |
| 2 | 0.466 | 0.22 / 0.27 / 0.37 |
| 3 | 0.625 | 0.39 / 0.45 / 0.56 |
| 4 | 0.509 | 0.26 / 0.34 / 0.44 |

The dominant axis and the decision direction are *related* (cos ≈ 0.5), not
identical, and **44–72% of the head's decision direction lives outside the top 100
principal directions** — in the low-variance tail that a cosine ranking all but
ignores. So the two readouts are not geometrically the same projection at all:
cosine kNN ranks on high-variance directions, the head reads mostly the tail, and
they still agree at corr 0.948. Consistently, `drop_dirs` — deleting the head's
exact decision direction from the key — changes `p_img` AUC by 0.0021 and the
correlation from 0.948 to 0.941. **You cannot subtract the redundancy out of the
key, because it was never confined to a direction.**

The redundancy is *semantic*: both readouts are functions of the same encoder's
representation of the same image, and that encoder has already formed its opinion
about malignancy. Retrieval returns the images this encoder thinks look alike —
and "looks alike" already carries the encoder's class judgement, including where
that judgement is wrong.

---

## 5. Answers to the three suggestions in `docs/retrieval.md` Part VII

**(1) "Key on `fpn_features`, the spatial pyramid, not the pooled vector."**
**Tested, refuted.** Every pooling GAP destroys was measured: spatial standard
deviation (`fpn.std`, the purest "what GAP throws away" signal) gives the best
same-encoder `p_img` AUC of the study, **0.8699** — and still ranks *below* the D1
key once fused (LOSO −0.0009), with correlation 0.937. Per-level keys are no
different: `fpn.gap@0..3` span 0.8618–0.8687 AUC, and per-level L2 normalisation
before concatenation (`fpn.gap.ln`, which tests whether the concat cosine is
secretly one dominant level) changes nothing (0.8667). Composites do not add:
`features+fpn.std` reaches 0.8686 with correlation 0.946. Dense/region structure
from **this** encoder is as redundant as its pooled summary.

**(2) "Use an encoder whose retrieval key is not the classifier's input."**
This turned out to be the right instinct for the wrong reason — see §6. Within
exp3n, `projections` *is* such a key (the SupCon head, trained by a different
objective, not read by the classifier), and it is the flattest result in the study:
AUC 0.8660, correlation 0.944, effective rank **1.08** — the most collapsed space
measured. SupCon does not spread the space; it tightens class clusters, which is
the same axis again.

**(3) "Secondary levers (route, k, T) cannot lift the ceiling."** Confirmed, and
not re-litigated: D3–D7 were held fixed throughout so that the key is the only
moving part.

---

## 6. The one thing that moved: a different encoder

The control that separates "wrong key" from "wrong encoder": build the bank and
the query keys from **exp1**'s frozen checkpoints — a CE-only encoder with no
SupCon and no magnification block, and a *weaker* classifier — while `p_param`
still comes from exp3n's own head. Row alignment between the two caches is asserted
before anything is scored.

| | same encoder (39 configs) | cross-encoder (exp1 keys, 4 configs) |
|---|---:|---:|
| **AUC where the head is wrong** | 0.051 – 0.100 (mean **0.070**) | 0.127 – 0.136 (mean **0.131**, ×1.88) |
| corr(`p_param`, `p_img`) | 0.885 – 0.948 (mean 0.918) | 0.868 – 0.908 (mean 0.889) |
| best fixed-α blend gain | +0.0000 – +0.0029 among the top-5 ranked | **+0.0020 – +0.0047** for all four, α ≈ 0.71–0.89 |
| gate weight kept on the memory | 0.027 – 0.090 across the top-5 ranked | **0.082 – 0.150** |
| best LOSO Δ image accuracy | **negative for all 39** (best −0.0009) | **+0.0013** (`pca_drop:10`), +0.0005 (`features`) |
| LOSO Δ patient accuracy | best +0.0152 (**+1** patient); most 0 or negative | **+0.0152 / +0.0303 / +0.0455 / +0.0606** — all four positive |

Every cross-encoder row, in full:

```
[exp1] features/none        p_img 0.8718  p_slide 0.8570 | LOSO auc 0.8872 (+0.0017)
                            acc 0.8659 (+0.0005)  patient acc 0.9242 (+0.0606, +4 patients)
                            w = (0.836, 0.132, 0.032)   best-α 0.71
[exp1] fpn.std/none         p_img 0.8723  p_slide 0.8602 | LOSO auc 0.8883 (+0.0028)
                            acc 0.8652 (−0.0002)  patient acc 0.9091 (+0.0455, +3 patients)
                            w = (0.808, 0.150, 0.042)   best-α 0.72
[exp1] features/whiten:128  p_img 0.8676  p_slide 0.8650 | LOSO auc 0.8868 (+0.0013)
                            acc 0.8651 (−0.0003)  patient acc 0.8939 (+0.0303, +2 patients)
[exp1] features/pca_drop:10 p_img 0.8593  p_slide 0.8039 | LOSO auc 0.8850 (−0.0005)
                            acc 0.8667 (+0.0013)  patient acc 0.8788 (+0.0152, +1 patient)
```

Three things stand out. The **error-decorrelation nearly doubles (×1.88)** — this is
the only quantity in the whole study that responds to *anything*, and it responds
to changing the encoder, not the key. **exp1 keys give the two highest genuine
retrieval AUCs measured** (0.8723, 0.8718) despite exp1 being the weaker
classifier, which is only possible if it ranks similarity on a partly different
notion. And the gate — which honestly drops to 0.03–0.09 on the best same-encoder
keys — **keeps 0.08–0.15 weight** on foreign-encoder evidence, with all four
configurations improving patient accuracy where the best of 39 same-encoder keys
managed one patient.

**This is not yet a win.** The patient-level +0.0606 is four patients out of 66,
inside the quantisation band `docs/results/classifier_ladder.md` §6 and `docs/results/threshold_calibration.md`
§5 both warn about; the bootstrap interval (§3.3) straddles zero; and exp1 was
picked because it was already trained, not by a pre-registered rule. It is a
**direction with a measured mechanism**, which is the most this study found.

---

## 7. Recommendation

**7.1 Keep D1 exactly as it is.** `retrieval.key: features`,
`retrieval.key_transform: none`. Not because it won — because nothing beat it, and
it is the simplest, already-reported, already-validated choice. Adopting
`pca_drop:20` on a +0.0042 AUC that is negative on accuracy and inside an interval
eleven times its width would be tuning to a number. Consider the key question
**closed** and cite this report for it.

**7.2 Record it as decision D10.** *"The retrieval key is configurable
(`retrieval.key` / `retrieval.key_transform`); 43 alternatives spanning every
forward-dict vector, four FPN poolings, every pyramid level and eight fitted
transforms were measured on pooled OOF; none beats the pooled `features` key, and
the geometry of the key space does not predict retrieval quality
(corr(log effective rank, p_img AUC) = +0.007). The binding constraint is that the
bank and the classifier share an encoder."*

**7.3 Amend `docs/retrieval.md` Part VII.** Its verdict — "the module does not help on
this encoder, report it, do not tune it" — stands and is now much better
evidenced. Its *mechanism* needs correcting: the failure is not the near-rank-1
geometry (§4 falsifies that directly) and not the pooling (§5), it is that a
retrieval branch reading a frozen encoder cannot know anything that encoder does
not already know. Its proposed fix #1 (key on `fpn_features`) is now tested and
does not work; fix #2 is right in spirit but must mean a *different encoder*, not
a different head on the same one.

**7.4 The next increment, if the module must improve accuracy: encoder
diversity.** In rough order of cost:

1. **Free** — promote the cross-encoder control to a proper experiment. exp1's five
   folds already exist; `--set retrieval.key_encoder=exp1` is not yet a production
   flag, and adding it (bank built from encoder A, `p_param` from encoder B) is a
   small, contained change to `build_memory_bank.py`. Pre-register it, run the
   full B1→B2→C pipeline once, report whatever comes out.
2. **Cheap** — a bank fused over *several* frozen encoders (exp1 + exp2 + exp3), one
   `p_img` per encoder, gate over 4–5 terms. The `term_mask` machinery in
   `FusionGate` already supports extra terms.
3. **Real work** — train a bank encoder *for* complementarity: a second encoder with
   a loss term that penalises agreement with the frozen head's errors. This is the
   only route that attacks the mechanism head-on rather than hoping for incidental
   diversity.

**7.5 If accuracy is not the goal, the module is already delivering.** Retrieval
returns *evidence* — named archived slides with subtypes and similarities that a
pathologist can inspect. `docs/retrieval.md` §9 already lists that as a contribution
independent of the accuracy delta, and this study does not weaken it. Retrieval
neighbourhoods carry a consistent **+0.11 to +0.15 subtype lift** over base rate
across all 39 keys: the retrieved slides are genuinely the morphologically related
ones, even where the label vote adds nothing the head does not already have.

**7.6 One measurement worth taking before any of the above.** `p_param` is at
**0.9598 patient-level AUC** on out-of-fold validation. The memory is being asked
to improve a head that is already close to the ceiling this dataset supports at the
patient level, on 66 patients where one patient is 1.5 accuracy points. Decide
whether the module's headline claim should be accuracy at all before spending GPU
hours on making it one.

---

## 8. Reproducing

```powershell
# Stage 1 — encode each fold once (bank split + val split), ~15 min GPU per encoder
python scripts/retrieval_key_ablation.py --stage cache                    # exp3n
python scripts/retrieval_key_ablation.py --stage cache --experiment exp1  # cross-encoder control

# Stage 2 — the sweep, ~30 min, no GPU training
python scripts/retrieval_key_ablation.py --stage eval --cross-encoder exp1

# per-row probabilities for the bootstrap intervals in §3.3
python scripts/retrieval_key_ablation.py --stage eval --dump-probs --only features
```

Outputs land in `analysis/retrieval_keys/<experiment>/`: `cache_fold<k>.npz`
(~91 MB each, deletable — regenerated from the checkpoints),
`key_ablation.json`, `key_ablation.log`, optional `probs_*.npz`.

A single configuration can be taken to production without this script:

```powershell
python scripts/build_memory_bank.py --experiment exp3n `
  --set retrieval.key=fpn.std --set retrieval.key_transform=whiten:128
python scripts/train_gate.py --experiment exp3n --set retrieval.key=fpn.std `
  --set retrieval.key_transform=whiten:128
```

The bank records both values and the loader refuses a mismatch, so a bank and a
config that disagree raise rather than quietly producing numbers.

**The 16-patient test set was not touched by any part of this study.** Every number
here is pooled out-of-fold validation. If a configuration is ever adopted, the
honest confirmation is one run of the normal B1 → B2 → C pipeline, once.
