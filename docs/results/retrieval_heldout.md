# Stage C — the Retrieval Memory on the held-out test set

> **Status: the loose end is closed.** One thing was outstanding — *"the 5-fold production retrieval run (B1 → B2 → C on the test set) has
> not been executed."* It has now been executed, on all five folds, with the production
> config and the OOF-fitted gate, and this report is the result.
>
> **Verdict: nothing changed.** The held-out test reproduces the pooled-OOF conclusion
> of `docs/retrieval.md` Part VII and `docs/results/retrieval_key_ablation.md` exactly, including the
> quantity those reports identified as the *mechanism*. The memory does not add
> discrimination; the one number that looks like a gain in the raw logs
> (`prob_final` accuracy at 0.5) is an operating-point artefact and does not survive
> the pre-registered threshold. Two things are genuinely new and both are on the
> **evidence** side, not the accuracy side (§7).

---

## 1. What was run, and the integrity checks

| | |
|---|---|
| Encoder | exp3n (D9), frozen — `checkpoints/best.pt` of each of the 5 training runs |
| Banks | `analysis/retrieval/exp3n/bank_fold{0..4}.npz`, built 2026-07-23 |
| Gate | **one** `gate.pt`, fitted once on pooled OOF (6,256 rows / 66 patients), applied to all five folds |
| Config | production defaults: `key=features`, `key_transform=none`, image view `route=same_mag k=15 cap=3 T=0.07`, slide view `route=all k=5 cap=1`, `merge_levels=False`, `block_query_patients=True` |
| Test set | 16 held-out patients, 1,653 images (**4 benign / 12 malignant**) |
| Outputs | `runs/exp3n_swin_supcon_cls/*/test/test_metrics_retrieval.json`, `test_predictions_retrieval.csv`, `exemplars/` |

Everything the protocol asserts, passed:

- **Leakage.** `retrieval: leakage guard passed -- none of the 16 evaluated patients is in the bank`, all five folds. Independently re-checked from the dumped neighbour lists: each fold retrieved from 52–53 distinct bank patients, **overlap with the test patients = 0** in every fold.
- **No encoder drift.** `prob_param` is bit-identical to the plain (non-retrieval) test run: fold 0 `accuracy 0.9292 / auc 0.9710` in both `test_metrics.json` and `test_metrics_retrieval.json`. The retrieval run is a strict superset of the baseline, so every Δ below is a clean paired comparison, not two different models.
- **Routing / D5 invariant.** `same_magnification_rate = 1.0000` under `route=same_mag` (expected by construction), `mean_neighbours = 15.00`, `short_neighbourhood_fraction = 0.0%`, `mean_slide_neighbours = 5.00`, `p_slide_std = 0.37–0.46` (non-zero ⇒ the slide ranking is genuinely independent of the image ranking, not merged).
- **Per-patient cap working.** `mean_distinct_patients = 7.40–8.03` per query out of 15 neighbours, against the D4 floor of 2.0.

So: no bug, no leak, no misconfiguration. What follows is a measurement, not a debugging session.

---

## 2. The four streams, per fold

`prob_param` = the parametric head alone (= the exp3n baseline). `prob_img` / `prob_slide` = the two retrieval votes. `prob_final` = the gate's fusion.

**Image-level AUC** (the discrimination number — threshold-free):

| fold | prob_param | prob_img | prob_slide | prob_final | Δ(final − param) |
|---|---:|---:|---:|---:|---:|
| 0 | **0.9710** | 0.9595 | 0.9187 | 0.9704 | −0.0006 |
| 1 | 0.9194 | **0.9256** | 0.8721 | 0.9149 | −0.0045 |
| 2 | **0.9622** | 0.9476 | 0.9199 | 0.9579 | −0.0043 |
| 3 | **0.9737** | 0.9470 | 0.9230 | 0.9727 | −0.0010 |
| 4 | **0.9701** | 0.9496 | 0.9266 | 0.9646 | −0.0054 |
| **mean** | **0.9593** | 0.9459 | 0.9121 | 0.9561 | **−0.0032** |

**`prob_final` loses AUC to `prob_param` in 5 / 5 folds.** `prob_img` beats the head in 1/5 (fold 1, +0.0062); `prob_slide` in 0/5, and it is the weakest stream everywhere.

**Patient-level AUC:** `prob_param` = **1.0000 in all five folds**. `prob_final` = 1.0000 in four and **0.9792 in fold 1** — i.e. the only movement the memory produces at the patient level is to *break* a perfect ranking. There is no headroom for it to do anything else.

**Image accuracy @ 0.5:**

| fold | prob_param | prob_img | prob_slide | prob_final | Δ(final − param) |
|---|---:|---:|---:|---:|---:|
| 0 | 0.9292 | 0.9274 | 0.9298 | **0.9316** | +0.0024 |
| 1 | 0.8972 | **0.9159** | 0.8875 | 0.9008 | +0.0036 |
| 2 | 0.8959 | 0.8917 | **0.9056** | 0.8978 | +0.0018 |
| 3 | 0.9177 | 0.9171 | **0.9183** | 0.9177 | +0.0000 |
| 4 | 0.8530 | **0.9250** | 0.9201 | 0.8808 | +0.0278 |
| **mean** | 0.8986 | **0.9154** | 0.9126 | 0.9057 | **+0.0071** |

This is the only table in the run that looks like a win, and it is where the trap is. §4.

---

## 3. The fold-3 warning — what it actually means

You saw `WARNING: The gate has collapsed onto the parametric head — mean w_param = 0.922: the memory is barely contributing` in fold 3 only. That is **a threshold crossing, not a fold-3 pathology.**

The banner fires when the fold's *mean* `w_param` exceeds the constant `RETRIEVAL_GATE_COLLAPSE = 0.90` ([rap_mst/utils/reporting.py:498](../../rap_mst/utils/reporting.py#L498), test at [line 645](../../rap_mst/utils/reporting.py#L645)). The five means are:

| fold | mean `w_param` | mean `w_img` | mean `w_slide` | banner |
|---|---:|---:|---:|---|
| 0 | 0.876 | 0.098 | 0.026 | — |
| 1 | 0.681 | 0.230 | 0.089 | — |
| 2 | 0.687 | 0.247 | 0.066 | — |
| 3 | **0.922** | 0.070 | 0.008 | **fires** |
| 4 | 0.827 | 0.163 | 0.009 | — |
| **pooled** | **0.799** | 0.161 | 0.040 | — |

Two readings follow, and the second is the correct one:

**(a) Every fold is parameter-dominated.** Pooled over all 8,265 test queries the gate keeps **0.799** on the head, and **69.8% of individual queries** sit above `w_param > 0.9` on their own. Fold 3 is not qualitatively different — it is the fold whose *average* happens to land on the far side of a round number. Folds 0 and 4 (0.876, 0.827) are the same behaviour one notch below the line.

**(b) Fold 3 is simply the most redundant fold, and the diagnostics say so consistently.** The gate is doing exactly what it should:

| fold | mean `w_param` | neighbour agreement | corr(p_param, p_img) | AUC of `p_img` on the head's errors |
|---|---:|---:|---:|---:|
| 0 | 0.876 | 0.925 | 0.987 | 0.012 |
| 1 | 0.681 | 0.820 | 0.925 | 0.172 |
| 2 | 0.687 | 0.748 | 0.953 | 0.155 |
| 3 | **0.922** | **0.955** | **0.993** | **0.010** |
| 4 | 0.827 | 0.882 | 0.866 | 0.089 |

`corr(mean w_param, neighbour agreement)` across the five folds = **+0.946**. Fold 3 has the most unanimous neighbourhoods (0.955), the highest head↔memory correlation (0.993) and the lowest error-decorrelation (0.010) — the memory there is the most redundant copy of the head in the whole run, and the gate closes on it hardest. That is the module reporting the truth about fold 3's encoder state, exactly as `docs/retrieval.md` §6 prescribes.

**Do not act on this banner.** Its own message says so: *"report it as 'the module does not help on this encoder' rather than tuning until it does."* The only thing worth noting for the write-up is that the 0.90 constant is a reporting convenience — the honest statement is *"the gate keeps 0.68–0.92 on the parametric head across folds (pooled 0.80), and 70% of queries individually exceed 0.9."*

---

## 4. The trap: the @0.5 accuracy gain is calibration, not information

`prob_final` beats `prob_param` on accuracy@0.5 in **5/5** folds (mean +0.0071) while losing AUC in **5/5** (mean −0.0032). Those two facts can only coexist if the fusion is moving the *operating point*, not the ranking. Two independent controls confirm it:

**Control 1 — the oracle (test-optimal) threshold.** Give each stream its best possible threshold on the test set itself and the gain evaporates or inverts:

| fold | Δacc @ 0.5 | Δacc @ oracle threshold |
|---|---:|---:|
| 0 | +0.0024 | −0.0006 |
| 1 | +0.0036 | +0.0006 |
| 2 | +0.0018 | +0.0042 |
| 3 | +0.0000 | −0.0036 |
| 4 | **+0.0278** | **−0.0139** |
| **mean** | **+0.0071** | **−0.0027** |

Fold 4 is the whole effect, and it is textbook. Fold 4's head is the *miscalibrated* fold — `prob_param` at 0.5 gives sensitivity 0.816 / specificity 0.950, an accuracy of 0.8530 against an AUC of 0.9701 (its own oracle accuracy is 0.9262). Mixing in the retrieval vote drags the probabilities toward the middle and lands the fixed 0.5 cut in a better place. That is a re-calibration you could have obtained with a scalar temperature and no memory at all — and once fold 4 is allowed its correct threshold, the memory *costs* it 1.4 points.

**Control 2 — the pre-registered threshold, which is the honest protocol.** `docs/results/threshold_calibration.md` fixes thresholds on pooled OOF validation and locks them before the test set is touched (`prob_final` → 0.6923 image, 0.3775 patient). At those locked thresholds:

| fold | prob_param | prob_img | prob_slide | prob_final | Δ(final − param) |
|---|---:|---:|---:|---:|---:|
| 0 | **0.9304** | 0.9268 | 0.9298 | 0.9280 | −0.0024 |
| 1 | **0.8984** | 0.8808 | 0.8633 | 0.8730 | −0.0254 |
| 2 | **0.8947** | 0.8548 | 0.8923 | 0.8845 | −0.0103 |
| 3 | 0.9153 | 0.9135 | **0.9183** | 0.9105 | −0.0048 |
| 4 | 0.8518 | 0.9141 | **0.9201** | 0.8409 | −0.0109 |
| **mean** | **0.8981** | 0.8980 | 0.9048 | 0.8874 | **−0.0108** |

**Under the pre-registered protocol the memory loses accuracy in 5/5 folds, by −0.011 on average.** The +0.0071 at 0.5 and the −0.0108 at the locked threshold are the same run; the difference is entirely which threshold you were allowed to choose. Report the locked one.

*(A note on the locked **patient** threshold row: `prob_final` "wins" it 0.9750 vs 0.9375. That is not a memory effect either — `prob_param`'s locked patient threshold of 0.1704 was a plateau pick from OOF that transfers badly, and `prob_param` scores 0.9750 at plain 0.5. Both branches land on 0.975; the difference is a threshold-selection artefact on 16 patients where one patient = 6.25 points.)*

---

## 5. Five-fold ensemble and the significance test

The five folds share one test set, so they are not independent replicates and "5/5" is a consistency statement, not a p-value. The defensible aggregate is the fold-ensemble (mean probability over the 5 folds) with a patient-clustered bootstrap over the 16 test patients:

| stream | image AUC | image acc @0.5 | patient AUC | patient acc @0.5 |
|---|---:|---:|---:|---:|
| **prob_param** | **0.9761** | 0.9335 | 1.0000 | **16/16** |
| prob_img | 0.9742 | **0.9443** | 1.0000 | 16/16 |
| prob_slide | 0.9614 | 0.9353 | 1.0000 | 16/16 |
| prob_final | 0.9758 | 0.9371 | 1.0000 | 16/16 |

Patient-clustered bootstrap, 2,000 resamples of the 16 test patients, paired against `prob_param`:

| stream | Δ AUC | 95% CI | P(Δ>0) | Δ acc@0.5 | 95% CI | P(Δ>0) |
|---|---:|---|---:|---:|---|---:|
| prob_img | −0.0019 | [−0.0045, +0.0004] | 0.06 | +0.0106 | [−0.0012, +0.0271] | 0.93 |
| prob_slide | **−0.0145** | **[−0.0329, −0.0009]** | 0.00 | +0.0015 | [−0.0158, +0.0167] | 0.58 |
| prob_final | −0.0003 | [−0.0010, +0.0008] | 0.21 | +0.0034 | [−0.0058, +0.0128] | 0.73 |

Read: **the fusion is statistically indistinguishable from the head** (ΔAUC −0.0003, interval straddling zero, and the accuracy delta is the §4 calibration effect). The one interval that *excludes* zero is `prob_slide`'s AUC, and it excludes it on the **wrong side** — the slide-centroid vote is significantly worse than the head, which is consistent with the gate assigning it 0.008–0.089.

At the patient level all four streams are 16/16 and 1.000 AUC on the ensemble. `docs/results/retrieval_key_ablation.md` §7.6 predicted this: the head was already at 0.9598 patient AUC on OOF, and on this test set it is saturated. **The patient level had no room to show a difference in either direction, and the report should say so rather than quote 16/16 as a retrieval result.**

Decision-level view: fusion flips **18 of 1,653** ensemble predictions (1.09%); 12 flips become correct, 6 become wrong. Net +6 images. That is the entire mechanical footprint of the module on this test set.

---

## 6. Does it match what we concluded before? — yes, on the mechanism, not just the outcome

This is the part that matters for the paper. `docs/results/retrieval_key_ablation.md` §6 did not just say "it doesn't help" — it named **error-decorrelation from the head** as the quantity that predicts whether retrieval can help, and showed it responds only to *encoder identity*. That claim was made on pooled OOF validation. It now transfers to held-out data unchanged:

| quantity | OOF, same encoder (39 configs) | OOF, cross-encoder (exp1 keys) | **TEST, this run** |
|---|---:|---:|---:|
| AUC of `p_img` restricted to the head's errors | 0.051 – 0.100 (mean **0.070**) | 0.127 – 0.136 (mean **0.131**) | 0.010 – 0.172 (mean **0.088**); ensemble **0.073** |
| corr(`p_param`, `p_img`) | 0.885 – 0.948 (mean 0.918) | 0.868 – 0.908 (mean 0.889) | 0.866 – 0.993 (mean **0.945**); ensemble 0.991 |
| gate weight kept on the memory | 0.027 – 0.090 (top-5 keys) | 0.082 – 0.150 | 0.078 – 0.319 per fold; pooled **0.201** |
| Δ image accuracy vs head | negative for all 39 (best −0.0009) | +0.0005 / +0.0013 | **negative for all 5 at the locked threshold** (mean −0.0108) |

The test set lands **inside the same-encoder band and nowhere near the cross-encoder band** on the diagnostic quantity, and produces the same sign on the outcome. Nothing about the OOF study was an artefact of fitting the gate on the surface it was scored on — the D10 conclusion survives contact with held-out data.

Point by point against the prior establishments:

| prior claim | source | status after Stage C |
|---|---|---|
| The module is correctly implemented; every guard/invariant holds | `docs/retrieval.md` Part VII §1 | **Confirmed** — leakage, routing, D5, cap all re-verified on test (§1) |
| The gate honestly collapses onto the parametric head | `docs/retrieval.md` Part VII §1, §5 | **Confirmed** — pooled `w_param` 0.799 on test vs 0.823 fitted on OOF |
| Retrieval adds no discrimination over the head on this encoder | `docs/retrieval.md` Part VII §4, `key_ablation` §3.1 | **Confirmed** — `prob_final` AUC below `prob_param` in 5/5 folds; ensemble ΔAUC −0.0003 [−0.0010, +0.0008] |
| Fused "gains" are largely a gate/fitting artefact | `key_ablation` §3.2 | **Confirmed, in a new guise** — on test the artefact is the *threshold*, not the gate fit (§4). Same lesson |
| Nothing is statistically significant | `key_ablation` §3.3 | **Confirmed** — every interval straddles zero except `prob_slide`, which is significantly *worse* |
| The binding constraint is shared encoder, measured by error-decorrelation | `key_ablation` §6 | **Confirmed on held-out data** — test `awrong` mean 0.088 sits in the same-encoder band (0.051–0.100), not the cross-encoder band (0.127–0.136) |
| Retrieved neighbourhoods carry a real +0.11–0.15 subtype lift | `key_ablation` §7.5 | **Confirmed and slightly stronger on test** — §7 below |
| The patient level is near ceiling; decide if accuracy is the claim | `key_ablation` §7.6 | **Confirmed the hard way** — `prob_param` is 16/16 with patient AUC 1.000 on the ensemble; there was no headroom |
| Run B1→B2→C once on test and report whatever comes out | the one standing open item | **Done. This report.** |

**Nothing changed.** No number reverses a prior conclusion; the test set narrows the intervals and closes the "unfinished" objection the paper warned a reviewer would raise.

---

## 7. Two things that are genuinely new (both on the evidence side)

**7.1 The gate is input-conditioned, and it opens exactly where the head is wrong.**
Nothing before this run looked at the *distribution* of `w_param` — only its mean. It is strongly bimodal (pooled: 69% of queries above 0.95, 9% below 0.2, median 0.991), and it is not noise:

| fold | mean `w_param` where head is **correct** | where head is **wrong** | gap |
|---|---:|---:|---:|
| 0 | 0.899 | 0.564 | −0.335 |
| 1 | 0.712 | 0.417 | −0.295 |
| 2 | 0.705 | 0.533 | −0.172 |
| 3 | 0.936 | 0.765 | −0.171 |
| 4 | 0.871 | 0.576 | −0.295 |

In **every** fold the ~150-parameter gate relaxes onto the memory on precisely the images the parametric head misclassifies, using only the five retrieval summary features and never seeing the label at test time. It is a working, honest *selective-consultation* mechanism. It buys nothing in accuracy because — per §6 — the memory it consults is a redundant copy of the head, so it opens the door on a witness that gives the same wrong answer (`awrong` = 0.073 on the ensemble, i.e. on the head's errors the memory is wrong in the *same direction* almost every time). This is the cleanest single demonstration in the project that the module's architecture is sound and the *encoder* is the constraint — the gate behaves correctly and still cannot win. Worth a figure.

**7.2 The subtype lift holds on held-out patients — measured on test for the first time.**

| fold | P(neighbour subtype = query subtype) | marginal base rate | lift |
|---|---:|---:|---:|
| 0 | 0.422 | 0.255 | **+0.167** |
| 1 | 0.391 | 0.255 | +0.135 |
| 2 | 0.426 | 0.239 | **+0.186** |
| 3 | 0.386 | 0.222 | +0.164 |
| 4 | 0.357 | 0.227 | +0.130 |
| **mean** | 0.396 | 0.240 | **+0.156** |

Against the +0.11–0.15 measured across all 39 keys on OOF, the test set gives **+0.156**. For 16 patients the module has never seen, the neighbours it returns are the morphologically related archived slides — with patient ids, subtypes and similarities in `test_predictions_retrieval.csv` and `exemplars/*.json` — at a rate ~65% above chance, while contributing nothing to the label decision. That is the contribution `key_ablation` §7.5 flagged and had not confirmed on held-out data. Now it is confirmed.

**One negative worth recording as a design finding:** the slide-centroid level does not earn its place. It has the worst AUC of the four streams in 5/5 folds, is the only stream significantly worse than the head on the bootstrap, and the gate gives it 0.008–0.089. `docs/retrieval.md` D5's "unify the store, never unify the ranking" is architecturally right and was verified again here — but on *this* dataset (52 bank patients ⇒ 52 centroids) the slide ranking is too coarse to contribute. Report it; do not delete it — it is the level a WSI-scale bank would exercise.

---

## 8. What to state in the write-up

The defensible claims, in the order they should appear:

1. Held-out test, 16 patients / 1,653 images, five folds, pre-registered thresholds: **the retrieval branch does not improve the classifier.** Ensemble ΔAUC = −0.0003 (95% CI [−0.0010, +0.0008]); Δ accuracy at the locked threshold = −0.0108 averaged over folds, negative in 5/5. Patient level saturated at 16/16 for every stream.
2. **The mechanism identified on validation transfers.** Error-decorrelation from the head on test (mean 0.088, ensemble 0.073) sits in the same-encoder band measured over 43 key configurations (0.051–0.100) and far below the cross-encoder control (0.127–0.136). The constraint is that the bank and the classifier share a frozen encoder — confirmed out of sample.
3. **The gate works.** It is input-conditioned, opens on the head's errors in 5/5 folds, and still cannot win, because the evidence it consults is wrong in the same places. This is the positive control for the architecture.
4. **The retrieval is useful as evidence.** +0.156 subtype lift over base rate on unseen patients, with named, inspectable exemplars.
5. **Do not report accuracy@0.5.** It shows a +0.0071 mean gain that is a fold-4 calibration effect, reverses at the oracle threshold (−0.0027) and reverses at the pre-registered threshold (−0.0108). Quoting it would be exactly the "tuning to a number" that D9 and the fold-3 banner both forbid.

The framing in the paper ("boundary conditions, not failure") is unchanged and is now backed by a held-out confirmation rather than validation-only evidence. The next increment remains `key_ablation` §7.4 item 1 — promote the exp1 cross-encoder control to a production flag and run this same B1→B2→C pipeline once with `key_encoder=exp1`, pre-registered.

---

## 9. Reproducing this report

```bash
# Stage C, per fold (already run; see docs/COMMANDS.md §11)
python scripts/test.py --experiment exp5 --fold {0..4} --retrieval

# the analysis in this document (numpy/pandas/sklearn only, no GPU)
#   reads runs/exp3n_swin_supcon_cls/*/test/test_predictions_retrieval.csv
#            + test_metrics_retrieval.json
```

Source artefacts: `runs/exp3n_swin_supcon_cls/*_train_fold{k}/test/test_metrics_retrieval.json`
(metrics + diagnostics + config echo), `test_predictions_retrieval.csv` (per-image
probabilities, per-query gate weights, neighbour ids/subtypes/similarities),
`test.log` (banners), `analysis/retrieval/exp3n/gate_fit.json` (the OOF gate this run used).
