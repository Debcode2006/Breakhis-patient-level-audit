# COMMANDS — copy-paste PowerShell

All commands assume:
- Working directory is the repo root: `d:\Projects\RAP-MST-v1`
- The conda env is active: `conda activate rapmst`

To activate and move in:
```powershell
conda activate rapmst
Set-Location "d:\Projects\RAP-MST-v1"
```

---

## 1. Create patient splits (run once)

```powershell
python scripts/prepare_splits.py --config config/config.yaml
```

Regenerate from scratch (invalidates prior experiments):
```powershell
python scripts/prepare_splits.py --config config/config.yaml --force
```

---

## 2–6. Train the 5 cross-validation folds

These examples use **Experiment 1**. Swap `--experiment exp1` for `exp2` / `exp3`
to run the other experiments.

**Train fold 1** (fold index 0):
```powershell
python scripts/train.py --experiment exp1 --fold 0
```

**Train fold 2** (fold index 1):
```powershell
python scripts/train.py --experiment exp1 --fold 1
```

**Train fold 3** (fold index 2):
```powershell
python scripts/train.py --experiment exp1 --fold 2
```

**Train fold 4** (fold index 3):
```powershell
python scripts/train.py --experiment exp1 --fold 3
```

**Train fold 5** (fold index 4):
```powershell
python scripts/train.py --experiment exp1 --fold 4
```

Run all five folds in sequence:
```powershell
foreach ($k in 0..4) { python scripts/train.py --experiment exp1 --fold $k }
```

---

## 7. Test on the held-out 16-patient test set

Point at the `best.pt` of the run you want to evaluate (architecture + config are
read from the checkpoint):
```powershell
python scripts/test.py --checkpoint "runs\exp1_swin_cls\<TIMESTAMP>_train_fold0\checkpoints\best.pt"
```

Find the newest run directory for an experiment:
```powershell
Get-ChildItem "runs\exp1_swin_cls" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

Results are written to `runs\exp1_swin_cls\<run>\test\` (`test_metrics.json`,
`test_predictions.csv`).

---

## 8. Resume training

Resume from a run's `last.pt` (restores optimizer/scheduler/scaler/RNG/epoch):
```powershell
python scripts/train.py --experiment exp1 --fold 0 --resume "runs\exp1_swin_cls\<TIMESTAMP>_train_fold0\checkpoints\last.pt"
```

> Resume starts a **new** run directory but continues from the saved epoch and
> best-metric bookkeeping, so previous outputs are preserved.

---

## 9. Change experiments

Experiment 2 (adds magnification embedding):
```powershell
python scripts/train.py --experiment exp2 --fold 0
```

Experiment 3 (adds SupCon; needs the projection head, enabled by the preset):
```powershell
python scripts/train.py --experiment exp3 --fold 0
```

> The line above runs SupCon in **single-view** mode (in-batch same-class
> positives only). SupCon now always computes in fp32 internally, so this is safe,
> but for the strongest contrastive signal enable the **two-view** pipeline below.

**Experiment 3 with two-view SupCon (recommended).** `data.two_view=true` emits a
second augmented crop per image so every anchor gets a guaranteed positive pair
(the `[B, 2, D]` contrastive setup). This doubles the forward batch, so drop
`batch_size` to 8 (and raise `grad_accum_steps` to keep the effective batch) to
stay within 4 GB VRAM:
```powershell
python scripts/train.py --experiment exp3 --fold 0 --set data.two_view=true --set data.batch_size=8 --set train.grad_accum_steps=4
```

If you have more VRAM, you can keep the larger batch and skip the accumulation:
```powershell
python scripts/train.py --experiment exp3 --fold 0 --set data.two_view=true
```

Run all five folds with two-view SupCon:
```powershell
foreach ($k in 0..4) { python scripts/train.py --experiment exp3 --fold $k --set data.two_view=true --set data.batch_size=8 --set train.grad_accum_steps=4 }
```

> `data.two_view` only affects the **training** split; validation/test stay
> single-view. It is a no-op for exp1/exp2 (no projection head).

**Experiment 3n — SupCon without the magnification embedding.** The base encoder
recommended for the retrieval module (`docs/results/magnification_audit.md` Part 1 / `docs/retrieval.md`
D9). Same recipe as exp3, magnification off:
```powershell
foreach ($k in 0..4) {
  python scripts/train.py --experiment exp3n --fold $k `
    --set data.two_view=true --set data.batch_size=8 --set train.grad_accum_steps=4
}
```

---

## 9b. Bring exp3n up to parity with exp1–exp3 (the full analysis suite)

Run these **in order** after exp3n's five folds have trained. Together they
reproduce, for exp3n, every analysis that exists for exp1–exp3, so the two can be
compared on identical footing before picking the retrieval base encoder
(`docs/results/magnification_audit.md` Part 1 / `docs/retrieval.md` D9).

**Step 1 — train all five folds** (≈10 h on a 3050; skip if already done):
```powershell
foreach ($k in 0..4) {
  python scripts/train.py --experiment exp3n --fold $k `
    --set data.two_view=true --set data.batch_size=8 --set train.grad_accum_steps=4
}
```

**Step 2 — score every fold on the held-out test set** → `docs/results/classifier_ladder.md` §2 numbers
(`test/test_metrics.json`, `test/test_predictions.csv` per fold):
```powershell
foreach ($k in 0..4) {
  $run = Get-ChildItem "runs\exp3n_swin_supcon_cls" -Filter "*train_fold$k" |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
  python scripts/test.py --checkpoint "$($run.FullName)\checkpoints\best.pt"
}
```

**Step 3 — per-patient fold diagnosis** → `docs/results/classifier_ladder.md` §3/§4 numbers:
```powershell
python scripts/diagnose_folds.py --experiment exp3n
```

**Step 4 — threshold calibration** → `docs/results/threshold_calibration.md`. Include
the other experiments in the same run so the comparison table is regenerated as a
whole (re-runs validation inference for each; allow ~1 h):
```powershell
python scripts/threshold_calibration.py --experiments exp1 exp2 exp3 exp3n
```

**Step 5 — UMAP + separation metrics** → `docs/results/embedding_geometry.md`, and the
`.npz` embedding dumps every retrieval probe reads. The projections column is now
detected from each checkpoint's saved config, so exp3n gets one automatically:
```powershell
python scripts/visualize_embeddings.py --experiments exp1 exp2 exp3 exp3n
```

**Step 6 — re-run the retrieval probes on the new embeddings** → `docs/retrieval.md`
and `docs/retrieval.md`. This is what actually decides the base encoder:
```powershell
python scripts/retrieval_probe.py  --experiments exp1 exp2 exp3 exp3n
python scripts/retrieval_probe2.py --experiments exp1 exp2 exp3 exp3n
python scripts/retrieval_probe3.py --experiments exp3 exp3n
python scripts/retrieval_probe4.py --experiments exp1 exp2 exp3 exp3n
```

**What to look at when it finishes** — the D9 predictions in `docs/retrieval.md`
§1.6, in this order:

| where | quantity | prediction for exp3n |
|---|---|---|
| step 2 | test AUC | ≥ 0.9607 (exp3's) |
| step 6, probe 1 | `projections` same-mag neighbour rate | **≈ 0.33**, down from exp3's 0.976 |
| step 6, probe 1 | key same-mag neighbour rate | ≈ 0.33 (not 1.00) |
| step 5 | binary silhouette of `embeddings` | ≥ 0.62, up from exp3's 0.569 |
| step 6, probe 2 | error-rescue rate | ≥ 0.30 (exp3 level) |

If test AUC drops materially, D9 is falsified — keep exp3 as the retrieval base
and say so.

---

## 10. Edit / override configuration

**Permanent** changes — edit the file:
```powershell
notepad config\config.yaml
```

**One-off** overrides on the CLI (repeatable, YAML-typed values):
```powershell
# smaller batch if you hit CUDA OOM (keep effective batch via accumulation)
python scripts/train.py --experiment exp1 --fold 0 --set data.batch_size=8 --set train.grad_accum_steps=4

# lighter FPN if still tight on 4 GB VRAM (256 -> 128 pyramid channels)
python scripts/train.py --experiment exp1 --fold 0 --set model.fpn.out_channels=128

# disable the FPN entirely (deepest-stage baseline)
python scripts/train.py --experiment exp1 --fold 0 --set model.fpn.enabled=false

# shorter run + different LR
python scripts/train.py --experiment exp3 --fold 0 --set train.epochs=30 --set optimizer.lr=0.0001

# train on a single magnification only
python scripts/train.py --experiment exp2 --fold 0 --set "data.magnifications=[100]"
```

---

## 11. Retrieval Memory module (Stages B1 → B2 → C)

The full design is `docs/retrieval.md`; the narrative walkthrough is `docs/retrieval.md`
Part VI. **The base encoder is `exp3n` and Stage A is already done** — these commands
start from its five trained folds and add ~20 minutes in total.

> **Do not mix encoders.** The bank, the gate and the test run must all come from
> `exp3n`. Each bank records the encoder, fold and key that produced it and the
> loader refuses a mismatch, so a mistake raises instead of quietly producing
> numbers.

### 11.1 Stage B1 — build one memory bank per fold *(~10 min GPU)*

```powershell
python scripts/build_memory_bank.py --experiment exp3n
```

One fold at a time (identical result):
```powershell
foreach ($k in 0..4) { python scripts/build_memory_bank.py --experiment exp3n --fold $k }
```

Writes `analysis\retrieval\exp3n\bank_fold0.npz` … `bank_fold4.npz` (~17.5 MB each),
`bank_summary.json` and `build_memory_bank.log`. The script prints a sanity table
(image rows ≈ 4,541 · slide rows = training patients · bank ∩ val/test empty · every
key's L2 norm 1.000 · four non-empty magnification shards · key dim **1024**) and
**raises** if any of it is wrong. Never work around a leakage assertion.

If you hit CUDA OOM:
```powershell
python scripts/build_memory_bank.py --experiment exp3n --set data.batch_size=8
```

### 11.2 Stage B2 — fit the fusion gate, ONCE, on pooled out-of-fold val *(~5 min)*

```powershell
python scripts/train_gate.py --experiment exp3n
```

Writes `analysis\retrieval\exp3n\gate.pt` (one gate for all folds, ~147 parameters)
plus `gate_fit.json`. It also locks the **pooled-OOF decision thresholds** for
`p_param` / `p_img` / `p_slide` / `p_final` at image and patient level inside
`gate.pt`, so Stage C can report calibrated numbers without ever touching the test
set. Expect ~7,900 pooled rows over **66** distinct patients; the script raises if a
test patient appears and warns if coverage is short.

### 11.3 Stage C — test the 16-patient held-out set with retrieval *(~5 min GPU)*

```powershell
foreach ($k in 0..4) {
  $run = Get-ChildItem "runs\exp3n_swin_supcon_cls" -Filter "*train_fold$k" |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
  python scripts/test.py --checkpoint "$($run.FullName)\checkpoints\best.pt" --retrieval
}
```

Bank and gate paths come from `config\config.yaml`; pass them explicitly only if
they live elsewhere:
```powershell
python scripts/test.py --checkpoint "<run>\checkpoints\best.pt" --retrieval `
  --bank "analysis\retrieval\exp3n\bank_fold0.npz" --gate "analysis\retrieval\exp3n\gate.pt"
```

Each fold's `test\` directory **gains** (never overwrites) `test_metrics_retrieval.json`,
`test_predictions_retrieval.csv` and `exemplars\<patient_id>.json`. Read the results
in the order `docs/retrieval.md` Part VI prescribes: named cases first, then
sensitivity at matched specificity, then aggregates with the 1/16-patient caveat.

> Run the **plain** `python scripts/test.py --checkpoint ...` (no flag) first if a
> fold has no `test_metrics.json` yet — the parametric baseline is what everything
> is compared against, and `threshold_calibration.py` reads it.

### 11.4 The ablation ladder

Each row is one `--set`, each has a pre-registered prediction (`docs/retrieval.md` §10.7).

```powershell
$run = Get-ChildItem "runs\exp3n_swin_supcon_cls" -Filter "*train_fold0" |
       Sort-Object LastWriteTime -Descending | Select-Object -First 1
$ck  = "$($run.FullName)\checkpoints\best.pt"

# routing (D3): expect -1.1 image points
python scripts/test.py --checkpoint $ck --retrieval --out "runs\ablation\route_all" `
  --set retrieval.levels.image.route=all

# per-patient cap (D4): expect -0.6 image, -2.6 specificity
python scripts/test.py --checkpoint $ck --retrieval --out "runs\ablation\nocap" `
  --set retrieval.levels.image.per_patient_cap=0

# fixed alpha instead of the learned gate (D6): expect ~equal
python scripts/test.py --checkpoint $ck --retrieval --out "runs\ablation\fixed_alpha" `
  --set retrieval.gate.enabled=false
```

Two ablations remove an evidence term, so the 3-term gate no longer applies — the
loader refuses it. Either add `--set retrieval.gate.enabled=false`, or refit:

```powershell
# granularity (D5): expect PC-9146 and TA-16184 to regress
python scripts/test.py --checkpoint $ck --retrieval --out "runs\ablation\no_slide" `
  --set retrieval.levels.slide.enabled=false --set retrieval.gate.enabled=false

# merged index (D5 / `docs/retrieval.md` Part V): expect degeneration to image-only
python scripts/test.py --checkpoint $ck --retrieval --out "runs\ablation\merged" `
  --set retrieval.merge_levels=true --set retrieval.gate.enabled=false
```

The **key ablation (D1)** needs its own bank — the key is baked into the stored
vectors — so build it first and point at it:

```powershell
python scripts/build_memory_bank.py --experiment exp3n --fold 0 `
  --out "analysis\retrieval\ablation_key_embeddings" --set retrieval.key=embeddings
python scripts/test.py --checkpoint $ck --retrieval --out "runs\ablation\key_embeddings" `
  --bank "analysis\retrieval\ablation_key_embeddings\bank_fold0.npz" `
  --set retrieval.key=embeddings --set retrieval.gate.enabled=false
```
(On exp3n this is a no-op by construction — with no magnification block
`embeddings == features`. Run it on an exp3 bank to reproduce the ~100%
magnification-locked neighbourhood that D1 refutes.)

### 11.5 Troubleshooting

| message | cause | fix |
|---|---|---|
| `LEAKAGE: n validation patient(s) present ...` | wrong fold / wrong split | never bypass; check `--fold` matches the checkpoint |
| `Bank belongs to experiment X, not Y` | mixed encoders | rebuild the bank for the encoder under test |
| `Bank was built on key='...' but retrieval.key='...'` | key ablation without rebuilding | rebuild with the same `--set retrieval.key=...` |
| `Bank was built with key_transform='...' but ...` | transform changed without rebuilding | the transform is fitted *into* the stored keys — rebuild with the same `--set retrieval.key_transform=...` |
| `Gate term mask ... != current level configuration` | ablation removed a term the gate was fitted with | refit `train_gate.py` with the same `--set`, or `--set retrieval.gate.enabled=false` |
| `Memory bank not found` | Stage B1 not run for that fold | run §11.1 |
| CUDA OOM | batch too large for 4 GB | `--set data.batch_size=8` |

---

## 12. Retrieval key ablation — what should the memory index on?

Full report and verdict: `docs/results/retrieval_key_ablation.md` (**settled: keep the D1 key**;
43 configurations measured, none beats it). These commands only reproduce it.

```powershell
# Stage 1 — encode each fold once (bank split + val split). ~15 min GPU per encoder.
python scripts/retrieval_key_ablation.py --stage cache                    # exp3n
python scripts/retrieval_key_ablation.py --stage cache --experiment exp1  # cross-encoder control

# Stage 2 — sweep 43 (key, key_transform) configs offline. ~30 min, no training.
python scripts/retrieval_key_ablation.py --stage eval --cross-encoder exp1

# per-row probabilities, for recomputing the report's bootstrap intervals
python scripts/retrieval_key_ablation.py --stage eval --dump-probs --only features
```

Writes `analysis\retrieval_keys\<experiment>\`: `cache_fold<k>.npz` (~91 MB each —
safe to delete, regenerated from the checkpoints), `key_ablation.json`,
`key_ablation.log`. The sweep never touches the test set.

To take one configuration to production instead (the bank records the key **and**
the transform, and the loader refuses a mismatch):

```powershell
python scripts/build_memory_bank.py --experiment exp3n `
  --set retrieval.key=fpn.std --set retrieval.key_transform=whiten:128
python scripts/train_gate.py --experiment exp3n `
  --set retrieval.key=fpn.std --set retrieval.key_transform=whiten:128
```

---

## 13. Foundation-model baseline — CTransPath (Stages F1 → F2 → F3)

the paper: a **frozen** pathology foundation model + a linear
head, on the same splits, the same eval transform and the same metrics as
exp1–exp3n, so it is one more row in the main table. Design notes and the measured
result: `docs/results/foundation_baseline.md`.

Two presets, both fitted by `train_linear_probe.py` (never by `train.py` — that
refuses them loudly):

| preset | run dir | head |
|---|---|---|
| `expfm` | `expfm_ctranspath_linear` | `Dropout → Linear(768 → 2)` — **the reported row** |
| `expfm_mlp` | `expfm_ctranspath_mlp` | `… → Linear(768 → 512) → ReLU → Linear(512 → 2)` |

> **The encoder is frozen and never fine-tuned.** Stage F1 encodes every image
> once; F2/F3 only ever touch a `[N, 768]` cache. That is why a fold fits in ~3
> seconds and why the whole experiment is ~3 minutes end to end, not 3 days.

### 13.1 Stage F1 — cache the frozen features *(~3 min GPU, run ONCE)*

```powershell
python scripts/extract_foundation_features.py
```

Downloads the ungated CTransPath weights on first use (~220 MB, cached under
`~/.cache/huggingface`), then encodes all **7,909** protocol images under the
project's *eval* transform. Writes `analysis\foundation\ctranspath\features.npz`
(22.6 MB) plus `extract_foundation_features.log`.

It prints a sanity table and **raises** if any of it is wrong: 7,909 rows / 82
patients / 768-d, four non-empty magnification shards, no non-finite values,
non-degenerate per-dimension spread, and `183/183 checkpoint tensors identical in
the model ✓` — that last line is the one that matters, because timm loads
pretrained weights *silently* and a dropped block would leave part of the encoder
randomly initialised.

Re-running is refused (the cache is deterministic); pass `--force` to rebuild.
If you hit CUDA OOM:
```powershell
python scripts/extract_foundation_features.py --set foundation.batch_size=16
```

### 13.2 Stage F2 — fit the probe on every fold *(~15 s total)*

```powershell
python scripts/train_linear_probe.py --experiment expfm
```

One fold at a time (identical result):
```powershell
foreach ($k in 0..4) { python scripts/train_linear_probe.py --experiment expfm --fold $k }
```

Each fold gets a normal run directory —
`runs\expfm_ctranspath_linear\<TIMESTAMP>_train_fold<k>\` with `config.yaml`,
`train.log`, `metrics.csv`, `tensorboard\`, `checkpoints\{best,last}.pt` — **plus**
`val_predictions.csv`, which is what lets `threshold_calibration.py` score this
experiment without a model it cannot rebuild (§13.5).

The standardiser and the head are fitted on the fold's **TRAIN rows only**;
`train ∩ val` and `train/val ∩ test` patient disjointness is asserted before a
single feature row is read. Never work around an assertion.

The MLP variant:
```powershell
python scripts/train_linear_probe.py --experiment expfm_mlp
```

### 13.3 Stage F3 — score the 16-patient held-out set *(~5 s)*

```powershell
python scripts/test_linear_probe.py --experiment expfm
```

Per fold:
```powershell
foreach ($k in 0..4) { python scripts/test_linear_probe.py --experiment expfm --fold $k }
```

Writes `test\test_metrics.json` and `test\test_predictions.csv` into each run
directory — **the same filenames and the same columns `scripts/test.py` writes**,
so `diagnose_folds.py` and `threshold_calibration.py` read this experiment with no
special case. It prints a per-fold table and the 5-fold mean.

### 13.4 The whole experiment, start to finish

```powershell
python scripts/extract_foundation_features.py
python scripts/train_linear_probe.py --experiment expfm
python scripts/test_linear_probe.py  --experiment expfm
```

### 13.5 Threshold calibration — required, not optional, for this row

CTransPath is the **most miscalibrated model in the study** (pooled-OOF optimal
image threshold **0.209**, against exp3n's 0.540), so its accuracy at a fixed 0.5
cut badly understates it. Report the locked-threshold number, per protocol P2:

```powershell
python scripts/threshold_calibration.py --experiments expfm --out "analysis/threshold_calibration_expfm"
```

> `--out` is deliberately a separate directory: the default
> `analysis\threshold_calibration\results.json` is the exp1–exp3n artifact and
> re-running it for the whole set costs ~1 h of GPU val inference. To regenerate
> the full comparison table in one go (and only then), run:
> ```powershell
> python scripts/threshold_calibration.py --experiments exp1 exp2 exp3 exp3n expfm
> ```

### 13.6 Per-patient diagnosis (optional, for the tracked-case table)

```powershell
python scripts/diagnose_folds.py --experiment expfm
```

### 13.7 Troubleshooting

| message | cause | fix |
|---|---|---|
| `'expfm' is a frozen foundation-model baseline and cannot be run by scripts/train.py` | wrong script | use §13.2 / §13.3 |
| `Feature cache not found: ...` | Stage F1 not run | run §13.1 |
| `Feature cache already exists` | F1 re-run | it is deterministic; `--force` only if you changed the encoder or the transform |
| `Pretrained weights did not fully land in the model` | timm version changed its Swin key remap | pin `timm==1.0.9`; do **not** relax the check — it is the only thing standing between you and a half-random encoder |
| `Feature cache was built with encoder 'X', but the config asks for 'Y'` | mixed encoders | re-run F1 for the encoder under test |
| `N test patient(s) have no rows in the feature cache` | splits regenerated after F1 | re-run §13.1 |
| `... is not a probe checkpoint (no 'probe' block)` | pointed F3 at a Swin `best.pt` | use `scripts/test.py` |
| CUDA OOM in F1 | batch too large for 4 GB | `--set foundation.batch_size=16` |

---

## Monitoring

```powershell
tensorboard --logdir runs
```
Then open the URL it prints (usually http://localhost:6006).
