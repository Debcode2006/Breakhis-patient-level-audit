# What Actually Moves the Needle on BreaKHis?

**A patient-level audit of magnification conditioning, contrastive learning, retrieval and pretraining.**

Debanjan Sarkar¹, Raju Naskar², Nibaran Das²
¹ Government College of Engineering and Ceramic Technology, Kolkata · ² Jadavpur University, Kolkata

📄 **[Paper (PDF)](paper/PAUL_Conference_2026_Debanjan_Raju.pdf)** ·
📚 **[Documentation](docs/)** ·
🔬 **[Results reports](docs/results/)** ·
⚙️ **[Commands](docs/COMMANDS.md)**

---

## Summary

BreaKHis is the standard benchmark for breast-histopathology classification, but
patient leakage and unconstrained decision thresholds make reported gains hard to
interpret. We evaluate four commonly proposed enhancements — **magnification
conditioning**, **supervised contrastive learning**, **retrieval-augmented memory**,
and **pathology-specific pretraining** — as controlled add-ons to a single
Swin-Tiny + FPN encoder, under a leakage-free, patient-clustered, subtype-stratified
protocol.

**None of them improves discrimination.** Across 16 held-out patients, image-level
AUC stays inside a one-point band and no paired comparison clears zero. What the
methods *do* change is distinct and measurable:

| Finding | Mechanism |
|---|---|
| Magnification conditioning is a **bias term**, not a representation change | The learned block moves logits by one scalar per zoom, spanning a seventh of the image-driven spread. Zeroing it, replacing it by its mean, or forcing every image to one magnification all return the same AUC (0.96050 vs 0.96068 deployed). |
| Contrastive learning improves **geometry**, and accuracy only after the magnification offset is removed | SupCon raises binary silhouette whether or not the block is present (best: 0.680 for Exp3n). With the block on, the space is 100 % magnification-locked. |
| Retrieval memory fails for a **diagnosable** reason | Useful retrieval needs error-diversifying evidence, but even decorrelated neighbours must out-perform a classifier already reading the same features. On identical frozen CTransPath features, a cosine kNN vote (0.9149 AUC) loses to a 1,538-parameter linear layer (0.9415). |
| A frozen pathology encoder shows the **same dissociation** | CTransPath has the best AUC in the study (0.9668) and the worst accuracy at 0.5 (0.8580) — it is the most miscalibrated model measured (locked threshold 0.209). |

The headline is methodological: **on a benchmark this small, evaluation protocol
decides what a reported number means.** A fixed 0.5 threshold measures calibration,
not capability.

---

## Main result

Held-out benchmark — 16 patients, 1,653 images, never seen in training, validation,
model selection or threshold calibration. Point estimates are five-fold means;
thresholds are locked on pooled out-of-fold validation **before** the test set is
touched.

| Configuration | Train. params | Acc @0.5 | Locked thr. | Acc @thr. | Image AUC | Sens. | Spec. | Pat. acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Exp1** baseline | 30.8 M | **0.9054** | 0.450 | **0.9064** | 0.9559 | **0.9242** | 0.8563 | **0.9875** |
| **Exp2** + mag | 30.8 M | 0.8974 | 0.428 | 0.9016 | 0.9550 | 0.9054 | 0.8764 | 0.9625 |
| **Exp3** + mag + SupCon | 30.8 M | 0.8558 | 0.360 | 0.8955 | 0.9607 | 0.8894 | 0.8764 | 0.9375 |
| **Exp3n** + SupCon | 30.8 M | 0.8986 | 0.540 | 0.8981 | 0.9593 | 0.9193 | 0.8445 | 0.9750 |
| ⤷ + retrieval memory | +147 | 0.9057 | 0.692 | 0.8874 | 0.9561 | — | — | 0.9750 |
| **CTransPath** (frozen) | 1,538 | 0.8580 | 0.209 | 0.8985 | **0.9668** | 0.8190 | **0.9590** | 0.9375 |

Paired differences, patient-clustered bootstrap over the 16 held-out patients
(2,000 resamples) — **every interval contains zero**:

| Paired difference | Δ image AUC [95 % CI] | Δ image acc. [95 % CI] |
|---|---|---|
| Exp2 − Exp1 | +0.0032 [−0.0046, +0.0141] | +0.0005 [−0.0064, +0.0057] |
| Exp3 − Exp1 | +0.0038 [−0.0014, +0.0117] | −0.0023 [−0.0154, +0.0092] |
| Exp3n − Exp1 | +0.0029 [−0.0041, +0.0112] | +0.0083 [−0.0135, +0.0344] |
| Exp3n − Exp3 | −0.0009 [−0.0130, +0.0090] | +0.0106 [−0.0176, +0.0475] |
| + retrieval − Exp3n | −0.0003 [−0.0009, +0.0007] | −0.0254 [−0.0599, +0.0028] |
| CTransPath − Exp1 | +0.0028 [−0.0490, +0.0573] | −0.0063 [−0.0836, +0.0755] |

> **Read the accuracy columns carefully.** At a fixed 0.5 cut the spread across
> configurations is four times wider than at each model's own locked threshold. That
> spread is calibration, not discrimination — which is the paper's point.

Full tables, per-fold breakdowns and the per-patient audit:
[`docs/results/`](docs/results/).

---

## Protocol

The protocol is the contribution; it is enforced in code, not by convention.

**Patient-level partitioning.** Each patient contributes 60–235 images of one
specimen, so the patient — never the image — is the atomic unit. Image-level splits
leak near-duplicate views across train/test and inflate accuracy by up to 41 %.

**Held-out set: 16 of 82 patients (19.5 %)**, untouched by training, validation,
model selection *and* threshold calibration. The remaining 66 form five
class-stratified, patient-disjoint folds. Splits are generated **once** under seed 42
by [`scripts/prepare_splits.py`](scripts/prepare_splits.py), serialized with their
policy to [`splits/breakhis_splits.json`](splits/breakhis_splits.json), and reloaded
verbatim. Disjointness is asserted at generation, at every load, and again when the
retrieval bank is built.

**Subtype-stratified test set.** The eight tumour subtypes are long-tailed, so a
class-only split lets rare subtypes clump — one draw pinned image-level test accuracy
near 82 % for every model trained. The held-out set is instead stratified over all
eight subtypes by largest-remainder allocation with two guards: *phyllodes tumour*
(3 patients total) is reserved for the CV pool, and any patient exceeding 12 % of the
test-image budget is rerouted so no single slide dominates. The superseded
class-only split is kept at
[`splits/breakhis_splits.classstrat_backup.json`](splits/) so the change is auditable.

**Thresholds are locked out of fold.** Each model's operating point is chosen on
pooled out-of-fold validation predictions and fixed before the test set is read. A
test-fitted threshold is reported only as an oracle bound.

**Statistics.** Paired AUC and accuracy differences carry 95 % intervals from a
patient-clustered bootstrap over the 16 held-out patients (2,000 resamples). The
effective sample size is the patient count, not the image count: one patient is worth
6.25 points of patient-level accuracy.

### Dataset

[BreaKHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
— 7,909 H&E images, 82 patients, 700×460 RGB, four magnifications
(40× / 100× / 200× / 400× → 1,995 / 2,081 / 2,013 / 1,820 images). Binary
benign/malignant labels derived from eight subtypes. **One unified model is trained
across all four magnifications**; the magnification-unified setting is part of the
protocol, not a post-hoc choice.

The dataset is **not** redistributed here. Download it, then point
`data.dataset_root` in [`config/config.yaml`](config/config.yaml) at your copy.

---

## Architecture

```
SwinBackbone → FPN → FeatureFusion → [MagnificationEmbedding] → [ProjectionHead] + ClassificationHead
```

An ImageNet-pretrained **Swin-Tiny** produces four hierarchical stage maps
(96/192/384/768 channels at strides 4/8/16/32). A **Feature Pyramid Network**
unifies them to 256 channels top-down, each level is GAP-pooled, and the
concatenation is the 1024-d descriptor `z` **shared by every configuration**:

```
z = Concat(GAP(P₁), …, GAP(P₄)) ∈ ℝ¹⁰²⁴
```

Holding `z` fixed makes each enhancement a controlled modification of one model
rather than a separate model. `z` is also defined *before* any magnification
information enters the network, which is what makes it usable as a retrieval key.

`forward()` returns a dict — `{logits, features, embeddings, projections,
fpn_features}` — which is the extension contract. `features` is `z`; `embeddings` is
`z` after magnification fusion; `fpn_features` is the spatial pyramid. New modules
consume the dict and register a loss term in
[`losses/combined.py`](rap_mst/losses/combined.py); the trainer, backbone and FPN do
not change.

> ⚠️ **Never use `embeddings` or `projections` as a retrieval key** on a
> magnification-enabled encoder: 99.6–100 % of their cosine neighbours share the
> query's magnification (chance is 25 %). See
> [`docs/results/magnification_audit.md`](docs/results/magnification_audit.md).

### The experiment ladder

A 2×2 factorial over the two trained enhancements. Switching is one flag —
`--experiment expN` — never a new model class.

| Preset | Magnification embedding | SupCon | Objective |
|---|:---:|:---:|---|
| `exp1` | — | — | ℒ_CE |
| `exp2` | ✓ | — | ℒ_CE |
| `exp3` | ✓ | ✓ | ℒ_CE + λℒ_SupCon |
| `exp3n` | — | ✓ | ℒ_CE + λℒ_SupCon |
| `exp5` | — | ✓ | exp3n's frozen weights **+ retrieval memory** (no encoder training) |
| `expfm` | — | — | frozen CTransPath + linear probe (external baseline, no Swin) |

`exp3n` isolates supervised contrastive learning *without* magnification
conditioning — the cell an incremental ablation never visits, and the base encoder
for the retrieval module. λ = 0.4, τ = 0.07.

### Retrieval-augmented memory

Post-hoc, non-parametric and **two-stage**: the encoder is trained, frozen, and then
used both to write the memory and to answer queries. The only trainable component is
a **147-parameter fusion gate**. No retrieval term enters the training objective, so
any measured difference belongs to the module alone.

One bank per fold holds two levels in one store: an `image` row per training image
(magnification-sharded) and a `slide` centroid per training patient. The two levels
are ranked **independently** by cosine similarity and combined only at the gate — a
single top-k over their union is dominated by the far more numerous image rows.
Votes are temperature-scaled softmax (τ = 0.07) with a cap of 3 neighbours per bank
patient; a query never retrieves its own patient, asserted at write, at load and per
query.

Design, decisions D1–D10, and the operator handbook:
**[`docs/retrieval.md`](docs/retrieval.md)**.

---

## Repository layout

```
rap_mst/                  the package — config-driven builders, no logic in __main__
├── data/                 BreaKHis parsing, patient-level splits, transforms
├── models/               backbone · fpn · fusion · magnification · projection · classifier
├── losses/               CE + SupCon, combined as a weighted sum
├── engine/               trainer (AMP, checkpointing, early stop, resume) · evaluator
├── retrieval/            bank · memory · gate · keys · transform · foreign
├── foundation/           frozen CTransPath encoder, feature cache, linear probe
└── utils/                config · seeding · metrics · checkpoints · research logging

scripts/                  one stage per script; see the map below
config/config.yaml        single source of truth — nothing important is hardcoded
splits/                   the frozen split artifact + the superseded one, for audit
docs/                     protocol, design and every results report
paper/                    the compiled paper
assets/schematics/        hand-authored SVG sources for Figures 1–2 (not regenerable)
images/                   output directory for all rendered figures (git-ignored)
analysis/                 measured artifacts — JSON, logs, gate, probability vectors
runs/                     per-run configs, logs, metrics, checkpoints (git-ignored)
```

### What is not in the repository

Everything excluded is regenerable by a documented command, and
[`.gitignore`](.gitignore) names the script for each:

| Excluded | Size | Rebuilt by |
|---|---|---|
| `runs/` — checkpoints, TensorBoard, per-run logs | ~14 GB | `train.py` (~2 GPU-h/fold), `train_linear_probe.py` (~15 s/fold) |
| `analysis/**/cache_fold*.npz` — key-ablation feature caches | ~860 MB | `retrieval_key_ablation.py --stage cache` |
| `analysis/embeddings/embeddings/*.npz` — test embeddings | ~130 MB | `visualize_embeddings.py` |
| `analysis/retrieval/**/bank_fold*.npz` — memory banks | ~92 MB | `build_memory_bank.py` (~10 min GPU) |
| `analysis/foundation/**/features.npz` — CTransPath cache | ~21 MB | `extract_foundation_features.py` (~3 min GPU) |
| `images/*` — rendered figures | ~5 MB | `render_svg_figures.py`, `make_panels.py`, `assemble_figures.py` |

The 147-parameter fusion gate (`analysis/retrieval/exp3n/gate.pt`, 4.6 KB) **is**
committed — without it the retrieval results cannot be reproduced from the committed
artifacts alone.

---

## Installation

```bash
conda create -n rapmst python=3.10 -y
conda activate rapmst

# Install a CUDA build of PyTorch first (not from requirements.txt):
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Then set `data.dataset_root` in [`config/config.yaml`](config/config.yaml) to your
BreaKHis copy.

**Reference hardware:** one NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM. That budget is
why mixed precision and gradient accumulation are on by default (batch 16 × 2 steps
= effective 32). On OOM:
`--set data.batch_size=8 --set train.grad_accum_steps=4` and/or
`--set model.fpn.out_channels=128`.

---

## Reproducing the study

Full copy-paste commands with expected outputs, timings and troubleshooting:
**[`docs/COMMANDS.md`](docs/COMMANDS.md)**.

```bash
# 0. Splits — run once. Everything downstream reloads this file verbatim.
python scripts/prepare_splits.py --config config/config.yaml

# 1. The ladder — five folds each, ~2 GPU-hours per fold.
python scripts/train.py --experiment exp1  --fold 0        # ... folds 0..4
python scripts/train.py --experiment exp3n --fold 0        # exp2, exp3, exp3n likewise

# 2. Held-out test, per fold.
python scripts/test.py --checkpoint "runs/exp1_swin_cls/<RUN>/checkpoints/best.pt"

# 3. Retrieval memory: build banks -> fit the gate once -> test.
python scripts/build_memory_bank.py --experiment exp3n
python scripts/train_gate.py        --experiment exp3n
python scripts/test.py --checkpoint "runs/exp3n_swin_supcon_cls/<RUN>/checkpoints/best.pt" --retrieval

# 4. Frozen CTransPath baseline (~3 min total).
python scripts/extract_foundation_features.py
python scripts/train_linear_probe.py --experiment expfm
python scripts/test_linear_probe.py  --experiment expfm

# 5. Locked thresholds and patient-clustered intervals for the main table.
python scripts/threshold_calibration.py --experiments exp1 exp2 exp3 exp3n expfm
python scripts/bootstrap_benchmark.py

# 6. Figures.
python scripts/render_svg_figures.py                       # Figures 1-2 from SVG
python scripts/precompute_umap_cache.py                    # fit Figure 4's UMAPs once
python scripts/assemble_figures.py                         # Figures 3-5 -> images/
```

> **Do not mix encoders.** The bank, the gate and the retrieval test run must all
> come from `exp3n`. Each bank records the encoder, fold and key that produced it and
> the loader refuses a mismatch, so a mistake raises rather than quietly producing
> numbers.

### Script → paper artifact

Every script in [`scripts/`](scripts/) produces something the paper reports. The
`retrieval_probe*` scripts are the design record: they are what fixed the module's
key, routing, vote rule, granularity pair and gate, and probe 4 is the data source
for Figure 3.

| Script | Produces |
|---|---|
| `prepare_splits.py` | The split artifact · Table 1 · §3.2 |
| `train.py` | Exp1 / Exp2 / Exp3 / Exp3n encoders · §3.6 |
| `test.py` | Table 3 rows (`--retrieval` adds the memory row) · §4.1, §4.4 |
| `build_memory_bank.py`, `train_gate.py` | The memory and its gate · §3.5, Fig. 2 |
| `extract_foundation_features.py`, `train_linear_probe.py`, `test_linear_probe.py` | The frozen CTransPath row · §3.8, §4.5 |
| `threshold_calibration.py` | Locked thresholds in Tables 3 and 4 · §3.6 |
| `bootstrap_benchmark.py` | Patient-clustered CIs under Table 3 · §3.7 |
| `visualize_embeddings.py` | Silhouette / kNN / same-mag rate · Table 4 |
| `precompute_umap_cache.py` | Figure 4's cached UMAP coordinates |
| `diagnose_folds.py` | Per-patient error audit · §4.5 |
| `retrieval_probe.py` | Key space, hubness, prototypes (D1) |
| `retrieval_probe2.py` | Complementarity ceiling, routing, vote rules (D2–D4) |
| `retrieval_probe3.py` | Learned metrics, two-level blend (D5, D7) |
| `retrieval_probe4.py` | **Figure 3** — magnification counterfactuals · §4.2 |
| `retrieval_probe5.py` | Key-space geometry behind the diagnosis · §4.4 |
| `retrieval_key_ablation.py` | The 43 key definitions · §4.4 |
| `retrieval_crossencoder_screen.py` | **Figure 5**, Table 5 · §4.6 |
| `paper_style.py`, `figure_panels.py` | The single plotting style and the six raw panels |
| `make_panels.py`, `assemble_figures.py` | Panels standalone and assembled into Figures 3–5 |
| `render_svg_figures.py` | Figures 1–2 from the hand-authored SVG |

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/COMMANDS.md`](docs/COMMANDS.md) | Every command, with expected output and timings |
| [`docs/retrieval.md`](docs/retrieval.md) | Retrieval memory — evidence, decisions D1–D10, architecture, handbook, and the diagnosis of why it does not help |
| [`docs/FIGURE_STYLE.md`](docs/FIGURE_STYLE.md) | The figure house style; `scripts/paper_style.py` is its executable form |
| [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) | The full chronological record — decision register, what did not work and why |

**Results reports** — each is self-contained and cites its own artifacts:

| Report | Question it closes |
|---|---|
| [`classifier_ladder.md`](docs/results/classifier_ladder.md) | Cross-experiment fold and test analysis for Exp1–Exp3n |
| [`threshold_calibration.md`](docs/results/threshold_calibration.md) | Does a validation-tuned threshold recover the accuracy AUC says is there? |
| [`embedding_geometry.md`](docs/results/embedding_geometry.md) | UMAP and separation metrics — direct evidence that SupCon reshapes the space |
| [`magnification_audit.md`](docs/results/magnification_audit.md) | What the magnification block does, and why the base encoder is Exp3n |
| [`retrieval_heldout.md`](docs/results/retrieval_heldout.md) | The memory on the held-out test set |
| [`retrieval_key_ablation.md`](docs/results/retrieval_key_ablation.md) | 43 key definitions — the key is not the bottleneck |
| [`crossencoder_screen.md`](docs/results/crossencoder_screen.md) | Does an un-shared encoder fix retrieval? (pooled out-of-fold; verdict: no) |
| [`foundation_baseline.md`](docs/results/foundation_baseline.md) | Frozen CTransPath + linear probe |

---

## Reproducibility

- `seed_everything` covers Python, NumPy and Torch; workers are seeded; cuDNN
  determinism is a config toggle.
- Checkpoints carry full state — model, optimizer, scheduler, AMP scaler and RNG —
  so a resumed run continues the same trajectory.
- Every run writes its resolved config, logs, `metrics.csv` and TensorBoard events to
  its own timestamped directory. Runs are never overwritten.
- Metrics are reported at both image level and patient level (mean-pooled per
  patient). The default checkpoint and early-stopping monitor is patient accuracy.
- The research logging layer ([`utils/reporting.py`](rap_mst/utils/reporting.py))
  emits environment and architecture reports, a one-time forward-shape trace,
  per-epoch resource lines, and automatic WARNING banners for NaN/Inf, exploding
  gradients, projection collapse and validation collapse. The retrieval module's
  "unified store, two independent rankings" invariant is verified at runtime on the
  first batch, not only in a test.
- Protocol violations fail loudly. Leakage assertions are not warnings, and working
  around one invalidates every number downstream.

---

## Limitations

Sixteen held-out patients quantise patient accuracy at 6.25 points, which is why
image-level results with patient-clustered intervals are the primary reading. The
study covers one dataset, 82 patients and a binary task. Training was 10–15 epochs on
a single 4 GB GPU, which also left the contrastive two-view path unused. The
cross-encoder screen (§4.6) is out of fold by design, uses one foundation encoder,
and mixes probabilities convexly.

## Future work

Because the binding constraint is the **read-out** rather than the bank, the next
step is a learned combiner over several frozen encoders rather than a better key.
Three directions follow: an encoder trained *for* complementarity; feature-wise
magnification modulation tested against the one-point oracle headroom of §4.2; and
subtype-aware contrastive learning. Each fits additively into the forward-dict
contract above.

---

## Citation

```bibtex
@inproceedings{sarkar2026breakhis,
  title     = {What Actually Moves the Needle on {BreaKHis}? A Patient-Level Audit
               of Magnification, Contrastive Learning, Retrieval and Pretraining},
  author    = {Sarkar, Debanjan and Naskar, Raju and Das, Nibaran},
  year      = {2026}
}
```

## License

Code is released under the [MIT License](LICENSE). The BreaKHis dataset and the
CTransPath weights are distributed by their respective authors under their own
terms and are not redistributed here.

## Acknowledgements

BreaKHis is provided by the P&D Laboratory, Universidade Federal do Paraná.
CTransPath weights are used via an ungated Hugging Face mirror. The Swin Transformer
implementation comes from [`timm`](https://github.com/huggingface/pytorch-image-models).
