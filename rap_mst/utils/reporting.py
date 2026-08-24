"""Research reporting toolkit -- the narrative layer over the plain logger.

This module does NOT introduce a second logging backend. It formats structured,
consistent "reports" (startup, architecture, dataset, per-epoch resources,
SupCon diagnostics, checkpoints, testing, final summary) and writes them through
the *existing* ``logging.Logger`` created by :func:`rap_mst.utils.logging_utils.get_logger`.
That keeps console, file, and (via the trainer) TensorBoard telling one story.

Design goals
------------
* **Extend, don't replace.** Everything here is additive; the trainer still owns
  the CSV / TensorBoard writers and the checkpoint format.
* **One place for the vocabulary.** Section banners, warning banners, version
  collection, resource sampling, and numeric guards live here so the training
  and testing scripts stay readable and future modules reuse the same look.
* **Ready for future modules.** :class:`DiagnosticRegistry` lets Retrieval Memory
  / Prototype Learning / Reasoning modules register their own per-epoch
  diagnostics without editing the training loop -- the trainer simply collects
  whatever is registered and logs it alongside the built-in metrics.

Nothing here imports the trainer, so it is safe to use from scripts and modules
alike.
"""

from __future__ import annotations

import logging
import math
import platform
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

# --------------------------------------------------------------------------- #
# Warning thresholds (single source of truth for the automatic guards)
# --------------------------------------------------------------------------- #
GRAD_NORM_WARN = 1.0e3        # gradient global norm above this is "exploding"
SUPCON_LOSS_WARN = 50.0       # SupCon term above this is "exploding"
PROJ_VAR_WARN = 1.0e-4        # projection variance below this ~= collapse
EMB_NORM_WARN = 1.0e-3        # near-zero embedding norm is suspicious
VAL_COLLAPSE_DROP = 0.25      # monitored metric dropping this much in one epoch

_WIDTH = 74


# --------------------------------------------------------------------------- #
# Formatting primitives
# --------------------------------------------------------------------------- #
def _rule(char: str = "─") -> str:
    return char * _WIDTH


def section(logger: logging.Logger, title: str, rows: Sequence[str] = ()) -> None:
    """Emit a titled report block as a single (multi-line) INFO record."""
    lines = [_rule("═"), f" {title}", _rule("═")]
    lines.extend(rows)
    lines.append(_rule())
    logger.info("\n".join(lines))


def kv(label: str, value: Any) -> str:
    """Format one ``label : value`` row with aligned columns."""
    return f"  {label:<26}: {value}"


def warn_banner(logger: logging.Logger, title: str, *details: str) -> None:
    """Emit a highly visible WARNING block that is hard to miss in the log."""
    lines = ["", "!" * _WIDTH, f"  WARNING: {title}"]
    lines.extend(f"  - {d}" for d in details)
    lines.append("!" * _WIDTH)
    logger.warning("\n".join(lines))


# --------------------------------------------------------------------------- #
# Environment / version collection
# --------------------------------------------------------------------------- #
def collect_env(device: torch.device) -> Dict[str, Any]:
    """Gather the versions / hardware facts that make a run reproducible."""
    try:
        import timm

        timm_version = timm.__version__
    except Exception:  # pragma: no cover - timm always present in real runs
        timm_version = "n/a"

    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "timm": timm_version,
        "cuda_build": torch.version.cuda or "cpu-only",
        "cudnn": torch.backends.cudnn.version() if cuda_available else "n/a",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else "n/a",
        "num_gpus": torch.cuda.device_count() if cuda_available else 0,
    }


# --------------------------------------------------------------------------- #
# Resource sampling
# --------------------------------------------------------------------------- #
def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def gpu_memory_mb(device: torch.device) -> Dict[str, float]:
    """Return allocated / reserved / peak GPU memory in MiB (empty on CPU)."""
    if device.type != "cuda":
        return {}
    mb = 1024.0 * 1024.0
    return {
        "gpu_alloc_mb": torch.cuda.memory_allocated(device) / mb,
        "gpu_reserved_mb": torch.cuda.memory_reserved(device) / mb,
        "gpu_peak_mb": torch.cuda.max_memory_allocated(device) / mb,
    }


def cpu_ram_mb() -> Optional[float]:
    """Resident memory of this process in MiB, or ``None`` if psutil is absent."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Model introspection
# --------------------------------------------------------------------------- #
def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def grad_global_norm(model: torch.nn.Module) -> float:
    """L2 norm of all gradients (call *after* ``scaler.unscale_``)."""
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += float(p.grad.detach().norm(2)) ** 2
    return math.sqrt(total_sq)


def _fmt_params(n: int) -> str:
    return f"{n:,} ({n / 1e6:.2f}M)"


# --------------------------------------------------------------------------- #
# Report: startup
# --------------------------------------------------------------------------- #
def report_startup(logger: logging.Logger, cfg, device: torch.device, fold: int) -> None:
    env = collect_env(device)
    amp = bool(cfg.train.mixed_precision) and device.type == "cuda"
    rows = [
        kv("Experiment", cfg.experiment.name),
        kv("Fold", fold),
        kv("Seed", cfg.seed),
        kv("Deterministic", cfg.deterministic),
        kv("Mixed precision (AMP)", amp),
        _rule(),
        kv("Device", env["device"]),
        kv("GPU", env["gpu_name"]),
        kv("Num GPUs", env["num_gpus"]),
        kv("CUDA (torch build)", env["cuda_build"]),
        kv("cuDNN", env["cudnn"]),
        _rule(),
        kv("PyTorch", env["torch"]),
        kv("timm", env["timm"]),
        kv("Python", env["python"]),
        kv("Platform", env["platform"]),
    ]
    section(logger, "RAP-MST RUN START", rows)


# --------------------------------------------------------------------------- #
# Report: architecture
# --------------------------------------------------------------------------- #
def report_architecture(logger: logging.Logger, model: torch.nn.Module) -> None:
    def mark(active: bool) -> str:
        return "✓" if active else "✗"

    # Implemented, optional-but-present components are introspected from the
    # model; the future modules are declared here so the report always shows the
    # full intended pipeline with the not-yet-built stages struck through.
    components = [
        ("Swin Backbone", True),
        ("Feature Pyramid Network", getattr(model, "uses_fpn", model.fpn is not None)),
        ("Feature Fusion", True),
        ("Magnification Embedding", getattr(model, "uses_magnification", False)),
        ("Projection Head", getattr(model, "uses_projection", False)),
        ("Retrieval Memory", False),   # future work
        ("Prototype Learning", False),  # future work
        ("Reasoning Module", False),    # future work
    ]
    rows = [f"  {mark(active)} {name}" for name, active in components]

    params = count_parameters(model)
    rows.append(_rule())
    rows.append(kv("Total parameters", _fmt_params(params["total"])))
    rows.append(kv("Trainable parameters", _fmt_params(params["trainable"])))
    if params["frozen"]:
        rows.append(kv("Frozen parameters", _fmt_params(params["frozen"])))
    section(logger, "ARCHITECTURE", rows)


# --------------------------------------------------------------------------- #
# Report: dataset (+ hard leakage guard)
# --------------------------------------------------------------------------- #
def _distribution(samples) -> Dict[str, Dict[str, int]]:
    cls: Dict[str, int] = {}
    mag: Dict[int, int] = {}
    for s in samples:
        cls[s.class_name] = cls.get(s.class_name, 0) + 1
        mag[s.magnification] = mag.get(s.magnification, 0) + 1
    return {"class": cls, "mag": mag}


def report_dataset(logger: logging.Logger, dm, fold: int) -> None:
    """Report dataset provenance for the active fold and re-verify no leakage.

    Raises
    ------
    RuntimeError
        If any patient appears in more than one of {train, val, test}. The split
        files are already leakage-checked, but re-asserting here fails loudly if
        an in-memory filter ever went wrong.
    """
    splits = dm.splits
    train_p = set(dm.train_set_patient_ids())
    val_p = set(dm.val_set_patient_ids())
    test_p = set(splits["test_patients"])

    # --- Hard leakage guard: stop the run if patients overlap. --------------- #
    overlaps = {
        "train/val": train_p & val_p,
        "train/test": train_p & test_p,
        "val/test": val_p & test_p,
    }
    leaked = {k: v for k, v in overlaps.items() if v}
    if leaked:
        warn_banner(
            logger,
            "PATIENT LEAKAGE DETECTED -- aborting run",
            *[f"{pair}: {sorted(pids)}" for pair, pids in leaked.items()],
        )
        raise RuntimeError(f"Patient leakage across splits: {leaked}")

    train_dist = _distribution(dm.train_set.samples)
    val_dist = _distribution(dm.val_set.samples)
    n_train_img = len(dm.train_set)
    n_val_img = len(dm.val_set)

    balance = splits.get("class_balance", {})
    rows = [
        kv("Dataset path", dm.cfg.data.dataset_root),
        kv("Total images scanned", len(dm._samples)),
        kv("Total patients", splits.get("total_patients", "n/a")),
        kv("Patient class balance", f"benign={balance.get('benign', '?')}, "
                                    f"malignant={balance.get('malignant', '?')}"),
        kv("Held-out test patients", f"{len(test_p)} patients"),
        _rule(),
        kv("Current fold", fold),
        kv("Train patients", f"{len(train_p)}"),
        kv("Val patients", f"{len(val_p)}"),
        kv("Train images", f"{n_train_img}  {train_dist['class']}"),
        kv("Val images", f"{n_val_img}  {val_dist['class']}"),
        kv("Train magnifications", train_dist["mag"]),
        kv("Val magnifications", val_dist["mag"]),
        _rule(),
        "  ✓ No patient leakage detected (train ∩ val ∩ test = ∅).",
    ]
    section(logger, "DATASET", rows)


# --------------------------------------------------------------------------- #
# Report: first forward pass shapes (via temporary forward hooks)
# --------------------------------------------------------------------------- #
def _shape_str(out: Any) -> str:
    if isinstance(out, torch.Tensor):
        return str(list(out.shape))
    if isinstance(out, (list, tuple)):
        return "[" + ", ".join(str(list(t.shape)) for t in out if isinstance(t, torch.Tensor)) + "]"
    return str(type(out).__name__)


class ForwardShapeReporter:
    """Capture and print tensor shapes flowing through the model -- exactly once.

    Registers forward hooks on the named submodules, captures their output shapes
    on the *first* real forward pass (zero extra compute), prints the flow, then
    removes itself. Because it hooks by module, adding a future module means the
    reporter picks it up automatically once it is inserted into the model.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.shapes: Dict[str, Any] = {}
        self.handles: List[Any] = []
        self._reported = False

        self._hook("backbone", getattr(model, "backbone", None))
        self._hook("fpn", getattr(model, "fpn", None))
        self._hook("fusion", getattr(model, "fusion", None))
        self._hook("magnification", getattr(model, "magnification", None))
        self._hook("projection", getattr(model, "projection", None))
        self._hook("classifier", getattr(model, "classifier", None))

    def _hook(self, name: str, module: Optional[torch.nn.Module]) -> None:
        if module is None:
            return

        def _capture(_m, _inp, out, _name=name):
            self.shapes.setdefault(_name, out)

        self.handles.append(module.register_forward_hook(_capture))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def maybe_report(self, logger: logging.Logger, input_tensor: torch.Tensor) -> None:
        """Print the captured flow once, then detach the hooks."""
        if self._reported:
            return
        self._reported = True

        arrow = "        ↓"
        rows: List[str] = [f"  Input                    {list(input_tensor.shape)}"]

        def add(label: str, key: str) -> None:
            if key in self.shapes:
                rows.append(arrow)
                rows.append(f"  {label:<24} {_shape_str(self.shapes[key])}")

        # Backbone emits a list of stage maps -> label each stage explicitly.
        if "backbone" in self.shapes and isinstance(self.shapes["backbone"], (list, tuple)):
            for i, m in enumerate(self.shapes["backbone"]):
                rows.append(arrow)
                rows.append(f"  Swin Stage {i + 1:<13} {list(m.shape)}")
        add("FPN Outputs", "fpn")
        add("Feature Fusion", "fusion")
        add("Magnification Embedding", "magnification")
        add("Projection Head", "projection")
        add("Classifier", "classifier")

        section(logger, "FIRST FORWARD PASS (shape trace)", rows)
        self.remove()


# --------------------------------------------------------------------------- #
# SupCon diagnostics + numeric guards
# --------------------------------------------------------------------------- #
def supcon_diagnostics(outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, float]:
    """Per-batch SupCon health signals (embedding geometry + pair counts).

    Returns an empty dict when the model produced no projections (exp1/exp2).
    """
    proj = outputs.get("projections")
    if proj is None:
        return {}

    proj = proj.detach().float()
    if proj.dim() == 3:  # [B, n_views, D] two-view SupCon -> flatten the view axis
        proj = proj.reshape(-1, proj.shape[-1])
    emb = outputs["embeddings"].detach().float()

    # Positive / negative pair counts from the label equality matrix.
    lab = labels.detach().view(-1, 1)
    same = torch.eq(lab, lab.T)
    b = lab.shape[0]
    pos_pairs = int(same.sum().item()) - b  # exclude the diagonal (self)
    neg_pairs = b * b - b - pos_pairs

    return {
        "supcon_proj_var": float(proj.var(dim=0).mean()),
        "supcon_proj_norm": float(proj.norm(dim=1).mean()),
        "supcon_emb_norm": float(emb.norm(dim=1).mean()),
        "supcon_pos_pairs": float(pos_pairs),
        "supcon_neg_pairs": float(neg_pairs),
    }


def check_supcon_health(logger: logging.Logger, epoch: int, diag: Dict[str, float],
                        supcon_loss: float) -> None:
    """Raise WARNING banners for the classic SupCon failure modes."""
    if not diag:
        return
    if not math.isfinite(supcon_loss):
        warn_banner(logger, f"[epoch {epoch + 1}] SupCon loss is {supcon_loss} (NaN/Inf)")
    elif supcon_loss > SUPCON_LOSS_WARN:
        warn_banner(logger, f"[epoch {epoch + 1}] SupCon loss exploding",
                    f"supcon={supcon_loss:.2f} (> {SUPCON_LOSS_WARN})")

    var = diag.get("supcon_proj_var", 1.0)
    if var < PROJ_VAR_WARN:
        warn_banner(logger, f"[epoch {epoch + 1}] Projection collapse suspected",
                    f"projection variance={var:.2e} (< {PROJ_VAR_WARN:.0e})",
                    "all projections are nearly identical; SupCon has no signal.")
    if diag.get("supcon_emb_norm", 1.0) < EMB_NORM_WARN:
        warn_banner(logger, f"[epoch {epoch + 1}] Near-zero embedding norm",
                    f"mean embedding norm={diag['supcon_emb_norm']:.2e}")
    if diag.get("supcon_pos_pairs", 1) == 0:
        warn_banner(logger, f"[epoch {epoch + 1}] No positive pairs in SupCon batches",
                    "every sample had a unique label; contrastive loss is undefined.")


def check_numeric(logger: logging.Logger, epoch: int, train_stats: Dict[str, float],
                  grad_norm: float, lr: float) -> None:
    """Guard the shared numeric failure modes (loss / grad / lr)."""
    total = train_stats.get("train_total", 0.0)
    if not math.isfinite(total):
        warn_banner(logger, f"[epoch {epoch + 1}] Training loss is not finite",
                    f"train_total={total}")
    if math.isfinite(grad_norm) and grad_norm > GRAD_NORM_WARN:
        warn_banner(logger, f"[epoch {epoch + 1}] Very large gradient norm",
                    f"grad_norm={grad_norm:.1f} (> {GRAD_NORM_WARN:.0f}) -- exploding gradients?")
    if not math.isfinite(lr) or lr <= 0.0:
        warn_banner(logger, f"[epoch {epoch + 1}] Learning rate is invalid", f"lr={lr}")


def check_val_collapse(logger: logging.Logger, epoch: int, monitor: str,
                       current: float, previous: Optional[float]) -> None:
    if previous is None or current is None:
        return
    if math.isfinite(previous) and (previous - current) >= VAL_COLLAPSE_DROP:
        warn_banner(logger, f"[epoch {epoch + 1}] Validation metric collapsed",
                    f"{monitor}: {previous:.4f} -> {current:.4f} "
                    f"(drop {previous - current:.4f} >= {VAL_COLLAPSE_DROP})")


# --------------------------------------------------------------------------- #
# Report: checkpoint save / resume
# --------------------------------------------------------------------------- #
def report_checkpoint(logger: logging.Logger, *, path, reason: str, monitor: str,
                      current: float, best: float, epoch: int) -> None:
    logger.info(
        f"[checkpoint] epoch {epoch + 1} | {reason} | "
        f"{monitor}={current:.4f} | best={best:.4f} -> {path}"
    )


def report_resume(logger: logging.Logger, *, path, restored: Dict[str, bool],
                  epoch: int, best_metric: float, monitor: str) -> None:
    rows = [
        kv("Checkpoint", path),
        kv("Resumed at epoch", epoch),
        kv(f"Best {monitor} so far", f"{best_metric:.4f}"),
        _rule(),
        kv("Model weights restored", restored.get("model", False)),
        kv("Optimizer restored", restored.get("optimizer", False)),
        kv("Scheduler restored", restored.get("scheduler", False)),
        kv("AMP scaler restored", restored.get("scaler", False)),
        kv("RNG state restored", restored.get("rng", False)),
    ]
    section(logger, "RESUME", rows)


# --------------------------------------------------------------------------- #
# Report: testing
# --------------------------------------------------------------------------- #
def report_test_header(logger: logging.Logger, *, checkpoint, cfg, fold, n_patients,
                       n_images) -> None:
    rows = [
        kv("Checkpoint", checkpoint),
        kv("Experiment", cfg.experiment.name),
        kv("Fold (trained on)", fold),
        kv("Held-out test patients", n_patients),
        kv("Test images", n_images),
    ]
    section(logger, "HELD-OUT TEST", rows)


def report_test_metrics(logger: logging.Logger, metrics: Dict[str, float], out_dir) -> None:
    image_rows = [kv(k, f"{metrics[k]:.4f}") for k in sorted(metrics) if not k.startswith("patient_")]
    patient_rows = [kv(k, f"{metrics[k]:.4f}") for k in sorted(metrics) if k.startswith("patient_")]
    rows = ["  -- Image-level --", *image_rows, _rule(), "  -- Patient-level --", *patient_rows,
            _rule(), kv("Output directory", out_dir)]
    section(logger, "TEST METRICS", rows)


# --------------------------------------------------------------------------- #
# Report: retrieval memory (docs/retrieval.md §10.6)
# --------------------------------------------------------------------------- #
#: Retrieval health thresholds. The baselines come from the measurements in
#: `docs/retrieval.md` / `docs/retrieval.md`, so these fire on genuinely odd runs
#: rather than on the normal BreaKHis regime (mean top-1 cosine ~0.998 is
#: *expected* here -- neighbouring fields of different patients really are that
#: similar -- so only a near-exact 1.0 is treated as suspicious).
RETRIEVAL_TOP1_SUSPICIOUS = 0.99999   # ~identical vector -> possible duplicate/leak
RETRIEVAL_TOP1_LOW = 0.50             # nothing in the bank resembles the query
RETRIEVAL_SHORT_NEIGHBOURHOOD = 0.05  # fraction of queries that got fewer than k
RETRIEVAL_MIN_DISTINCT_PATIENTS = 2.0 # per-patient cap should spread the evidence
RETRIEVAL_MAG_LOCK = 0.90             # same-mag rate under a non-same_mag route
RETRIEVAL_GATE_COLLAPSE = 0.90        # mean w_param above this = memory unused


def retrieval_diagnostics(raw: Dict[str, Any], k: Optional[int] = None,
                          log_subtypes: bool = True,
                          route: Optional[str] = None) -> Dict[str, Any]:
    """Health + interpretability signals from a retrieval-enabled ``evaluate()``.

    Neighbourhood size, top-1 similarity, agreement, distinct-patient count (the
    per-patient cap's diagnostic), the same-magnification rate (the D1/D3 canary --
    ~0.25 is chance, ~1.00 means the key is magnification-locked) and the
    retrieved-subtype histogram. Subtypes are recorded for *interpretability only*,
    never as a vote weight (D8), and can be switched off with
    ``retrieval.diagnostics.log_retrieved_subtypes``.

    Returned as a plain dict so it can be JSON-dumped by ``scripts/test.py``, fed to
    :func:`check_retrieval_health`, or -- when a future version retrieves during
    training -- wrapped in a closure and handed to
    ``trainer.diagnostics.register(name, fn)`` with no training-loop edits.
    """
    import numpy as np

    def _mean(key: str) -> float:
        values = raw.get(key, [])
        return float(np.mean(values)) if len(values) else float("nan")

    def _cells(key: str):
        return [[t for t in str(cell).split(";") if t] for cell in raw.get(key, [])]

    neighbours = _cells("neighbour_patients")
    sizes = np.asarray([len(n) for n in neighbours], dtype=float) if neighbours else np.zeros(0)

    mags = _cells("neighbour_mags")
    query_mags = raw.get("magnification", [])
    same_mag_rate = float("nan")
    if mags and len(query_mags) == len(mags):
        per_query = [
            np.mean([str(m) == str(q) for m in row]) for row, q in zip(mags, query_mags) if row
        ]
        same_mag_rate = float(np.mean(per_query)) if per_query else float("nan")

    diag: Dict[str, Any] = {
        "route": route,
        "n_queries": int(len(raw.get("label", []))),
        "mean_neighbours": float(sizes.mean()) if sizes.size else float("nan"),
        "short_neighbourhood_fraction": (
            float((sizes < k).mean()) if (sizes.size and k) else 0.0
        ),
        "mean_top1_similarity": _mean("top1_sim"),
        "mean_neighbour_agreement": _mean("agreement"),
        "mean_distinct_patients": _mean("n_distinct_patients"),
        "same_magnification_rate": same_mag_rate,
        "mean_gate_weights": {
            "param": _mean("w_param"), "img": _mean("w_img"), "slide": _mean("w_slide"),
        },
    }

    # --- the two levels, side by side (D5) ------------------------------- #
    # If the slide view were merged away (or disabled) p_slide would be a constant
    # 0.5 and the disagreement rate 0: these two numbers make the "one bank, two
    # rankings" claim checkable from the log rather than taken on trust.
    slide_rows = _cells("slide_patients")
    p_img = np.asarray(raw.get("prob_img", []), dtype=float)
    p_slide = np.asarray(raw.get("prob_slide", []), dtype=float)
    if p_img.size and p_img.size == p_slide.size:
        diag["levels"] = {
            "mean_slide_neighbours": (
                float(np.mean([len(r) for r in slide_rows])) if slide_rows else 0.0
            ),
            "p_slide_std": float(p_slide.std()),
            "img_slide_disagreement": float(
                ((p_img >= 0.5) != (p_slide >= 0.5)).mean()
            ),
            "mean_abs_p_img_minus_p_slide": float(np.abs(p_img - p_slide).mean()),
        }

    if log_subtypes:
        hist: Dict[str, int] = {}
        for row in _cells("neighbour_subtypes"):
            for s in row:
                hist[s] = hist.get(s, 0) + 1
        total = sum(hist.values()) or 1
        diag["retrieved_subtype_histogram"] = {
            key: {"count": val, "fraction": round(val / total, 4)}
            for key, val in sorted(hist.items(), key=lambda kv: -kv[1])
        }
    return diag


def check_retrieval_health(logger: logging.Logger, diag: Dict[str, Any], *,
                           k: Optional[int] = None, route: str = "same_mag",
                           gate_learned: bool = True) -> None:
    """Emit WARNING banners for the retrieval failure modes the docs enumerate.

    Mirrors :func:`check_supcon_health` / :func:`check_numeric`: the run tells you
    when something is wrong instead of leaving it to be discovered in the metrics.
    Every threshold is a module constant above, with its measured justification.
    """
    import math

    def _finite(value) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(value)

    short = diag.get("short_neighbourhood_fraction", 0.0)
    if k and _finite(short) and short > RETRIEVAL_SHORT_NEIGHBOURHOOD:
        warn_banner(
            logger, "Many queries retrieved fewer than k neighbours",
            f"{short:.1%} of queries got < {k} neighbours "
            f"(mean {diag.get('mean_neighbours', float('nan')):.1f}).",
            "The routing may be too strict for this bank, or a magnification shard is thin.",
        )

    distinct = diag.get("mean_distinct_patients", float("nan"))
    if _finite(distinct) and distinct < RETRIEVAL_MIN_DISTINCT_PATIENTS:
        warn_banner(
            logger, "Neighbourhoods are dominated by very few bank patients",
            f"mean distinct patients per query = {distinct:.2f}.",
            "Check retrieval.levels.image.per_patient_cap (D4): the cap exists because "
            "BreaKHis slides contribute 60-235 near-identical fields each.",
        )

    top1 = diag.get("mean_top1_similarity", float("nan"))
    if _finite(top1) and top1 > RETRIEVAL_TOP1_SUSPICIOUS:
        warn_banner(
            logger, "Top-1 similarity is essentially 1.0",
            f"mean top-1 cosine = {top1:.6f}.",
            "A query is retrieving a near-identical vector -- check the bank does not "
            "contain the evaluated patients (expected on this dataset: ~0.998).",
        )
    elif _finite(top1) and top1 < RETRIEVAL_TOP1_LOW:
        warn_banner(
            logger, "Nothing in the bank resembles the queries",
            f"mean top-1 cosine = {top1:.4f}.",
            "Likely a key mismatch between the bank and the encoder (retrieval.key).",
        )

    same_mag = diag.get("same_magnification_rate", float("nan"))
    if _finite(same_mag) and route != "same_mag" and same_mag > RETRIEVAL_MAG_LOCK:
        warn_banner(
            logger, "Retrieval looks magnification-locked",
            f"{same_mag:.1%} of neighbours share the query's magnification under "
            f"route='{route}' (chance is ~25%).",
            "This is the signature of indexing `embeddings` instead of `features` "
            "(docs/retrieval.md D1).",
        )

    w_param = (diag.get("mean_gate_weights") or {}).get("param", float("nan"))
    if gate_learned and _finite(w_param) and w_param > RETRIEVAL_GATE_COLLAPSE:
        warn_banner(
            logger, "The gate has collapsed onto the parametric head",
            f"mean w_param = {w_param:.3f}: the memory is barely contributing.",
            "docs/retrieval.md §6 allows this (alpha is a property of the encoder) -- report it "
            "as 'the module does not help on this encoder' rather than tuning until it does.",
        )


def report_retrieval_config(logger: logging.Logger, rcfg, *, title: str,
                            extra: Sequence[str] = ()) -> None:
    """Echo the retrieval settings a stage ran with, so the log is self-describing."""
    levels = getattr(rcfg, "levels", None)

    def level_row(name: str) -> str:
        node = getattr(levels, name, None) if levels is not None else None
        if node is None:
            return kv(f"{name} view", "(missing from config)")
        state = "" if getattr(node, "enabled", True) else "   [DISABLED]"
        return kv(
            f"{name} view",
            f"route={getattr(node, 'route', '?')} k={getattr(node, 'k', '?')} "
            f"cap={getattr(node, 'per_patient_cap', '?')} "
            f"T={getattr(node, 'temperature', '?')}{state}",
        )

    gate = getattr(rcfg, "gate", None)
    rows = [
        kv("key (D1)", getattr(rcfg, "key", "?")),
        kv("key_transform (D10)", getattr(rcfg, "key_transform", "none") or "none"),
        kv("base encoder (D9)", getattr(rcfg, "base_experiment", "?")),
        level_row("image"),
        level_row("slide"),
        kv("merge_levels (ablation)", getattr(rcfg, "merge_levels", False)),
        kv("block_query_patients", getattr(rcfg, "block_query_patients", True)),
        kv("gate", f"enabled={getattr(gate, 'enabled', '?')} "
                   f"hidden={getattr(gate, 'hidden', '?')} "
                   f"init={getattr(gate, 'init_weights', '?')} "
                   f"fit_on={getattr(gate, 'fit_on', '?')}"),
        *extra,
    ]
    section(logger, title, rows)


def _same_mag_row(diag: Dict[str, Any]) -> str:
    """Interpret the same-magnification rate against the route that produced it.

    Under ``same_mag`` a rate of 1.00 is a tautology and merely *confirms* the D3
    routing fired. It only carries D1 information under ``all``, where ~0.25 is
    chance and ~1.00 means the key itself is magnification-locked.
    """
    rate = diag.get("same_magnification_rate", float("nan"))
    route = diag.get("route")
    if route == "same_mag":
        note = "routing verified (1.0000 expected by construction, D3)"
    elif route == "cross_mag":
        note = "0.0000 expected by construction (D3)"
    elif route == "all":
        note = "chance ~0.25; ~1.00 would mean a magnification-locked key (D1)"
    else:
        note = "chance ~0.25"
    return f"{rate:.4f}   ({note})"


def report_retrieval_diagnostics(logger: logging.Logger, diag: Dict[str, Any],
                                 title: str = "RETRIEVAL DIAGNOSTICS") -> None:
    """Log the neighbourhood-quality block (independent of any classifier metric)."""
    rows = [
        kv("queries", diag.get("n_queries", "?")),
        kv("mean neighbours / query", f"{diag.get('mean_neighbours', float('nan')):.2f}"),
        kv("short neighbourhoods", f"{diag.get('short_neighbourhood_fraction', 0.0):.1%}"),
        kv("mean top-1 similarity", f"{diag.get('mean_top1_similarity', float('nan')):.4f}"),
        kv("mean neighbour agreement", f"{diag.get('mean_neighbour_agreement', float('nan')):.4f}"),
        kv("mean distinct patients", f"{diag.get('mean_distinct_patients', float('nan')):.2f}"),
        kv("same-magnification rate", _same_mag_row(diag)),
    ]
    levels = diag.get("levels") or {}
    if levels:
        rows.extend([
            _rule(),
            "  -- two independent rankings over the ONE bank (D5) --",
            kv("slide rows retrieved / query", f"{levels.get('mean_slide_neighbours', 0.0):.2f}"),
            kv("p_slide spread (std)", f"{levels.get('p_slide_std', float('nan')):.4f}"
                                       "   (0.0000 = level absent/merged)"),
            kv("image vs slide disagreement", f"{levels.get('img_slide_disagreement', float('nan')):.1%}"),
            kv("mean |p_img - p_slide|", f"{levels.get('mean_abs_p_img_minus_p_slide', float('nan')):.4f}"),
        ])
    weights = diag.get("mean_gate_weights") or {}
    if weights:
        rows.append(kv("mean gate weights", f"param={weights.get('param', float('nan')):.3f}  "
                                            f"img={weights.get('img', float('nan')):.3f}  "
                                            f"slide={weights.get('slide', float('nan')):.3f}"))
    hist = diag.get("retrieved_subtype_histogram")
    if hist:
        top = ", ".join(f"{s} {v['fraction']:.0%}" for s, v in list(hist.items())[:6])
        rows.append(kv("retrieved subtypes (D8)", top))
    section(logger, title, rows)


# --------------------------------------------------------------------------- #
# Report: final experiment summary
# --------------------------------------------------------------------------- #
def report_final_summary(logger: logging.Logger, *, cfg, fold, model, best_epoch,
                         best_metric, monitor, best_val_metrics, total_seconds,
                         avg_epoch_seconds, expdir) -> None:
    def hms(seconds: float) -> str:
        seconds = int(seconds)
        return f"{seconds // 3600:d}h {(seconds % 3600) // 60:02d}m {seconds % 60:02d}s"

    active = [n for n, on in (
        ("Swin", True),
        ("FPN", getattr(model, "uses_fpn", False)),
        ("Fusion", True),
        ("MagEmbed", getattr(model, "uses_magnification", False)),
        ("Projection", getattr(model, "uses_projection", False)),
    ) if on]

    key_metrics = {k: best_val_metrics.get(k) for k in ("accuracy", "auc", "f1",
                                                        "patient_accuracy", "patient_auc")
                   if k in best_val_metrics}
    rows = [
        kv("Experiment", cfg.experiment.name),
        kv("Fold", fold),
        kv("Architecture", " -> ".join(active)),
        kv("Random seed", cfg.seed),
        _rule(),
        kv("Best epoch", best_epoch + 1 if best_epoch is not None else "n/a"),
        kv(f"Best {monitor}", f"{best_metric:.4f}" if math.isfinite(best_metric) else "n/a"),
    ]
    rows.extend(kv(f"  best val_{k}", f"{v:.4f}") for k, v in key_metrics.items()
                if v is not None)
    rows.extend([
        _rule(),
        kv("Total training time", hms(total_seconds)),
        kv("Avg epoch time", f"{avg_epoch_seconds:.1f}s"),
        _rule(),
        kv("Checkpoints", expdir.checkpoints),
        kv("Config", expdir.config_path),
        kv("Output directory", expdir.root),
    ])
    section(logger, "EXPERIMENT SUMMARY", rows)


# --------------------------------------------------------------------------- #
# Extensibility: diagnostic registry for future modules
# --------------------------------------------------------------------------- #
class DiagnosticRegistry:
    """Collect per-epoch diagnostics contributed by (future) optional modules.

    A module registers a callback ``fn(context) -> {name: float}``; the trainer
    calls :meth:`collect` once per epoch and merges the returned scalars into the
    CSV / TensorBoard logs. This is the extension point that lets Retrieval Memory
    / Prototype Learning / Reasoning modules surface their own health signals
    **without editing the training loop**.
    """

    def __init__(self) -> None:
        self._hooks: Dict[str, Callable[[Dict[str, Any]], Optional[Dict[str, float]]]] = {}

    def register(self, name: str, fn: Callable[[Dict[str, Any]], Optional[Dict[str, float]]]) -> None:
        self._hooks[name] = fn

    def __len__(self) -> int:
        return len(self._hooks)

    def collect(self, context: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, fn in self._hooks.items():
            try:
                result = fn(context)
            except Exception as exc:  # a broken diagnostic must never kill training
                if logger is not None:
                    warn_banner(logger, f"Diagnostic '{name}' raised {type(exc).__name__}", str(exc))
                continue
            if result:
                out.update({f"{name}/{k}": v for k, v in result.items()})
        return out
