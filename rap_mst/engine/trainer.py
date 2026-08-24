"""Training engine.

Owns the full training lifecycle for a single fold: epoch loop, mixed-precision,
gradient accumulation, validation, checkpointing (best + last), early stopping,
resume, and logging (console, CSV, TensorBoard). It is intentionally framework-
agnostic (no PyTorch Lightning) so future retrieval / prototype hooks can be
inserted explicitly.

Logging note
------------
The trainer owns the CSV / TensorBoard writers and the *cadence* of logging; the
human-readable "reports" (startup, dataset, architecture, first-forward, SupCon
diagnostics, checkpoints, final summary) and the numeric guards live in
:mod:`rap_mst.utils.reporting`. Future modules attach epoch-level diagnostics via
``trainer.diagnostics.register(...)`` without touching this loop.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rap_mst.engine.evaluator import evaluate
from rap_mst.utils import reporting
from rap_mst.utils.checkpoint import build_state, load_checkpoint, save_checkpoint
from rap_mst.utils.logging_utils import CSVMetricLogger, get_logger


class Trainer:
    def __init__(self, cfg, model, datamodule, loss_fn, optimizer, scheduler, expdir, device):
        self.cfg = cfg
        self.model = model.to(device)
        self.dm = datamodule
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.expdir = expdir
        self.device = device

        self.logger = get_logger("rap_mst.train", expdir.log_path)
        self.csv = CSVMetricLogger(expdir.metrics_path)
        self.tb = SummaryWriter(str(expdir.tensorboard))

        self.amp = bool(cfg.train.mixed_precision) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.accum_steps = max(1, int(getattr(cfg.train, "grad_accum_steps", 1)))
        self.grad_clip = float(getattr(cfg.train, "grad_clip_norm", 0.0))

        # Early-stopping / best-checkpoint bookkeeping. Higher monitored metric
        # is better (accuracy / f1 / auc).
        self.monitor = getattr(cfg.train, "monitor", "patient_accuracy")
        self.patience = int(getattr(cfg.train, "early_stopping_patience", 0))
        self.best_metric = float("-inf")
        self.best_epoch: Optional[int] = None
        self.best_val_metrics: dict = {}
        self.epochs_no_improve = 0
        self.start_epoch = 0
        self._prev_monitor: Optional[float] = None

        # Shape trace is printed exactly once, on the first forward pass.
        self._shape_reporter = reporting.ForwardShapeReporter(self.model)
        # Extension point: future modules register per-epoch diagnostics here.
        self.diagnostics = reporting.DiagnosticRegistry()

    # ------------------------------------------------------------------ #
    # Resume
    # ------------------------------------------------------------------ #
    def maybe_resume(self, resume_path: Optional[str]) -> None:
        if not resume_path:
            return
        ckpt = load_checkpoint(resume_path, map_location=self.device)  # raises if missing
        restored = {"model": False, "optimizer": False, "scheduler": False,
                    "scaler": False, "rng": False}

        self.model.load_state_dict(ckpt["model"])
        restored["model"] = True
        if ckpt.get("optimizer"):
            self.optimizer.load_state_dict(ckpt["optimizer"])
            restored["optimizer"] = True
        if ckpt.get("scheduler") and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
            restored["scheduler"] = True
        if ckpt.get("scaler") and self.amp:
            self.scaler.load_state_dict(ckpt["scaler"])
            restored["scaler"] = True
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_metric = ckpt.get("best_metric", float("-inf"))
        rng = ckpt.get("rng_state")
        if rng:
            torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
            if rng.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            restored["rng"] = True

        reporting.report_resume(
            self.logger, path=resume_path, restored=restored,
            epoch=self.start_epoch, best_metric=self.best_metric, monitor=self.monitor,
        )

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    def fit(self) -> None:
        train_loader = self.dm.train_dataloader()
        val_loader = self.dm.val_dataloader()
        epochs = self.cfg.train.epochs

        self.logger.info(f"Experiment dir: {self.expdir}")
        self.logger.info(
            f"Starting fit: epochs={epochs} | AMP={self.amp} | accum={self.accum_steps} | "
            f"grad_clip={self.grad_clip} | monitor={self.monitor} | "
            f"diagnostics_registered={len(self.diagnostics)}"
        )

        fit_start = time.time()
        epoch_times: list[float] = []
        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.time()
            reporting.reset_peak_memory(self.device)

            train_stats = self._train_one_epoch(train_loader, epoch, epochs)
            val_metrics, _ = evaluate(self.model, val_loader, self.device, amp=self.amp)

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_duration = time.time() - epoch_start
            epoch_times.append(epoch_duration)
            self._run_guards(epoch, train_stats, val_metrics)
            self._log_epoch(epoch, train_stats, val_metrics, epoch_duration)

            improved = self._update_best(epoch, val_metrics)
            self._save(epoch, val_metrics, is_best=improved)

            if self.patience and self.epochs_no_improve >= self.patience:
                self.logger.info(
                    f"Early stopping at epoch {epoch + 1}: no {self.monitor} improvement "
                    f"for {self.patience} epochs."
                )
                break

        self.tb.close()
        total_seconds = time.time() - fit_start
        avg_epoch = sum(epoch_times) / max(len(epoch_times), 1)
        reporting.report_final_summary(
            self.logger, cfg=self.cfg, fold=self.dm.fold, model=self.model,
            best_epoch=self.best_epoch, best_metric=self.best_metric, monitor=self.monitor,
            best_val_metrics=self.best_val_metrics, total_seconds=total_seconds,
            avg_epoch_seconds=avg_epoch, expdir=self.expdir,
        )

    def _train_one_epoch(self, loader, epoch: int, epochs: int) -> dict:
        self.model.train()
        running = {"total": 0.0, "ce": 0.0, "supcon": 0.0}
        supcon_acc: dict = {}
        grad_norm_sum, grad_norm_peak, grad_steps = 0.0, 0.0, 0
        n_batches, n_images = 0, 0
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for step, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            mag_index = batch["mag_index"].to(self.device, non_blocking=True)

            # SupCon two-view path: a second correlated crop (dataset two_view)
            # gives every anchor a guaranteed same-instance positive. Both views
            # share one forward (batch-concatenated) so BatchNorm / activations
            # stay consistent; CE uses view-1 logits, SupCon the stacked [B, 2, D].
            view2 = batch.get("image2")
            two_view = view2 is not None and self.model.uses_projection

            with torch.autocast(device_type="cuda", enabled=self.amp):
                if two_view:
                    images2 = view2.to(self.device, non_blocking=True)
                    b = images.size(0)
                    outputs = self.model(
                        torch.cat([images, images2], dim=0),
                        torch.cat([mag_index, mag_index], dim=0),
                    )
                    proj = outputs["projections"]  # [2B, D]
                    outputs["projections"] = torch.stack([proj[:b], proj[b:]], dim=1)  # [B, 2, D]
                    outputs["logits"] = outputs["logits"][:b]  # CE on view 1 only
                    outputs["features"] = outputs["features"][:b]
                    outputs["embeddings"] = outputs["embeddings"][:b]
                else:
                    outputs = self.model(images, mag_index)
                loss, parts = self.loss_fn(outputs, labels)
                loss = loss / self.accum_steps

            # One-time shape trace (uses the real first batch; zero extra compute).
            self._shape_reporter.maybe_report(self.logger, images)

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0:
                # Always unscale so the reported gradient norm is the true norm.
                self.scaler.unscale_(self.optimizer)
                grad_norm = reporting.grad_global_norm(self.model)
                # Only clip on finite grads -- clipping inf grads would zero them
                # and defeat AMP's automatic inf-skip in scaler.step().
                if self.grad_clip > 0 and math.isfinite(grad_norm):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if math.isfinite(grad_norm):
                    grad_norm_sum += grad_norm
                    grad_norm_peak = max(grad_norm_peak, grad_norm)
                    grad_steps += 1

            # SupCon diagnostics (no-op for exp1/exp2 where projections is None).
            diag = reporting.supcon_diagnostics(outputs, labels)
            for k, v in diag.items():
                supcon_acc[k] = supcon_acc.get(k, 0.0) + v

            for k in running:
                running[k] += parts.get(k, 0.0)
            n_batches += 1
            n_images += images.size(0)
            pbar.set_postfix({k: f"{parts.get(k, 0.0):.3f}" for k in ("total", "ce", "supcon")})

        stats = {f"train_{k}": v / max(n_batches, 1) for k, v in running.items()}
        stats["grad_norm"] = grad_norm_sum / max(grad_steps, 1)
        stats["grad_norm_peak"] = grad_norm_peak
        stats["n_images"] = n_images
        # Mean per-batch SupCon diagnostics for the epoch.
        for k, v in supcon_acc.items():
            stats[k] = v / max(n_batches, 1)
        return stats

    # ------------------------------------------------------------------ #
    # Guards (automatic, highly-visible warnings)
    # ------------------------------------------------------------------ #
    def _run_guards(self, epoch: int, train_stats: dict, val_metrics: dict) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        reporting.check_numeric(self.logger, epoch, train_stats,
                                grad_norm=train_stats.get("grad_norm_peak", 0.0), lr=lr)
        supcon_diag = {k: v for k, v in train_stats.items() if k.startswith("supcon_")}
        reporting.check_supcon_health(self.logger, epoch, supcon_diag,
                                      train_stats.get("train_supcon", 0.0))
        current = val_metrics.get(self.monitor)
        reporting.check_val_collapse(self.logger, epoch, self.monitor, current, self._prev_monitor)
        self._prev_monitor = current

    # ------------------------------------------------------------------ #
    # Logging / checkpointing helpers
    # ------------------------------------------------------------------ #
    def _log_epoch(self, epoch: int, train_stats: dict, val_metrics: dict,
                   epoch_duration: float) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        images_per_sec = train_stats.get("n_images", 0) / max(epoch_duration, 1e-9)
        gpu = reporting.gpu_memory_mb(self.device)
        cpu_ram = reporting.cpu_ram_mb()

        # --- Diagnostics from future modules (Retrieval Memory, etc.) -------- #
        module_diag = self.diagnostics.collect(
            {"epoch": epoch, "model": self.model, "device": self.device,
             "train_stats": train_stats, "val_metrics": val_metrics},
            logger=self.logger,
        )

        # --- CSV (the richest, machine-readable record) ---------------------- #
        row = {"epoch": epoch, "lr": lr, **train_stats}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row.update({"epoch_seconds": epoch_duration, "images_per_sec": images_per_sec})
        row.update(gpu)
        if cpu_ram is not None:
            row["cpu_ram_mb"] = cpu_ram
        row.update(module_diag)
        self.csv.log(row)

        # --- TensorBoard ----------------------------------------------------- #
        for k, v in train_stats.items():
            self.tb.add_scalar(k.replace("_", "/", 1), v, epoch)
        for k, v in val_metrics.items():
            self.tb.add_scalar(f"val/{k}", v, epoch)
        self.tb.add_scalar("lr", lr, epoch)
        self.tb.add_scalar("time/epoch_seconds", epoch_duration, epoch)
        self.tb.add_scalar("time/images_per_sec", images_per_sec, epoch)
        for k, v in gpu.items():
            self.tb.add_scalar(f"resource/{k}", v, epoch)
        for k, v in module_diag.items():
            self.tb.add_scalar(k, v, epoch)

        # --- Console / file summary line ------------------------------------- #
        gpu_str = (f" | gpu={gpu['gpu_alloc_mb']:.0f}/{gpu['gpu_reserved_mb']:.0f}MB "
                   f"(peak {gpu['gpu_peak_mb']:.0f})") if gpu else ""
        supcon_str = ""
        if "train_supcon" in train_stats and train_stats["train_supcon"]:
            supcon_str = (f" | supcon={train_stats['train_supcon']:.3f} "
                          f"proj_var={train_stats.get('supcon_proj_var', float('nan')):.2e}")
        self.logger.info(
            f"[epoch {epoch + 1}] "
            f"lr={lr:.2e} | loss={train_stats['train_total']:.4f} "
            f"(ce={train_stats.get('train_ce', 0.0):.4f}{supcon_str}) | "
            f"grad_norm={train_stats.get('grad_norm', float('nan')):.2f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_{self.monitor.replace('patient_', 'pat_')}="
            f"{val_metrics.get(self.monitor, float('nan')):.4f} | "
            f"val_auc={val_metrics.get('auc', float('nan')):.4f} | "
            f"{epoch_duration:.1f}s ({images_per_sec:.0f} img/s){gpu_str}"
        )

    def _update_best(self, epoch: int, val_metrics: dict) -> bool:
        current = val_metrics.get(self.monitor)
        if current is None:
            raise KeyError(f"Monitored metric '{self.monitor}' not in val metrics {list(val_metrics)}")
        if current > self.best_metric:
            self.best_metric = current
            self.best_epoch = epoch
            self.best_val_metrics = dict(val_metrics)
            self.epochs_no_improve = 0
            return True
        self.epochs_no_improve += 1
        return False

    def _save(self, epoch: int, val_metrics: dict, is_best: bool) -> None:
        state = build_state(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.amp else None,
            epoch=epoch,
            best_metric=self.best_metric,
            config=self.cfg.to_dict(),
            extra={"monitor": self.monitor},
        )
        current = val_metrics.get(self.monitor, float("nan"))
        save_checkpoint(state, self.expdir.last_ckpt)
        reporting.report_checkpoint(
            self.logger, path=self.expdir.last_ckpt, reason="rolling last",
            monitor=self.monitor, current=current, best=self.best_metric, epoch=epoch,
        )
        if is_best:
            save_checkpoint(state, self.expdir.best_ckpt)
            reporting.report_checkpoint(
                self.logger, path=self.expdir.best_ckpt, reason="new best",
                monitor=self.monitor, current=current, best=self.best_metric, epoch=epoch,
            )
