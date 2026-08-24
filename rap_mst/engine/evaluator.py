"""Evaluation loop shared by validation and testing.

Runs the model over a dataloader with no grad, collects image-level predictions
and the metadata needed for patient-level aggregation, and returns both a metrics
dict and the raw arrays (so `test.py` can dump per-image / per-patient CSVs).

Optional retrieval
------------------
Passing a ``RetrievalRunner`` (``rap_mst/retrieval/builder.py``) makes ``raw`` also
carry the memory module's evidence -- ``prob_param`` / ``prob_img`` / ``prob_slide``
/ ``prob_final``, the gate weights and the retrieved neighbours -- and switches
``prob``/``pred`` (and therefore the returned metrics) to the fused ``p_final``.
When it is ``None`` this function behaves exactly as before, so the trainer's
validation path is untouched.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from rap_mst.retrieval.keys import extract_key
from rap_mst.utils.metrics import compute_metrics, patient_level_metrics

#: Extra `raw` columns populated only when a retrieval runner is supplied.
RETRIEVAL_KEYS = (
    "prob_param", "prob_img", "prob_slide", "prob_final",
    "w_param", "w_img", "w_slide",
    "agreement", "top1_sim", "n_distinct_patients",
    "neighbour_patients", "neighbour_subtypes", "neighbour_sims", "neighbour_mags",
    "slide_patients", "slide_subtypes", "slide_sims",
)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    amp: bool = False,
    retrieval: Optional[object] = None,
) -> Tuple[Dict[str, float], Dict[str, list]]:
    """Evaluate ``model`` over ``loader``.

    Returns ``(metrics, raw)`` where ``raw`` holds parallel lists of
    ``patient_id``, ``label``, ``pred``, ``prob`` (malignant probability),
    ``magnification`` and ``image_path`` for downstream analysis.
    """
    model.eval()
    keys = ["patient_id", "label", "pred", "prob", "magnification", "image_path"]
    if retrieval is not None:
        keys += list(RETRIEVAL_KEYS)
    raw = {k: [] for k in keys}

    autocast = torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda")
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        mag_index = batch["mag_index"].to(device, non_blocking=True)
        with autocast:
            outputs = model(images, mag_index)
        probs = F.softmax(outputs["logits"].float(), dim=1)[:, 1]  # P(malignant)

        if retrieval is not None:
            # The key is a *spec* (rap_mst/retrieval/keys.py), not a bare dict
            # lookup, so pyramid-derived keys work here exactly as in the bank.
            key_vec = extract_key(outputs, retrieval.key)
            ev = retrieval(
                key_vec.detach().float(), mag_index, probs, patient_ids=batch["patient_id"]
            )
            probs = ev["p_final"].to(probs.device).float()
            _extend_retrieval(raw, ev)

        preds = (probs >= 0.5).long()
        raw["patient_id"].extend(batch["patient_id"])
        raw["label"].extend(batch["label"].tolist())
        raw["pred"].extend(preds.cpu().tolist())
        raw["prob"].extend(probs.cpu().tolist())
        raw["magnification"].extend(batch["magnification"])
        raw["image_path"].extend(batch["image_path"])

    metrics = compute_metrics(raw["label"], raw["pred"], raw["prob"])
    metrics.update(
        patient_level_metrics(
            np.asarray(raw["patient_id"]), raw["label"], raw["prob"]
        )
    )
    return metrics, raw


def _extend_retrieval(raw: Dict[str, list], ev: Dict[str, object]) -> None:
    """Append one batch of retrieval evidence to the raw columns."""
    weights = ev["weights"].cpu()
    raw["prob_param"].extend(ev["p_param"].cpu().tolist())
    raw["prob_img"].extend(ev["p_img"].cpu().tolist())
    raw["prob_slide"].extend(ev["p_slide"].cpu().tolist())
    raw["prob_final"].extend(ev["p_final"].cpu().tolist())
    raw["w_param"].extend(weights[:, 0].tolist())
    raw["w_img"].extend(weights[:, 1].tolist())
    raw["w_slide"].extend(weights[:, 2].tolist())
    raw["agreement"].extend(ev["agreement"].cpu().tolist())
    raw["top1_sim"].extend(ev["top1_sim"].cpu().tolist())
    raw["n_distinct_patients"].extend(ev["n_distinct_patients"].cpu().tolist())

    # Retrieved evidence, per level: who was retrieved, of which subtype, how close.
    # This is the interpretability deliverable -- a pathologist can inspect *which*
    # archived slides drove a decision (docs/retrieval.md §9).
    n = ev["p_final"].shape[0]
    for prefix, result in (("neighbour", ev.get("image_result")), ("slide", ev.get("slide_result"))):
        image_level = prefix == "neighbour"
        pid_key = "neighbour_patients" if image_level else "slide_patients"
        sub_key = "neighbour_subtypes" if image_level else "slide_subtypes"
        sim_key = "neighbour_sims" if image_level else "slide_sims"
        keys = (pid_key, sub_key, sim_key) + (("neighbour_mags",) if image_level else ())
        if result is None:
            for key in keys:
                raw[key].extend([""] * n)
            continue
        raw[pid_key].extend(";".join(p) for p in result.patient_ids)
        raw[sub_key].extend(";".join(s) for s in result.subtypes)
        sims = result.sim.cpu().numpy()
        valid = result.valid.cpu().numpy()
        raw[sim_key].extend(";".join(f"{s:.4f}" for s in sims[b][valid[b]]) for b in range(n))
        if image_level:
            # The D1/D3 canary: what fraction of neighbours share the query's zoom?
            raw["neighbour_mags"].extend(";".join(str(m) for m in mags)
                                         for mags in result.magnifications)
