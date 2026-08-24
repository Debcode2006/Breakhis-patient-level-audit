"""What the Retrieval Memory keys on -- a small spec language over the forward dict.

``docs/retrieval.md`` D1 fixed the key as the pooled 1024-d ``features`` vector. The
diagnosis in ``docs/retrieval.md`` Part VII then showed *why the module cannot help
with that choice*: on exp3n the classifier is a single ``Linear`` over the very
same vector, so ``p_param`` and the kNN vote are two readouts of one near-rank-1
representation and an honest gate collapses onto the parametric branch.

Testing the alternatives requires the key to be a **configurable** thing rather
than a hardcoded dict lookup, which is what this module provides::

    retrieval.key: features          # D1 baseline (unchanged default)
    retrieval.key: projections       # the SupCon space
    retrieval.key: fpn.std           # spatial dispersion -- GAP throws this away
    retrieval.key: fpn.gap@0         # the finest pyramid level alone
    retrieval.key: fpn.gap.ln        # per-level L2 norm, so no level dominates
    retrieval.key: features+fpn.std  # concatenation of two L2-normalised parts

Grammar
-------
``<spec> := <part> ("+" <part>)*``  -- parts are L2-normalised then concatenated,
so a composite key gives each part equal weight in the cosine regardless of its
natural scale.

``<part>``:

===========================  =================================================
``features``                 forward dict entry (pre-magnification fused vector)
``embeddings``               forward dict entry (post-magnification)
``projections``              forward dict entry (L2-normalised SupCon head)
``fpn.<pool>``               concat of ``pool`` over every pyramid level
``fpn.<pool>@<i>``           that pool over pyramid level ``i`` only (0 = finest)
``fpn.<pool>.ln``            per-level L2 norm *before* concatenating
===========================  =================================================

``<pool>`` is one of ``gap`` (global average -- what ``FeatureFusion`` uses),
``max``, ``std`` (per-channel spatial standard deviation) or ``gem`` (generalised
mean, p=3). ``fpn.gap`` reproduces ``features`` exactly whenever the FPN is on and
``model.fusion.mode == "concat"``; :func:`assert_gap_matches_features` checks that
identity at runtime so the spec language cannot silently drift from the model.

Pure functions over the forward dict -- no state, no parameters, no model edits.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch

#: Spatial poolings available on an FPN level. ``gap`` is the one the model's own
#: ``FeatureFusion`` applies; the rest exist because GAP is exactly the operation
#: that discards the within-image structure the classifier therefore cannot read.
POOLINGS = ("gap", "max", "std", "gem")

#: Forward-dict entries usable verbatim as a key.
DIRECT_KEYS = ("features", "embeddings", "projections")

GEM_P = 3.0

_EPS = 1e-12


def l2(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + _EPS)


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #
def pool_map(m: torch.Tensor, how: str) -> torch.Tensor:
    """``[B, C, H, W] -> [B, C]`` under one of :data:`POOLINGS`."""
    if how == "gap":
        return m.mean(dim=(2, 3))
    if how == "max":
        return m.amax(dim=(2, 3))
    if how == "std":
        # Population std over the spatial grid: "how heterogeneous is this channel
        # across the field of view", which mean-pooling destroys by construction.
        return m.flatten(2).std(dim=2, unbiased=False)
    if how == "gem":
        x = m.clamp(min=_EPS).pow(GEM_P).mean(dim=(2, 3))
        return x.pow(1.0 / GEM_P)
    raise ValueError(f"Unknown pooling {how!r}; expected one of {POOLINGS}.")


# --------------------------------------------------------------------------- #
# Spec parsing
# --------------------------------------------------------------------------- #
def _parse_part(part: str) -> Dict:
    part = part.strip()
    if part in DIRECT_KEYS:
        return {"kind": "direct", "name": part}
    if not part.startswith("fpn."):
        raise ValueError(
            f"Unknown retrieval key part {part!r}. Expected one of {DIRECT_KEYS} "
            f"or 'fpn.<{'|'.join(POOLINGS)}>[.ln][@level]'."
        )
    body = part[len("fpn."):]
    level = None
    if "@" in body:
        body, lvl = body.split("@", 1)
        try:
            level = int(lvl)
        except ValueError as exc:
            raise ValueError(f"Bad pyramid level in {part!r}: {lvl!r}") from exc
    per_level_norm = body.endswith(".ln")
    if per_level_norm:
        body = body[: -len(".ln")]
    if body not in POOLINGS:
        raise ValueError(f"Unknown pooling {body!r} in {part!r}; expected {POOLINGS}.")
    return {"kind": "fpn", "pool": body, "level": level, "ln": per_level_norm}


def parse_key_spec(spec: str) -> List[Dict]:
    """Validate a spec string up-front (so a typo fails before a 10-minute encode)."""
    parts = [p for p in str(spec).split("+") if p.strip()]
    if not parts:
        raise ValueError(f"Empty retrieval key spec: {spec!r}")
    return [_parse_part(p) for p in parts]


def is_direct(spec: str) -> bool:
    """True when the spec is a plain forward-dict lookup (the pre-D1-ablation path)."""
    parsed = parse_key_spec(spec)
    return len(parsed) == 1 and parsed[0]["kind"] == "direct"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def assemble(
    spec: str,
    direct: Callable[[str], Optional[torch.Tensor]],
    level_vectors: Callable[[str], Sequence[torch.Tensor]],
) -> torch.Tensor:
    """Build the key for ``spec`` from two lookups. The single assembly path.

    ``direct(name)`` returns a forward-dict vector; ``level_vectors(pool)`` returns
    the per-pyramid-level ``[B, C]`` vectors for one pooling. Both the live-model
    path (:func:`extract_key`) and the pre-pooled cache path
    (:func:`extract_key_from_pooled`) go through here, so the ``.ln`` / ``@level``
    / ``+`` semantics cannot drift between an offline study and production.
    """
    parts = parse_key_spec(spec)
    out_parts: List[torch.Tensor] = []
    for part in parts:
        if part["kind"] == "direct":
            vec = direct(part["name"])
            if vec is None:
                raise KeyError(
                    f"No usable '{part['name']}' available for retrieval key {spec!r}. "
                    "('projections' needs model.projection.enabled=true.)"
                )
            out_parts.append(vec.float())
            continue

        levels = list(level_vectors(part["pool"]))
        if not levels:
            raise KeyError(
                f"Retrieval key {spec!r} needs the FPN pyramid but none is available "
                "(model.fpn.enabled=false)."
            )
        if part["level"] is not None:
            i = part["level"]
            if not -len(levels) <= i < len(levels):
                raise IndexError(
                    f"Pyramid level {i} out of range: {len(levels)} levels (0 = finest)."
                )
            levels = [levels[i]]
        if part["ln"]:
            levels = [l2(v) for v in levels]
        out_parts.append(torch.cat(levels, dim=1).float() if len(levels) > 1 else levels[0].float())

    if len(out_parts) == 1:
        return out_parts[0]
    # Composite specs L2-normalise every part before concatenating: without that
    # the cosine is dominated by whichever part happens to have the larger norm,
    # which is a scale accident rather than a modelling decision.
    return torch.cat([l2(v) for v in out_parts], dim=1)


def extract_key(out: Dict[str, torch.Tensor], spec: str) -> torch.Tensor:
    """``forward dict -> [B, D]`` retrieval key for ``spec`` (see the module docstring)."""
    pyramid = out.get("fpn_features") or []
    return assemble(
        spec,
        direct=out.get,
        level_vectors=lambda pool: [pool_map(m, pool) for m in pyramid],
    )


def extract_key_from_pooled(
    spec: str,
    direct: Dict[str, torch.Tensor],
    pooled: Dict[str, Sequence[torch.Tensor]],
) -> torch.Tensor:
    """Same key, built from **already pooled** vectors (the ablation study's cache).

    ``pooled[pool]`` is the list of per-level ``[N, C]`` vectors. Encoding a split
    once and re-keying it offline is what makes a 30-configuration sweep cost one
    forward pass per fold instead of thirty.
    """
    return assemble(spec, direct=direct.get, level_vectors=lambda p: pooled.get(p, ()))


# --------------------------------------------------------------------------- #
# Runtime invariant
# --------------------------------------------------------------------------- #
def assert_gap_matches_features(out: Dict[str, torch.Tensor], atol: float = 1e-4) -> None:
    """``fpn.gap`` must equal ``features`` (FPN on + ``fusion.mode='concat'``).

    Checked on the first batch of any pyramid-keyed run: it is the one identity
    that ties this spec language to the model's own ``FeatureFusion``. If a future
    change to fusion breaks it, every pyramid key silently stops being comparable
    to the D1 baseline -- so fail loudly instead.
    """
    if out.get("fpn_features") is None or out.get("features") is None:
        return
    gap = extract_key(out, "fpn.gap")
    ref = out["features"].float()
    if gap.shape != ref.shape:
        raise AssertionError(
            f"fpn.gap {tuple(gap.shape)} != features {tuple(ref.shape)}: the fusion mode is "
            "not 'concat', so pyramid keys are not comparable to the D1 baseline."
        )
    delta = (gap - ref).abs().max().item()
    if delta > atol:
        raise AssertionError(
            f"fpn.gap differs from 'features' by {delta:.2e} (> {atol:.0e}): FeatureFusion no "
            "longer equals concat(GAP(level)) and pyramid keys are not comparable to D1."
        )
