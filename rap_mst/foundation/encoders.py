"""Frozen pathology foundation encoders (Stage F1).

One registry, one contract: ``build_foundation_encoder(cfg)`` returns
``(nn.Module, meta)`` where the module maps ``[B, 3, H, W] -> [B, D]`` pooled
features with **no** classifier head, in eval mode, with every parameter frozen.
Adding a second foundation model (a Phikon / Virchow2 variant that clears its
access terms) is a new entry in ``ENCODER_BUILDERS`` plus a
block under ``foundation.encoders:`` -- no other file changes.

CTransPath
----------
CTransPath is a Swin-Tiny whose patch-embedding is replaced by a small CNN stem
(:class:`ConvStem`), pretrained with semantically-relevant contrastive learning on
~15 M TCGA/PAIP histology patches. The published weights (Wang et al., *Medical
Image Analysis* 2022) ship as a Google-Drive ``.pth`` that needs a patched timm
0.5.4; we load the **ungated, timm-native HuggingFace mirror** instead, which is
byte-equivalent and needs no vendored timm.

Two timm-version details this file exists to absorb:

* timm >= 0.9 keeps Swin activations in **NHWC**, so ``ConvStem.forward`` permutes
  its NCHW conv output; the original TransPath stem flattened to NLC for timm
  0.5.4 and would silently mis-shape here.
* the CTransPath weights are in the **pre-0.9 key layout**, where ``downsample``
  sits at the *end* of stage *i* rather than the start of stage *i+1*. timm's own
  ``checkpoint_filter_fn`` performs exactly that remap (it is the same path timm
  uses to load the original MSRA Swin weights), so the load is complete -- but it
  is silent, hence :func:`verify_encoder_load`, which fails loudly if a future timm
  ever stops doing it.

At ~27.5 M parameters CTransPath is param-matched to RAP-MST's 30.8 M, which is
the whole point: the comparison isolates *pathology pretraining* from *model
scale* (the paper).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# CTransPath's CNN patch-embedding stem
# --------------------------------------------------------------------------- #
class ConvStem(nn.Module):
    """CTransPath's convolutional replacement for Swin's linear patch embedding.

    Two stride-2 conv/BN/ReLU blocks (3 -> D/8 -> D/4) then a 1x1 conv to the
    stage-0 width, i.e. an overall stride of 4 -- the same 56x56 grid Swin's 4x4
    patch embedding produces, so the transformer stages are untouched.

    The signature accepts every keyword timm's ``SwinTransformer`` passes to its
    ``embed_layer`` (``strict_img_size``, ``output_fmt``, ...) and exposes
    ``grid_size``, which timm reads to size the stages.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten: bool = True,
        output_fmt: str | None = None,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ) -> None:
        super().__init__()
        from timm.layers import to_2tuple

        if patch_size != 4:
            raise ValueError(f"ConvStem is hard-wired to patch_size=4, got {patch_size}.")
        if embed_dim % 8:
            raise ValueError(f"ConvStem needs embed_dim divisible by 8, got {embed_dim}.")

        img_size, patch_size = to_2tuple(img_size), to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.output_fmt = output_fmt
        self.strict_img_size = strict_img_size

        layers, in_dim, out_dim = [], in_chans, embed_dim // 8
        for _ in range(2):
            layers += [
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
            ]
            in_dim, out_dim = out_dim, out_dim * 2
        layers.append(nn.Conv2d(in_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*layers)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                  # [B, C, H/4, W/4]  (NCHW)
        x = x.permute(0, 2, 3, 1)         # -> NHWC, which timm >= 0.9 Swin expects
        return self.norm(x)


EMBED_LAYERS: Dict[str, Callable] = {"convstem": ConvStem}


# --------------------------------------------------------------------------- #
# Load verification
# --------------------------------------------------------------------------- #
def verify_encoder_load(model: nn.Module, hub_id: str) -> Dict[str, Any]:
    """Re-download the raw checkpoint and prove every pretrained tensor landed.

    ``timm.create_model(..., pretrained=True)`` does not raise when its key-remap
    drops a tensor -- it would just leave that block randomly initialised, and a
    randomly-initialised Swin stage still produces plausible-looking features. So
    this compares the model's parameters against the raw checkpoint tensors *by
    value*, allowing for the documented ``layers.i.downsample -> layers.i+1.downsample``
    rename, and raises if anything pretrained failed to arrive.

    Returns a small provenance dict that Stage F1 writes into the cache.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    repo_id = hub_id.split(":", 1)[1] if ":" in hub_id else hub_id
    raw = load_file(hf_hub_download(repo_id, "model.safetensors"))

    # Non-persistent buffers timm rebuilds itself; never present in a state_dict.
    skip = ("relative_position_index", "attn_mask")
    remap: Dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if any(s in key for s in skip):
            continue
        if ".downsample." in key and key.startswith("layers."):
            head, rest = key.split(".", 2)[1], key.split(".", 2)[2]
            key = f"layers.{int(head) + 1}.{rest}"   # pre-0.9 -> current layout
        remap[key] = value

    model_sd = model.state_dict()
    missing = sorted(k for k in remap if k not in model_sd)
    mismatched = sorted(
        k for k, v in remap.items()
        if k in model_sd and not torch.equal(model_sd[k].detach().cpu(), v.cpu())
    )
    if missing or mismatched:
        raise RuntimeError(
            "Pretrained weights did not fully land in the model -- the baseline would "
            "be measuring a partly random encoder.\n"
            f"  checkpoint keys with no home in the model : {missing[:8]}\n"
            f"  keys present but not equal to the weights : {mismatched[:8]}\n"
            f"  (timm={_timm_version()}; the pre-0.9 downsample remap in "
            "timm.models.swin_transformer.checkpoint_filter_fn may have changed.)"
        )
    # The head is intentionally absent (num_classes=0); everything else must match.
    unmatched = sorted(k for k in model_sd if k not in remap)
    return {
        "checkpoint_tensors": len(remap),
        "verified_identical": len(remap),
        "model_tensors_not_in_checkpoint": unmatched,
    }


def _timm_version() -> str:
    try:
        import timm

        return timm.__version__
    except Exception:  # pragma: no cover
        return "n/a"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def _build_timm_hub_encoder(spec, name: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """Any timm-native HuggingFace-hub encoder, optionally with a custom stem."""
    import timm

    hub_id = str(spec.hub_id)
    kwargs: Dict[str, Any] = {
        "pretrained": True,
        "num_classes": 0,                                   # no classifier head
        "global_pool": str(getattr(spec, "pool", "avg")),   # -> pooled [B, D]
    }
    embed_layer = getattr(spec, "embed_layer", None)
    if embed_layer:
        key = str(embed_layer).lower()
        if key not in EMBED_LAYERS:
            raise SystemExit(f"Unknown embed_layer '{embed_layer}'. Known: {list(EMBED_LAYERS)}")
        kwargs["embed_layer"] = EMBED_LAYERS[key]

    model = timm.create_model(hub_id, **kwargs)
    provenance = verify_encoder_load(model, hub_id)

    meta = {
        "encoder": name,
        "hub_id": hub_id,
        "embed_layer": str(embed_layer or "default"),
        "pool": kwargs["global_pool"],
        "feature_dim": int(model.num_features),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "timm": _timm_version(),
        **provenance,
    }
    return model, meta


ENCODER_BUILDERS: Dict[str, Callable] = {"ctranspath": _build_timm_hub_encoder}


def build_foundation_encoder(cfg, name: str | None = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build the configured frozen encoder. Returns ``(model, meta)``.

    The returned module is in ``eval()`` mode with ``requires_grad_(False)`` on
    every parameter: this baseline is a *linear probe*, and a stray gradient step
    into the encoder would quietly turn it into fine-tuning.
    """
    from rap_mst.foundation.builder import encoder_spec, foundation_cfg

    fcfg = foundation_cfg(cfg)
    name = (name or str(fcfg.encoder)).strip().lower()
    if name not in ENCODER_BUILDERS:
        raise SystemExit(
            f"Unknown foundation encoder '{name}'. Known: {list(ENCODER_BUILDERS)}. "
            "Add a builder in rap_mst/foundation/encoders.py and a block under "
            "`foundation.encoders:` in config/config.yaml."
        )
    spec = encoder_spec(cfg, name)
    model, meta = ENCODER_BUILDERS[name](spec, name)

    declared = getattr(spec, "feature_dim", None)
    if declared is not None and int(declared) != meta["feature_dim"]:
        raise SystemExit(
            f"foundation.encoders.{name}.feature_dim={declared} but the encoder "
            f"outputs {meta['feature_dim']}-d. Fix the config rather than the assert."
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta
