"""Modular losses. Add new objectives here and wire them into CombinedLoss."""

from rap_mst.losses.builder import build_loss
from rap_mst.losses.combined import CombinedLoss
from rap_mst.losses.supcon import SupConLoss

__all__ = ["SupConLoss", "CombinedLoss", "build_loss"]
