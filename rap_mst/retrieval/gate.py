"""FusionGate -- the only learned part of the retrieval module (~100 parameters).

``docs/retrieval.md`` D6. The evidence rejected the hard confidence gate proposed in
``docs/results/classifier_ladder.md`` §8: at tau=0.05 it touches 1.2% of images, moves image accuracy by
+0.002 and leaves patient accuracy unchanged. What works is blending nearly
everywhere at alpha ~ 0.5 -- and the optimal alpha is a **property of the trained
encoder** (0.5 for the SupCon encoders, ~0.8 for exp1), so it must be fitted, not
hardcoded.

Inputs (5, all in roughly [0, 1] so no feature scaler is needed)::

    p_param          the head's P(malignant)
    conf_param       |p_param - 0.5|            -- how decided the head is
    agreement        2*|p_img - 0.5|            -- neighbourhood consensus
    top1_sim         best image-level cosine    -- is anything actually similar?
    n_distinct_frac  distinct bank patients / k -- is the evidence broad or one slide?

Output: three softmax weights over (parametric, image-level, slide-level)::

    p_final = w_param * p_param + w_img * p_img + w_slide * p_slide

Fitting protocol (non-negotiable): **pooled out-of-fold**, never per fold. The
five folds' validation sets are a disjoint partition of the 66 CV patients, so
pooling gives full OOF coverage; ``docs/results/threshold_calibration.md`` §4 is the
cautionary tale -- operating parameters fitted per fold on ~13 patients swung
from 0.05 to 0.91 and *hurt* test.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

#: Order is fixed: it defines both the input layer and the saved-checkpoint contract.
GATE_INPUTS: Tuple[str, ...] = (
    "p_param", "conf_param", "agreement", "top1_sim", "n_distinct_frac",
)
GATE_TERMS: Tuple[str, ...] = ("param", "img", "slide")


class FusionGate(nn.Module):
    """5 -> hidden -> 3 MLP + softmax. Small enough to fit on OOF val without overfitting.

    Parameters
    ----------
    init_weights:
        Softmax output at initialisation, i.e. the blend the gate starts from
        (D6: ``(0.5, 0.5, 0.0)``).
    init_weight_floor:
        A weight of exactly 0 is ``log(0) = -inf`` in logit space, which would be
        an unrecoverable dead unit. The floor keeps the slide term initialised at
        a small-but-trainable value; with the default the effective start is
        ``(0.495, 0.495, 0.010)``.
    term_mask:
        ``(param, img, slide)`` availability. A masked term gets weight exactly 0,
        which is how ``levels.slide.enabled=false`` turns this into a clean 2-way
        blend instead of silently averaging in a neutral 0.5.
    """

    def __init__(
        self,
        hidden: int = 16,
        init_weights: Sequence[float] = (0.5, 0.5, 0.0),
        init_weight_floor: float = 0.01,
        term_mask: Sequence[int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        if len(init_weights) != 3:
            raise ValueError("init_weights must have 3 entries: (param, img, slide).")
        self.hidden = int(hidden)
        self.init_weights = tuple(float(w) for w in init_weights)
        self.init_weight_floor = float(init_weight_floor)

        self.net = nn.Sequential(
            nn.Linear(len(GATE_INPUTS), self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, len(GATE_TERMS)),
        )
        self.register_buffer(
            "mask", torch.tensor([float(m) for m in term_mask], dtype=torch.float32)
        )
        self._init_output_bias()

    def _init_output_bias(self) -> None:
        """Zero the last layer's weights and set its bias to log(init_weights).

        With a zero weight matrix the softmax output at init is *exactly*
        ``init_weights`` for every input, so an untrained gate reproduces the
        measured alpha=0.5 blend and training moves away from it deliberately.
        Gradients still flow (dL/dW = delta . hidden), so this is not a dead init.
        """
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        floored = [max(w, self.init_weight_floor) for w in self.init_weights]
        total = sum(floored)
        with torch.no_grad():
            last.bias.copy_(torch.tensor([math.log(w / total) for w in floored]))

    # ------------------------------------------------------------------ #
    @staticmethod
    def build_features(
        p_param: torch.Tensor, agreement: torch.Tensor, top1_sim: torch.Tensor,
        n_distinct_frac: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the ``[B, 5]`` gate input in the fixed :data:`GATE_INPUTS` order."""
        return torch.stack(
            [p_param, (p_param - 0.5).abs(), agreement, top1_sim, n_distinct_frac], dim=1
        ).to(torch.float32)

    def weights_from_features(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        logits = logits.masked_fill(self.mask.view(1, -1) == 0, float("-inf"))
        return torch.softmax(logits, dim=1)

    def forward(
        self,
        p_param: torch.Tensor,
        p_img: torch.Tensor,
        p_slide: torch.Tensor,
        agreement: torch.Tensor,
        top1_sim: torch.Tensor,
        n_distinct_frac: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> ``(p_final [B], weights [B, 3])``."""
        w = self.weights_from_features(
            self.build_features(p_param, agreement, top1_sim, n_distinct_frac)
        )
        p = torch.stack([p_param, p_img, p_slide], dim=1).to(torch.float32)
        return (w * p).sum(dim=1), w

    # ------------------------------------------------------------------ #
    def save(self, path, extra: Optional[Dict] = None) -> None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "state_dict": self.state_dict(),
            "hidden": self.hidden,
            "init_weights": list(self.init_weights),
            "init_weight_floor": self.init_weight_floor,
            "term_mask": self.mask.tolist(),
            "inputs": list(GATE_INPUTS),
            "terms": list(GATE_TERMS),
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> Tuple["FusionGate", Dict]:
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Fusion gate not found: {path}. Run scripts/train_gate.py first "
                "(or set retrieval.gate.enabled=false for the fixed-alpha ablation)."
            )
        state = torch.load(path, map_location=map_location, weights_only=False)
        if list(state.get("inputs", GATE_INPUTS)) != list(GATE_INPUTS):
            raise ValueError(
                f"Gate was fitted on inputs {state['inputs']} but this build expects "
                f"{list(GATE_INPUTS)} -- refit the gate."
            )
        gate = cls(
            hidden=int(state["hidden"]),
            init_weights=state["init_weights"],
            init_weight_floor=float(state.get("init_weight_floor", 0.01)),
            term_mask=state["term_mask"],
        )
        gate.load_state_dict(state["state_dict"])
        gate.eval()
        return gate, state


# --------------------------------------------------------------------------- #
# Fitting (Stage B2) -- kept here so the script only does I/O
# --------------------------------------------------------------------------- #
def fit_gate(
    gate: FusionGate,
    features: torch.Tensor,
    probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int = 400,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    log_every: int = 50,
    logger=None,
) -> Dict[str, float]:
    """Fit by binary cross-entropy on the pooled out-of-fold validation rows.

    Full-batch Adam: ~8,000 rows x 5 features against ~100 parameters is a convex-
    ish, tiny problem where minibatch noise buys nothing and reproducibility is
    worth more.

    Parameters
    ----------
    features: ``[N, 5]`` gate inputs (see :data:`GATE_INPUTS`).
    probs:    ``[N, 3]`` the three candidate probabilities (param, img, slide).
    labels:   ``[N]`` ground truth in {0, 1}.
    """
    gate.train()
    opt = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=weight_decay)
    y = labels.to(torch.float32)
    history = []
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        w = gate.weights_from_features(features)
        p = (w * probs).sum(dim=1).clamp(1e-6, 1 - 1e-6)
        loss = nn.functional.binary_cross_entropy(p, y)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        if logger is not None and log_every and (epoch % log_every == 0 or epoch == 1):
            mw = w.mean(dim=0).detach()
            logger.info(
                f"  epoch {epoch:4d}  bce={loss.item():.5f}  "
                f"mean w=({mw[0]:.3f}, {mw[1]:.3f}, {mw[2]:.3f})"
            )
    gate.eval()
    with torch.no_grad():
        w = gate.weights_from_features(features)
        p = (w * probs).sum(dim=1)
        acc = float(((p >= 0.5).to(torch.float32) == y).float().mean().item())
    return {
        "final_bce": history[-1],
        "initial_bce": history[0],
        "pooled_val_accuracy": acc,
        "mean_weights": [float(v) for v in w.mean(dim=0)],
    }
