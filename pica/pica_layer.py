"""
PiCa Layer Definitions.

Key design principle (LoRA-style adapter storage):
  - The frozen base weight W is NOT stored inside this layer.
    Instead, it is stored in the original Linear layer (self.base_layer).
  - The frozen projector P (top-r right singular vectors of PyTorch's W) is stored as
    a plain buffer (not nn.Parameter), so it is excluded from requires_grad
    tracking but can be part of state dict if needed.
  - The ONLY trainable parameter is `shared_m` (shape: [out_dim, rank]),
    which is shared across all layers of the same module type.

Forward computation:  y = x @ (W + M @ P).T + bias
"""

import torch
import torch.nn as nn


class LinearWithPiCa(nn.Module):
    """
    Wraps a frozen nn.Linear with a PiCa low-rank adapter.

    Args:
        base_layer:  The original frozen nn.Linear module.
        rank:        Adapter rank.
        shared_m:    Shared trainable parameter M (shape: [out_dim, rank]).
                     If None, creates a layer-local M.
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 1, shared_m: nn.Parameter = None):
        super().__init__()
        self.base_layer = base_layer  # frozen – weights accessed read-only
        self.rank = rank

        out_dim, in_dim = base_layer.weight.shape

        # ── Compute P from SVD of frozen weight ──────────────────────────────
        # P is NOT a trainable parameter; register as buffer so it is saved
        # with the base model but excluded from optimizer.
        # Note: PyTorch weight is (out_dim, in_dim). SVD gives right singular vectors (Vh)
        # of shape (min(out, in), in_dim). These correspond to the left singular vectors of W_0 in paper.
        with torch.no_grad():
            _, _, v = torch.linalg.svd(base_layer.weight.detach().float(), full_matrices=False)
            p = v[:rank, :].to(base_layer.weight.dtype)
        self.register_buffer("p", p)          # shape: [rank, in_dim]

        # ── M: trainable adapter matrix ──────────────────────────────────────
        # Shared across all layers of the same module group.
        if shared_m is not None:
            self.shared_m = shared_m          # external Parameter, not owned
            self._own_m = False
        else:
            self.shared_m = nn.Parameter(
                torch.zeros(out_dim, rank, dtype=base_layer.weight.dtype)
            )
            self._own_m = True

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Avoid materializing the full [out_dim, in_dim] effective weight matrix.
        # Instead split into two matmuls whose intermediate is only [B, seq, rank].
        #   W_eff = W + M @ P
        #   y = x @ W_eff.T  ≡  x @ W.T  +  (x @ P.T) @ M.T
        # Ensure p is on same device/dtype as shared_m (multi-GPU guard)
        p = self.p.to(device=self.shared_m.device, dtype=self.shared_m.dtype)
        out = x @ self.base_layer.weight.T               # base output
        out = out + (x @ p.T) @ self.shared_m.T          # adapter: [B,S,rank]→[B,S,out]
        if self.base_layer.bias is not None:
            out = out + self.base_layer.bias
        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_merged_weight(self) -> torch.Tensor:
        """Return W + M @ P as a contiguous tensor (used by merge_and_unload)."""
        return (self.base_layer.weight + self.shared_m @ self.p).contiguous()

    @property
    def weight(self) -> torch.Tensor:
        """Expose merged weight for compatibility with code that calls .weight."""
        return self.get_merged_weight()

    @property
    def bias(self):
        return self.base_layer.bias

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base_layer.in_features}, "
            f"out_features={self.base_layer.out_features}, "
            f"rank={self.rank}, "
            f"shared_m={'yes' if not self._own_m else 'no'}"
        )