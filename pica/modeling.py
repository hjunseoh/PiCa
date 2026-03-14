"""
PiCa Modeling — PEFT-like Abstraction
======================================

Usage example::

    from pica import get_pica_model, load_pica_model

    # ── Training ──
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
    model = get_pica_model(model, target_modules=["q_proj", "v_proj"], rank=256)
    # ... train ...
    model.save_adapter("./adapter_dir")   # saves only shared_m + config

    # ── Inference (merge into base model) ──
    base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
    model = load_pica_model(base_model, "./adapter_dir")
    model = model.merge_and_unload()      # fused plain model, ready for inference
"""

import os
import json
from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

from .pica_layer import LinearWithPiCa
from .config import PiCaConfig


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _iter_linear_paths(model: nn.Module, target_substrings: List[str]):
    """Yield (parent, child_name, full_path) for every nn.Linear whose
    full dotted path contains at least one of the target_substrings."""
    for full_path, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(kw in full_path for kw in target_substrings):
                parent_path = full_path[: full_path.rfind(".")] if "." in full_path else ""
                child_name = full_path.split(".")[-1]
                parent = model.get_submodule(parent_path) if parent_path else model
                yield parent, child_name, full_path


def _iter_pica_paths(model: nn.Module):
    """Yield (parent, child_name, full_path, LinearWithPiCa) for every PiCa layer."""
    for full_path, module in model.named_modules():
        if isinstance(module, LinearWithPiCa):
            parent_path = full_path[: full_path.rfind(".")] if "." in full_path else ""
            child_name = full_path.split(".")[-1]
            parent = model.get_submodule(parent_path) if parent_path else model
            yield parent, child_name, full_path, module


# ─── PiCaModel ────────────────────────────────────────────────────────────────

class PiCaModel(nn.Module):
    """
    Wraps a base model and injects PiCa adapters.

    Mirrors the PEFT ``PeftModel`` interface::

        model = get_pica_model(base_model, target_modules=[...], rank=256)
        model.save_adapter(output_dir)         # save adapter weights only

        model = load_pica_model(base_model, adapter_dir)
        model = model.merge_and_unload()       # optional: fuse into base model
    """

    def __init__(self, model: nn.Module, config: PiCaConfig):
        super().__init__()
        self.pica_config = config
        self.model = model

        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Build shared_m dict: one per keyword group (e.g. "q_proj")
        self._shared_ms: dict = {}

        for kw in self.pica_config.target_modules:
            # Find the first matching Linear to determine out_features
            for _, _, full_path in _iter_linear_paths(self.model, [kw]):
                mod: nn.Linear = self.model.get_submodule(full_path)
                out_dim = mod.out_features
                dtype = mod.weight.dtype
                self._shared_ms[kw] = nn.Parameter(
                    torch.zeros(out_dim, self.pica_config.rank, dtype=dtype), requires_grad=True
                )
                break  # only need the first matching layer to determine shape

        # Register shared_m tensors as top-level parameters so optimizer finds them
        for kw, m in self._shared_ms.items():
            safe_name = f"pica_shared_m_{kw.replace('.', '_')}"
            self.register_parameter(safe_name, m)

        # inject LinearWithPiCa into the base model (in-place)
        print("Applying PiCa adapters…")
        for parent, child_name, full_path in tqdm(
            list(_iter_linear_paths(self.model, self.pica_config.target_modules))
        ):
            base_linear: nn.Linear = getattr(parent, child_name)
            # find which keyword group this belongs to
            shared_m = None
            for kw, m in self._shared_ms.items():
                if kw in full_path:
                    shared_m = m
                    break
            adapted = LinearWithPiCa(base_linear, rank=self.pica_config.rank, shared_m=shared_m)
            setattr(parent, child_name, adapted)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    # ── Forward / Mapping ───────────────────────────────────────────────────

    @property
    def config(self):
        """Delegate any config access to the underlying base model."""
        return self.model.config

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # ── Adapter save / load ───────────────────────────────────────────────────

    def save_adapter(self, output_dir: str):
        """
        Save **only** the trainable adapter weights (shared_m) to *output_dir*.
        Produces two files:
          - ``pica_adapter.bin``  — state dict with shared_m matrices only
          - ``pica_config.json``  — adapter hyper-parameters
        """
        os.makedirs(output_dir, exist_ok=True)

        adapter_state = {
            kw: m.detach().cpu()
            for kw, m in self._shared_ms.items()
        }
        torch.save(adapter_state, os.path.join(output_dir, "pica_adapter.bin"))
        self.pica_config.save(output_dir)
        print(f"PiCa adapter saved to '{output_dir}'")

    # ── Merge & unload ────────────────────────────────────────────────────────

    def merge_and_unload(self) -> nn.Module:
        """
        Merge all adapter weights into the frozen base weights, then replace
        every ``LinearWithPiCa`` with a plain ``nn.Linear``.
        Returns the unwrapped base model (no more PiCa layers).
        """
        print("Merging PiCa adapters into base model…")
        for parent, child_name, _, adapted in tqdm(
            list(_iter_pica_paths(self.model))
        ):
            base: nn.Linear = adapted.base_layer
            merged_weight = adapted.get_merged_weight()

            fused = nn.Linear(
                base.in_features, base.out_features, bias=base.bias is not None,
                dtype=base.weight.dtype, device=base.weight.device,
            )
            fused.weight = nn.Parameter(merged_weight, requires_grad=False)
            if base.bias is not None:
                fused.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False)

            setattr(parent, child_name, fused)

        # Make all params contiguous before returning
        for p in self.model.parameters():
            p.data = p.data.contiguous()

        return self.model

    # ── Convenience ───────────────────────────────────────────────────────────

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable:,} || Total: {total:,} || {100*trainable/total:.4f}%")


# ─── Public API ───────────────────────────────────────────────────────────────

def get_pica_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int,
    base_model_name_or_path: str = None,
) -> PiCaModel:
    """
    Wrap *model* with PiCa adapters.

    Args:
        model:                    Pre-trained base model.
        target_modules:           List of substrings to match target Linear layers
                                  (e.g. ``["q_proj", "v_proj"]``).
        rank:                     Adapter rank.
        base_model_name_or_path:  Optional; stored in the config for later use.

    Returns:
        A :class:`PiCaModel` with only the shared_m parameters trainable.
    """
    config = PiCaConfig(
        target_modules=target_modules,
        rank=rank,
        base_model_name_or_path=base_model_name_or_path,
    )
    return PiCaModel(model, config)


def load_pica_model(model: nn.Module, adapter_dir: str) -> PiCaModel:
    """
    Reconstruct a :class:`PiCaModel` from a saved adapter directory.

    The frozen projector ``P`` is **recomputed on-the-fly** via SVD from the
    base model's weights — only ``shared_m`` weights are loaded from disk.

    Args:
        model:        The **same** base model (same checkpoint) used during training.
        adapter_dir:  Path containing ``pica_adapter.bin`` and ``pica_config.json``.

    Returns:
        A :class:`PiCaModel` with adapters loaded, ready for inference or
        continued training.
    """
    config = PiCaConfig.load(adapter_dir)
    pica_model = PiCaModel(model, config)

    adapter_state: dict = torch.load(
        os.path.join(adapter_dir, "pica_adapter.bin"),
        map_location="cpu",
        weights_only=True,
    )

    # Copy loaded weights into the registered shared_m parameters
    for kw, tensor in adapter_state.items():
        if kw not in pica_model._shared_ms:
            raise KeyError(f"Adapter key '{kw}' not found in reconstructed PiCaModel.")
        expected = pica_model._shared_ms[kw].shape
        if tensor.shape != expected:
            raise ValueError(
                f"Shape mismatch for adapter key '{kw}': "
                f"saved {tuple(tensor.shape)} vs expected {tuple(expected)}. "
                f"Ensure you are loading with the same base model used during training."
            )
        pica_model._shared_ms[kw].data.copy_(tensor)

    print(f"PiCa adapter loaded from '{adapter_dir}'")
    return pica_model
