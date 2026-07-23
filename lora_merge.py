"""
lora_merge.py

Core logic for merging one or more LoRA safetensors files into a base
diffusion model checkpoint (e.g. Krea 2 Turbo), producing a single merged
safetensors file with the LoRA permanently baked into the weights.

Supports the two most common LoRA key-naming conventions found in the wild:
  - "Kohya" / ComfyUI style:   lora_unet_..._lora_down.weight / _lora_up.weight (+ .alpha)
  - "Diffusers / PEFT" style:  ....lora_A.weight / ....lora_B.weight

Because exact key names vary by trainer/tool, key matching is done in two
passes: an exact-match pass, then a normalized fuzzy-match pass. Any LoRA
layers that cannot be matched to a base weight are reported back to the
caller rather than silently dropped, so the user can see what happened
before trusting the merged file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class LoraLayer:
    """A single matched (down, up) LoRA pair for one base weight."""
    base_key: str
    down: torch.Tensor
    up: torch.Tensor
    alpha: float | None  # None => alpha defaults to rank (scale = 1.0)


@dataclass
class MergeReport:
    lora_file: str
    matched: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    shape_mismatches: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #

def load_state_dict(path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load a .safetensors file fully into memory."""
    state = {}
    with safe_open(path, framework="pt", device=device) as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    return state


def peek_keys(path: str, limit: int = 20) -> list[str]:
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    return keys[:limit]


# --------------------------------------------------------------------------- #
# Key normalization for fuzzy matching
# --------------------------------------------------------------------------- #

_STRIP_PREFIXES = [
    "lora_unet_", "lora_te_", "lora_transformer_",
    "diffusion_model.", "model.diffusion_model.",
    "base_model.model.", "transformer.",
]
_STRIP_SUFFIXES = [
    ".weight", ".lora_down.weight", ".lora_up.weight",
    ".lora_A.weight", ".lora_B.weight", ".alpha",
    ".lora_A.default.weight", ".lora_B.default.weight",
]


def normalize_key(key: str) -> str:
    """
    Reduce a key to a canonical form so LoRA-side and base-side names can be
    compared even when prefixes/suffixes and dot-vs-underscore conventions
    differ between the two files.
    """
    k = key
    for suf in _STRIP_SUFFIXES:
        if k.endswith(suf):
            k = k[: -len(suf)]
            break
    for pre in _STRIP_PREFIXES:
        if k.startswith(pre):
            k = k[len(pre):]
            break
    # Kohya-style names replace '.' with '_' in the module path -- undo that
    # so both conventions collapse to the same alnum-only representation.
    k = k.lower()
    k = re.sub(r"[._]+", "_", k)
    k = k.strip("_")
    return k


# --------------------------------------------------------------------------- #
# LoRA pair extraction (handles both naming conventions)
# --------------------------------------------------------------------------- #

def extract_lora_pairs(lora_state: dict[str, torch.Tensor]) -> dict[str, dict]:
    """
    Group a raw LoRA state dict into {base_name: {"down":..., "up":..., "alpha":...}}
    base_name is the *raw* (un-normalized) key with the down/up/alpha suffix removed,
    so it still carries whatever prefix convention the file used.
    """
    pairs: dict[str, dict] = {}

    for key, tensor in lora_state.items():
        base_name = None
        role = None

        if key.endswith(".lora_down.weight"):
            base_name, role = key[: -len(".lora_down.weight")], "down"
        elif key.endswith(".lora_up.weight"):
            base_name, role = key[: -len(".lora_up.weight")], "up"
        elif key.endswith(".lora_A.weight"):
            base_name, role = key[: -len(".lora_A.weight")], "down"
        elif key.endswith(".lora_B.weight"):
            base_name, role = key[: -len(".lora_B.weight")], "up"
        elif key.endswith(".lora_A.default.weight"):
            base_name, role = key[: -len(".lora_A.default.weight")], "down"
        elif key.endswith(".lora_B.default.weight"):
            base_name, role = key[: -len(".lora_B.default.weight")], "up"
        elif key.endswith(".alpha"):
            base_name, role = key[: -len(".alpha")], "alpha"
        else:
            continue  # not a LoRA weight we recognize (skip metadata etc.)

        pairs.setdefault(base_name, {})[role] = tensor

    return pairs


# --------------------------------------------------------------------------- #
# Krea 2 specific: diffusers-style -> native/AI-Toolkit-style key translation
# --------------------------------------------------------------------------- #
#
# Verified directly against a real Krea 2 Turbo fp8-scaled checkpoint and a
# diffusers-format LoRA (both key skeletons dumped and cross-checked -- block
# counts, sub-module names, and component names all line up 1:1). This is an
# explicit rule table, not a guess:
#
#   diffusers (LoRA)                          native (base checkpoint)
#   -----------------------------------------  -------------------------------
#   transformer.transformer_blocks.N.*         blocks.N.*
#   transformer.text_fusion.layerwise_blocks.N  txtfusion.layerwise_blocks.N
#   transformer.text_fusion.refiner_blocks.N    txtfusion.refiner_blocks.N
#   transformer.text_fusion.projector           txtfusion.projector
#   transformer.img_in                          first
#   transformer.final_layer.linear              last.linear
#   transformer.time_embed.linear_1/2           tmlp.0 / tmlp.1
#   transformer.txt_in.linear_1/2               txtmlp.0 / txtmlp.1
#   transformer.time_mod_proj                   tproj.1   (lower confidence --
#                                                 single tensor, low impact if
#                                                 wrong; reported either way)
#
#   attn.to_q / to_k / to_v / to_gate           attn.wq / wk / wv / gate
#   attn.to_out.0                               attn.wo
#   ff.up / ff.down                             mlp.up / mlp.down

_TOP_LEVEL_RENAMES = [
    (re.compile(r"^transformer\.transformer_blocks\.(\d+)\."), r"blocks.\1."),
    (re.compile(r"^transformer\.text_fusion\.layerwise_blocks\.(\d+)\."), r"txtfusion.layerwise_blocks.\1."),
    (re.compile(r"^transformer\.text_fusion\.refiner_blocks\.(\d+)\."), r"txtfusion.refiner_blocks.\1."),
    (re.compile(r"^transformer\.text_fusion\.projector$"), "txtfusion.projector"),
    (re.compile(r"^transformer\.img_in$"), "first"),
    (re.compile(r"^transformer\.final_layer\.linear$"), "last.linear"),
    (re.compile(r"^transformer\.time_embed\.linear_1$"), "tmlp.0"),
    (re.compile(r"^transformer\.time_embed\.linear_2$"), "tmlp.1"),
    (re.compile(r"^transformer\.txt_in\.linear_1$"), "txtmlp.0"),
    (re.compile(r"^transformer\.txt_in\.linear_2$"), "txtmlp.1"),
    (re.compile(r"^transformer\.time_mod_proj$"), "tproj.1"),
]

_COMPONENT_RENAMES = [
    (re.compile(r"\.attn\.to_out\.0$"), ".attn.wo"),
    (re.compile(r"\.attn\.to_q$"), ".attn.wq"),
    (re.compile(r"\.attn\.to_k$"), ".attn.wk"),
    (re.compile(r"\.attn\.to_v$"), ".attn.wv"),
    (re.compile(r"\.attn\.to_gate$"), ".attn.gate"),
    (re.compile(r"\.ff\.up$"), ".mlp.up"),
    (re.compile(r"\.ff\.down$"), ".mlp.down"),
]


def diffusers_to_native(base_name: str) -> str | None:
    """
    Translate a diffusers-style Krea 2 LoRA base name into the native/
    AI-Toolkit-style name used by this project's base checkpoints.
    Returns None if base_name doesn't match a known diffusers-style pattern
    (caller should fall back to fuzzy matching in that case).
    """
    if not base_name.startswith("transformer."):
        return None

    translated = base_name
    matched_top = False
    for pattern, repl in _TOP_LEVEL_RENAMES:
        new = pattern.sub(repl, translated)
        if new != translated:
            translated = new
            matched_top = True
            break

    if not matched_top:
        return None

    for pattern, repl in _COMPONENT_RENAMES:
        translated = pattern.sub(repl, translated)

    return translated




def build_base_lookup(base_state: dict[str, torch.Tensor]) -> dict[str, str]:
    """normalized_key -> original base_state key, for weight tensors only."""
    lookup = {}
    for key in base_state.keys():
        if not key.endswith(".weight"):
            continue
        lookup[normalize_key(key)] = key
    return lookup


def match_layers(
    lora_pairs: dict[str, dict],
    base_state: dict[str, torch.Tensor],
    base_lookup: dict[str, str],
) -> tuple[list[LoraLayer], list[str], list[str]]:
    matched_layers: list[LoraLayer] = []
    unmatched: list[str] = []
    shape_mismatches: list[str] = []

    for base_name, parts in lora_pairs.items():
        if "down" not in parts or "up" not in parts:
            unmatched.append(base_name + " (incomplete pair)")
            continue

        candidate = base_name + ".weight"
        base_key = candidate if candidate in base_state else None

        if base_key is None:
            translated = diffusers_to_native(base_name)
            if translated is not None:
                candidate2 = translated + ".weight"
                base_key = candidate2 if candidate2 in base_state else None

        if base_key is None:
            norm = normalize_key(base_name)
            base_key = base_lookup.get(norm)

        if base_key is None:
            unmatched.append(base_name)
            continue

        down, up = parts["down"], parts["up"]
        alpha_t = parts.get("alpha")
        alpha = float(alpha_t.item()) if alpha_t is not None else None

        base_shape = base_state[base_key].shape
        expected_out, expected_in = base_shape[0], base_shape[1] if len(base_shape) > 1 else base_shape[0]
        if up.shape[0] != expected_out or down.shape[1] != expected_in:
            shape_mismatches.append(
                f"{base_key}: base{tuple(base_shape)} vs lora up{tuple(up.shape)}/down{tuple(down.shape)}"
            )
            continue

        matched_layers.append(LoraLayer(base_key=base_key, down=down, up=up, alpha=alpha))

    return matched_layers, unmatched, shape_mismatches


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #

def merge_layer_into_base(
    base_state: dict[str, torch.Tensor],
    layer: LoraLayer,
    user_weight: float,
    compute_dtype: torch.dtype,
) -> None:
    rank = layer.down.shape[0]
    scale = (layer.alpha / rank) if layer.alpha is not None else 1.0
    scale *= user_weight

    down = layer.down.to(compute_dtype)
    up = layer.up.to(compute_dtype)
    base_w = base_state[layer.base_key]
    orig_dtype = base_w.dtype

    delta = (up @ down) * scale

    # fp8-scaled checkpoints (e.g. Krea 2's *_fp8_scaled.safetensors) store a
    # companion "<key>_scale" tensor alongside each quantized weight. Adding a
    # LoRA delta directly to the raw fp8 values would silently corrupt them --
    # dequantize to full precision, merge, then requantize.
    is_fp8 = base_w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    scale_key = layer.base_key[: -len(".weight")] + ".weight_scale" if layer.base_key.endswith(".weight") else None
    has_scale = is_fp8 and scale_key is not None and scale_key in base_state

    if has_scale:
        weight_scale = base_state[scale_key].to(compute_dtype)
        dequantized = base_w.to(compute_dtype) * weight_scale
        merged = dequantized + delta.reshape(dequantized.shape)

        fp8_max = 448.0  # max representable magnitude of float8_e4m3fn
        max_abs = merged.abs().max()
        new_scale = (max_abs / fp8_max).clamp(min=1e-12)
        requantized = (merged / new_scale).clamp(-fp8_max, fp8_max).to(orig_dtype)

        base_state[layer.base_key] = requantized
        base_state[scale_key] = new_scale.reshape(base_state[scale_key].shape).to(base_state[scale_key].dtype)
    else:
        delta = delta.reshape(base_w.shape)
        merged = base_w.to(compute_dtype) + delta
        base_state[layer.base_key] = merged.to(orig_dtype)


def merge_lora_file(
    base_state: dict[str, torch.Tensor],
    lora_path: str,
    weight: float,
    compute_dtype: torch.dtype = torch.float32,
    progress_cb: Callable[[str], None] | None = None,
) -> MergeReport:
    """Merge a single LoRA file into base_state IN PLACE. Returns a report."""
    lora_state = load_state_dict(lora_path)
    pairs = extract_lora_pairs(lora_state)
    base_lookup = build_base_lookup(base_state)

    matched, unmatched, shape_mismatches = match_layers(pairs, base_state, base_lookup)

    for i, layer in enumerate(matched):
        merge_layer_into_base(base_state, layer, weight, compute_dtype)
        if progress_cb and i % 25 == 0:
            progress_cb(f"  merged {i}/{len(matched)} layers from {Path(lora_path).name}")

    return MergeReport(
        lora_file=Path(lora_path).name,
        matched=[l.base_key for l in matched],
        unmatched=unmatched,
        shape_mismatches=shape_mismatches,
    )


def check_compatibility(base_path: str, lora_path: str) -> MergeReport:
    """Dry-run: report how many LoRA layers would match, without merging."""
    base_state = load_state_dict(base_path)
    lora_state = load_state_dict(lora_path)
    pairs = extract_lora_pairs(lora_state)
    base_lookup = build_base_lookup(base_state)
    matched, unmatched, shape_mismatches = match_layers(pairs, base_state, base_lookup)
    return MergeReport(
        lora_file=Path(lora_path).name,
        matched=[l.base_key for l in matched],
        unmatched=unmatched,
        shape_mismatches=shape_mismatches,
    )


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #

DTYPE_MAP = {
    "keep original": None,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def save_merged(base_state: dict[str, torch.Tensor], output_path: str, out_dtype: torch.dtype | None) -> None:
    if out_dtype is not None:
        base_state = {k: v.to(out_dtype) for k, v in base_state.items()}
    # safetensors requires contiguous tensors
    base_state = {k: v.contiguous() for k, v in base_state.items()}
    save_file(base_state, output_path)
