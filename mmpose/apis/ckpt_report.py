
# -*- coding: utf-8 -*-
"""
ckpt_report.py — Compare a checkpoint's state_dict with a model's state_dict,
print detailed hit/miss stats (overall, per module, and for keypoint_head submodules).

Usage (import in your train/test code, right before loading the checkpoint):

    import torch
    from ckpt_report import summarize_ckpt_vs_model

    ckpt = torch.load(cfg.load_from, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)

    model_obj = runner.model.module if hasattr(runner.model, 'module') else runner.model
    summarize_ckpt_vs_model(model_obj, state,
                            ignore_keys={'backbone.fisheye2sphere.patches_2d'},
                            group_depth=2, print_topn=20)

Optionally, filter and load only matched keys afterwards:

    filtered = filter_state_by_model(state, model_obj, ignore_keys={'backbone.fisheye2sphere.patches_2d'})
    model_obj.load_state_dict(filtered, strict=False)
"""
from __future__ import annotations
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, Optional, Set

import torch
from torch import nn

def _strip_module_prefix(keys: Iterable[str], target_has_module: bool) -> Dict[str, str]:
    """Align 'module.' prefix between checkpoint and model keys by returning a mapping
    old_key -> new_key to rewrite the checkpoint keys if needed."""
    has_module = any(k.startswith('module.') for k in keys)
    mapping = {}
    if has_module and not target_has_module:
        for k in keys:
            mapping[k] = k[len('module.'):]
    elif (not has_module) and target_has_module:
        for k in keys:
            mapping[k] = 'module.' + k
    else:
        for k in keys:
            mapping[k] = k
    return mapping

def _group_key(key: str, depth: int = 2) -> str:
    parts = key.split('.')
    return '.'.join(parts[:depth])

def _fmt_ratio(h: int, t: int) -> str:
    if t == 0:
        return "n/a"
    return f"{h}/{t} = {h*100.0/t:.1f}%"

def filter_state_by_model(state: Dict[str, torch.Tensor],
                          model: nn.Module,
                          ignore_keys: Optional[Set[str]] = None) -> Dict[str, torch.Tensor]:
    """Return a filtered copy of `state` that only contains keys that exist in `model`
    with EXACTLY matching shapes. Optionally ignore specific keys (e.g., camera patches)."""
    ignore_keys = set(ignore_keys or set())
    msd = model.state_dict()
    target_has_module = any(k.startswith('module.') for k in msd.keys())
    key_map = _strip_module_prefix(state.keys(), target_has_module)
    filtered = {}
    for old_k, new_k in key_map.items():
        if old_k in ignore_keys or new_k in ignore_keys:
            continue
        if new_k in msd and state[old_k].shape == msd[new_k].shape:
            filtered[new_k] = state[old_k]
    return filtered

def summarize_ckpt_vs_model(model: nn.Module,
                            state: Dict[str, torch.Tensor],
                            ignore_keys: Optional[Set[str]] = None,
                            group_depth: int = 2,
                            print_topn: int = 30) -> Dict[str, object]:
    """Print a detailed diff of checkpoint vs model.

    Returns a dict with counts and per-group coverage you can log/save if desired.
    """
    ignore_keys = set(ignore_keys or set())

    msd = model.state_dict()
    target_has_module = any(k.startswith('module.') for k in msd.keys())
    key_map = _strip_module_prefix(state.keys(), target_has_module)

    # Rewrite checkpoint keys for alignment; also build an "ignored" set
    aligned_state = {}
    ignored = set()
    for old_k, new_k in key_map.items():
        if old_k in ignore_keys or new_k in ignore_keys:
            ignored.add(old_k)
            continue
        aligned_state[new_k] = state[old_k]

    state_keys = set(aligned_state.keys())
    model_keys = set(msd.keys())

    matched = []
    shape_mismatch = []
    for k in (state_keys & model_keys):
        if aligned_state[k].shape == msd[k].shape:
            matched.append(k)
        else:
            shape_mismatch.append(k)
    missing = sorted(model_keys - state_keys)  # in model, not in ckpt
    unexpected = sorted(state_keys - model_keys)  # in ckpt, not in model

    # Coverage by group prefix
    def cover_table(keys_total: Iterable[str], keys_hit: Iterable[str]) -> Dict[str, tuple]:
        cover = defaultdict(lambda: [0, 0])
        for k in keys_total:
            g = _group_key(k, depth=group_depth)
            cover[g][1] += 1
        for k in keys_hit:
            g = _group_key(k, depth=group_depth)
            cover[g][0] += 1
        return {g: (h, t, (h / t if t else 0.0)) for g, (h, t) in cover.items()}

    cover = cover_table(model_keys, matched)

    # Pretty print
    print("=" * 80)
    print("Checkpoint vs Model weight statistics")
    print("-" * 80)
    print(f"Total model params: {len(model_keys)}")
    print(f"Matched params    : {len(matched)}")
    print(f"Shape mismatches  : {len(shape_mismatch)}")
    print(f"Missing in ckpt   : {len(missing)}")
    print(f"Unexpected in ckpt: {len(unexpected)}")
    if ignored:
        print(f"Ignored by user   : {len(ignored)} (e.g., {next(iter(ignored))})")
    print("-" * 80)

    # Focused groups
    for focus in ["backbone", "keypoint_head", "keypoint_head.cross_tgfi",
                  "keypoint_head.deconv", "keypoint_head.final_conv",
                  "limb_head"]:
        focus_total = [k for k in model_keys if k.startswith(focus + ".")]
        focus_hit = [k for k in matched if k.startswith(focus + ".")]
        if focus_total:
            print(f"[{focus}] coverage: {_fmt_ratio(len(focus_hit), len(focus_total))}")

    print("-" * 80)
    def _print_list(title: str, keys: Iterable[str], limit: int = print_topn):
        keys = list(keys)
        print(f"{title} ({len(keys)}):")
        for k in keys[:limit]:
            print("  -", k)
        if len(keys) > limit:
            print(f"  ... ({len(keys) - limit} more)")
        print("-" * 80)

    _print_list("Shape mismatches", shape_mismatch, print_topn)
    _print_list("Missing in checkpoint", missing, print_topn)
    _print_list("Unexpected in checkpoint", unexpected, print_topn)

    # Per-group coverage table
    print("Per-group coverage (first {} levels):".format(group_depth))
    for g, (h, t, r) in sorted(cover.items()):
        print(f"  {g:<35} {_fmt_ratio(h, t)}")

    return dict(
        total_model_params=len(model_keys),
        matched=len(matched),
        shape_mismatch=len(shape_mismatch),
        missing=len(missing),
        unexpected=len(unexpected),
        per_group=cover
    )
