"""State-dict key migration helpers for ViTUp architecture renames."""

from __future__ import annotations

import re
from typing import Any, Dict


_Q_PROJ_RE = re.compile(r"^q_proj\.(\d+)\.")
_LOCAL_FEATURE_DECODER_RE = re.compile(r"^local_feature_decoder\.(\d+)\.")
_GLOBAL_LOCAL_RE = re.compile(r"^global_local_cross_attention\.(\d+)\.(.*)$")


def migrate_vit_up_state_key(key: str) -> str:
    """Map old ViTUp module keys to the current block-oriented names."""
    prefix = ""
    rest = key
    if rest.startswith("vit_up."):
        prefix = "vit_up."
        rest = rest.removeprefix("vit_up.")

    if rest.startswith("img_encoder."):
        rest = "query_embedding." + rest.removeprefix("img_encoder.")
    elif rest.startswith("q_proj."):
        rest = _Q_PROJ_RE.sub(r"vit_up_blocks.\1.transition_mlp.", rest, count=1)
    elif rest.startswith("local_feature_decoder."):
        rest = _LOCAL_FEATURE_DECODER_RE.sub(r"vit_up_blocks.\1.featx.", rest, count=1)
    elif rest.startswith("global_local_cross_attention."):
        match = _GLOBAL_LOCAL_RE.match(rest)
        if match is not None:
            block_idx, tail = match.groups()
            if tail.startswith("global_cross_attention."):
                tail = "cross_attention." + tail.removeprefix("global_cross_attention.")
            rest = f"vit_up_blocks.{block_idx}.{tail}"

    return prefix + rest


def migrate_vit_up_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return a state dict with old ViTUp keys rewritten to current names."""
    migrated: Dict[str, Any] = {}
    for key, value in state_dict.items():
        migrated[migrate_vit_up_state_key(key)] = value
    return migrated

