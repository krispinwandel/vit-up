#!/usr/bin/env python3
"""
Extract ViTUp model weights from a Lightning checkpoint.

This script extracts the ViTUp model parameters from a Lightning checkpoint
and saves them as a clean state dict file.

The saved ViTUp tensors are stored without the outer ``vit_up.`` prefix so the
inference wrapper can load them directly into ``self.vit_up``.

Usage:
    python scripts/extract_weights.py checkpoint.ckpt -o weights.pt
    python scripts/extract_weights.py checkpoint.ckpt -o weights.safetensors --format safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import torch
except ImportError:
    print("Error: torch is required. Install with: pip install torch")
    sys.exit(1)

try:
    import safetensors.torch

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vit_up.utils.state_dict_migration import migrate_vit_up_state_key


WRAPPER_PREFIXES = (
    "model.",
    "module.",
    "vit_up_pl.",
    "vit_up_pl_v2.",
)


def load_lightning_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load a Lightning checkpoint and extract the state dict.

    Args:
        checkpoint_path: Path to Lightning checkpoint (.ckpt file)

    Returns:
        Model state dict
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading Lightning checkpoint from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint: {exc}") from exc

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print(f"✓ Extracted state_dict with {len(state_dict)} parameters")
        return state_dict

    if isinstance(checkpoint, dict) and all(
        isinstance(key, str) for key in checkpoint.keys()
    ):
        print(f"✓ Loaded raw state_dict with {len(checkpoint)} parameters")
        return checkpoint

    raise ValueError(
        "Checkpoint does not contain a 'state_dict' key and is not a raw state dict."
    )


def normalize_checkpoint_key_for_extraction(key: str) -> str:
    """Map a PL checkpoint key to the key used in extracted weights."""
    normalized_key = key
    for prefix in WRAPPER_PREFIXES:
        if normalized_key.startswith(prefix):
            normalized_key = normalized_key.removeprefix(prefix)
            break

    normalized_key = normalized_key.replace("vit_up._orig_mod.", "vit_up.", 1)
    normalized_key = normalized_key.replace("backbone._orig_mod.", "backbone.", 1)

    if normalized_key.startswith("vit_up."):
        normalized_key = normalized_key.removeprefix("vit_up.")
    return migrate_vit_up_state_key(normalized_key)


def should_extract_checkpoint_key(key: str, include_backbone_lora: bool = True) -> bool:
    """Return whether a checkpoint key belongs in the extracted weight file."""
    normalized_key = normalize_checkpoint_key_for_extraction(key)
    is_backbone = normalized_key.startswith("backbone.")
    is_backbone_lora = is_backbone and "lora" in normalized_key
    is_vit_up = not is_backbone and (
        key.startswith("vit_up.")
        or key.startswith("vit_up._orig_mod.")
        or any(
            key.startswith(f"{prefix}vit_up.")
            or key.startswith(f"{prefix}vit_up._orig_mod.")
            for prefix in WRAPPER_PREFIXES
        )
    )
    return is_vit_up or (include_backbone_lora and is_backbone_lora)


def filter_vit_up_weights(
    state_dict: Dict[str, Any],
    include_backbone_lora: bool = True,
) -> Dict[str, Any]:
    """
    Filter state dict to include only ViTUp model weights.

    By default includes backbone LoRA parameters (trained adapters).
    Excludes base backbone weights (loaded from HuggingFace).

    Args:
        state_dict: Full Lightning model state dict
        include_backbone_lora: If True, include backbone.*.lora_* parameters

    Returns:
        Filtered state dict containing only ViTUp weights and optional backbone LoRA
    """
    vit_up_state_dict: Dict[str, Any] = {}

    excluded_count = 0
    included_count = 0
    lora_count = 0

    for key, value in state_dict.items():
        new_key = normalize_checkpoint_key_for_extraction(key)
        is_backbone = new_key.startswith("backbone.")
        is_backbone_lora = is_backbone and "lora" in new_key

        if should_extract_checkpoint_key(
            key, include_backbone_lora=include_backbone_lora
        ):
            vit_up_state_dict[new_key] = value
            included_count += 1
            if is_backbone_lora:
                lora_count += 1
        else:
            excluded_count += 1

    print("✓ Filtered weights:")
    print(f"  - ViTUp parameters: {included_count - lora_count}")
    if include_backbone_lora and lora_count > 0:
        print(f"  - Backbone LoRA parameters: {lora_count}")
    print(f"  - Excluded (backbone base, etc): {excluded_count}")

    if included_count == 0:
        raise ValueError(
            "No ViTUp parameters found in checkpoint. "
            "Checkpoint may not be from a ViTUpPL model."
        )

    return vit_up_state_dict


def validate_weights(state_dict: Dict[str, Any]) -> bool:
    """
    Basic validation of extracted weights.

    Args:
        state_dict: Extracted ViTUp weights

    Returns:
        True if validation passes
    """
    if not isinstance(state_dict, dict):
        print("✗ Weights must be a dictionary")
        return False

    if len(state_dict) == 0:
        print("✗ No weights to save")
        return False

    expected_modules = [
        "query_embedding",
        "rel_pos_enc",
        "vit_up_blocks",
        "decoder_mlp",
    ]
    found_modules = set()
    for key in state_dict:
        for module in expected_modules:
            if module in key:
                found_modules.add(module)

    print(f"✓ Found ViTUp modules: {', '.join(sorted(found_modules))}")

    total_size = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total_size += value.numel() * value.element_size()

    size_mb = total_size / (1024**2)
    print(f"✓ Total weight size: {size_mb:.2f} MB")

    return True


def save_weights_pytorch(state_dict: Dict[str, Any], output_path: Path) -> None:
    """Save weights as a PyTorch .pt file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving weights to PyTorch format: {output_path}")
    torch.save(state_dict, output_path)
    print("✓ Weights saved")


def save_weights_safetensors(state_dict: Dict[str, Any], output_path: Path) -> None:
    """Save weights as a safetensors file."""
    if not HAS_SAFETENSORS:
        raise ImportError(
            "safetensors is required for .safetensors format. "
            "Install with: pip install safetensors"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving weights to safetensors format: {output_path}")
    safetensors.torch.save_file(state_dict, output_path)
    print("✓ Weights saved")


def get_output_path(input_path: Path, format: str) -> Path:
    """Determine output path based on input path and format."""
    stem = input_path.stem
    ext = ".safetensors" if format == "safetensors" else ".pt"
    return input_path.parent / f"{stem}_weights{ext}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract ViTUp model weights from a Lightning checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract as PyTorch format (default)
  python scripts/extract_weights.py checkpoint.ckpt -o weights.pt

  # Extract as safetensors format (recommended for HF Hub)
  python scripts/extract_weights.py checkpoint.ckpt -o weights.safetensors --format safetensors

  # Auto-detect output format from extension
  python scripts/extract_weights.py checkpoint.ckpt -o output.safetensors

  # Extract without backbone LoRA parameters (LoRA is included by default)
  python scripts/extract_weights.py checkpoint.ckpt -o weights_vitup_only.safetensors --exclude-lora

  # Validate during extraction
  python scripts/extract_weights.py checkpoint.ckpt -o weights.pt --validate

  # Using with environment variable
  python scripts/extract_weights.py $CKPT_PATH -o weights.safetensors --format safetensors
        """,
    )

    parser.add_argument(
        "input", type=str, help="Path to Lightning checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output weights file (default: auto-generated based on input)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["pytorch", "safetensors", "auto"],
        default="auto",
        help="Output format. 'auto' detects from file extension (default: auto)",
    )
    parser.add_argument(
        "--exclude-lora",
        action="store_true",
        help="Exclude backbone LoRA parameters from extracted weights (default: include)",
    )
    parser.add_argument(
        "-v", "--validate", action="store_true", help="Validate extracted weights"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress verbose output"
    )

    args = parser.parse_args()

    try:
        input_path = Path(args.input)
        output_path = (
            Path(args.output)
            if args.output is not None
            else get_output_path(input_path, "pytorch")
        )

        if args.format == "auto":
            format_name = (
                "safetensors" if output_path.suffix == ".safetensors" else "pytorch"
            )
        else:
            format_name = args.format

        if not args.quiet:
            print("ViTUp Weight Extractor")
            print(f"{'=' * 60}")
            print(f"Input:  {input_path}")
            print(f"Output: {output_path}")
            print(f"Format: {format_name}")
            print(f"{'=' * 60}\n")

        state_dict = load_lightning_checkpoint(input_path)

        if not args.quiet:
            print("\nFiltering to ViTUp weights...")
        vit_up_weights = filter_vit_up_weights(
            state_dict,
            include_backbone_lora=not args.exclude_lora,
        )

        if args.validate:
            if not args.quiet:
                print("\nValidating weights...")
            if not validate_weights(vit_up_weights):
                print("✗ Validation failed!")
                return 1

        if not args.quiet:
            print("\nSaving weights...")

        if format_name == "safetensors":
            save_weights_safetensors(vit_up_weights, output_path)
        else:
            save_weights_pytorch(vit_up_weights, output_path)

        if not args.quiet:
            size_bytes = sum(
                v.numel() * v.element_size()
                for v in vit_up_weights.values()
                if isinstance(v, torch.Tensor)
            )
            size_mb = size_bytes / (1024**2)
            print(
                f"\n✓ Successfully extracted {len(vit_up_weights)} parameters ({size_mb:.2f} MB)"
            )

        return 0

    except Exception as exc:
        print(f"\n✗ Error: {exc}", file=sys.stderr)
        if not args.quiet:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
