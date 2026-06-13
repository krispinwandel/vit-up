"""
ViTUp Inference Wrapper

Clean inference API for ViTUp that doesn't depend on Lightning.
Loads weights from HuggingFace Hub or local files and provides simple inference interface.
"""

from __future__ import annotations

from contextlib import nullcontext
import importlib
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from peft import LoraConfig
from PIL import Image
import torchvision.transforms.v2 as T

from vit_up.model.vit_up import ViTUp
from vit_up.layers.backbones.dinov3_vit import DINOv3ViT
from vit_up.utils.img_transforms import RESNET_IMAGE_MEAN, RESNET_IMAGE_STD
from vit_up.utils.state_dict_migration import migrate_vit_up_state_dict_keys

HF_REPO_ID = "Krispin/vit-up"

MODEL_SPECS: dict[str, dict[str, str]] = {
    "vit_up_dinov3_base": {
        "config_path": "configs/runs/dinov3_base.yaml",
        "backbone_model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "weights_filename": "vit_up_dinov3_base.safetensors",
    },
    "vit_up_dinov3_splus": {
        "config_path": "configs/runs/dinov3_splus.yaml",
        "backbone_model_name": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "weights_filename": "vit_up_dinov3_splus.safetensors",
    },
}


class ViTUpWrapper(nn.Module):
    """
    ViTUp wrapper for easy inference.

    Handles loading the backbone from HuggingFace, loading ViTUp weights,
    and provides a simple interface for feature extraction.
    """

    def __init__(
        self,
        model_name: str = "vit_up_dinov3_base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_bfloat16: bool = True,
        hidden_layer_img_size: int = 448,
        query_chunk_size: Optional[int] = None,
    ):
        """
        Initialize ViTUp wrapper.

        Args:
            model_name: Model variant to load. Supported values are
                "vit_up_dinov3_base" and "vit_up_dinov3_splus".
            device: Device to load model on (cpu, cuda, mps)
            use_bfloat16: Whether to use bf16 inference and model weights
            hidden_layer_img_size: Image size to use for hidden backbone states
            query_chunk_size: Number of query points to process per chunk
        """
        super().__init__()

        self.model_name = self._normalize_model_name(model_name)
        self._model_spec = MODEL_SPECS[self.model_name]
        self.backbone_model_name = self._model_spec["backbone_model_name"]
        self.device = device
        self.device_type = torch.device(device).type
        self.use_bfloat16 = use_bfloat16
        self.dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.hidden_layer_img_size = hidden_layer_img_size
        self.query_chunk_size = query_chunk_size
        self._images: Optional[torch.Tensor] = None
        self._cache_data: Optional[dict[str, Any]] = None
        self._pil_image_transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=RESNET_IMAGE_MEAN, std=RESNET_IMAGE_STD),
            ]
        )

        self.model_config_path = self._resolve_model_config_path(self.model_name)
        self._model_config = self._load_model_config(self.model_config_path)

        model_init = dict(self._model_config["model"]["init_args"])
        self._backbone_lora_config = self._resolve_backbone_lora_config(model_init)
        self._vit_up_config = model_init["vit_up"]

        # Load backbone
        print(f"Loading backbone: {self.backbone_model_name}")
        self.backbone = self._load_backbone(
            self.backbone_model_name,
            backbone_lora_config=self._backbone_lora_config,
        )
        self.backbone = self.backbone.to(device=device, dtype=self.dtype).eval()

        # Load ViTUp model architecture from config (weights will be loaded separately)
        self.vit_up = self._create_vit_up_model(self._vit_up_config)
        self.vit_up = self.vit_up.to(device=device, dtype=self.dtype).eval()
        self.vit_up.compile()

        self._load_model_weights(
            repo_id=HF_REPO_ID,
            filename=self._model_spec["weights_filename"],
        )

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        aliases = {
            "dinov3_base": "vit_up_dinov3_base",
            "base": "vit_up_dinov3_base",
            "vit_up_dinov3_base": "vit_up_dinov3_base",
            "dinov3_splus": "vit_up_dinov3_splus",
            "splus": "vit_up_dinov3_splus",
            "vit_up_dinov3_splus": "vit_up_dinov3_splus",
        }
        normalized = aliases.get(str(model_name).strip().lower())
        if normalized is None:
            raise ValueError(
                f"Unknown model_name={model_name!r}. "
                f"Available: {sorted(MODEL_SPECS)}"
            )
        return normalized

    def _load_backbone(
        self,
        model_name: str,
        backbone_lora_config: Optional[LoraConfig] = None,
    ) -> DINOv3ViT:
        """
        Load DINOv3 backbone from HuggingFace.

        Args:
            model_name: HuggingFace model ID

        Returns:
            Loaded backbone model
        """
        try:
            backbone = DINOv3ViT.init_from_hf(
                backbone_model_name=model_name,
                backbone_lora_config=backbone_lora_config,
                freeze_weights=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load backbone {model_name}: {e}")

        return backbone

    def _load_weights_local(self, weights_path: str | Path) -> None:
        """
        Load weights from local file.

        Args:
            weights_path: Path to weights file
        """
        weights_path = Path(weights_path)

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        print(f"Loading weights from: {weights_path}")
        state_dict = self._load_state_dict_from_file(weights_path)

        self._init_vit_up_and_load_state(state_dict)

    def _load_weights_from_hub(self, repo_id: str, filename: str) -> None:
        """
        Load weights from HuggingFace Hub.

        Args:
            repo_id: Repository ID (format: "username/repo")
            filename: Filename in the repository
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required. Install with: pip install huggingface_hub"
            )

        print(f"Loading weights from HF Hub: {repo_id}/{filename}")

        try:
            weights_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )
            state_dict = self._load_state_dict_from_file(weights_path)
            self._init_vit_up_and_load_state(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from HF Hub: {e}")

    def _load_model_weights(self, repo_id: str, filename: str) -> None:
        try:
            self._load_weights_from_hub(repo_id=repo_id, filename=filename)
            return
        except RuntimeError as hub_error:
            repo_root = Path(__file__).resolve().parents[2]
            fallback_paths = [
                repo_root / "ckpts" / filename,
                repo_root / filename,
            ]
            for fallback_path in fallback_paths:
                if fallback_path.exists():
                    print(
                        "Falling back to local cached weights after HF load failed: "
                        f"{fallback_path}"
                    )
                    self._load_weights_local(fallback_path)
                    return
            raise hub_error

    @staticmethod
    def _load_state_dict_from_file(weights_path: str | Path) -> Dict[str, Any]:
        """Load a PyTorch or safetensors state dict from disk."""
        weights_path = Path(weights_path)
        if weights_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "safetensors is required to load .safetensors weights. Install with: pip install safetensors"
                ) from exc
            return load_file(weights_path, device="cpu")
        return torch.load(weights_path, map_location="cpu")

    @staticmethod
    def _resolve_model_config_path(model_name: str) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / MODEL_SPECS[model_name]["config_path"]

    @staticmethod
    def _load_model_config(model_config_path: Path) -> Dict[str, Any]:
        if not model_config_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_config_path}")
        config = OmegaConf.load(model_config_path)
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]

    @staticmethod
    def _resolve_backbone_lora_config(
        model_init: Dict[str, Any],
    ) -> Optional[LoraConfig]:
        backbone_lora_config = model_init.get("backbone_lora_config")
        if backbone_lora_config is None:
            return None
        if isinstance(backbone_lora_config, LoraConfig):
            return backbone_lora_config
        if not isinstance(backbone_lora_config, dict):
            raise TypeError("backbone_lora_config must be a dict when provided.")
        return LoraConfig(**backbone_lora_config)

    @staticmethod
    def _load_class(class_path: str):
        if class_path.startswith("nf_dino."):
            class_path = class_path.replace("nf_dino.", "vit_up.", 1)
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def _instantiate_config_tree(cls, node: Any) -> Any:
        if isinstance(node, dict):
            class_path = node.get("class_path") or node.get("_class_path")
            init_args = node.get("init_args") or node.get("_init_args")
            if class_path is not None:
                resolved_init_args = {
                    key: cls._instantiate_config_tree(value)
                    for key, value in dict(init_args or {}).items()
                }
                module_cls = cls._load_class(str(class_path))
                return module_cls(**resolved_init_args)
            return {
                key: cls._instantiate_config_tree(value) for key, value in node.items()
            }
        if isinstance(node, list):
            return [cls._instantiate_config_tree(value) for value in node]
        return node

    def _init_vit_up_and_load_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Initialize ViTUp model from state dict shape information.

        Args:
            state_dict: Model state dict
        """
        backbone_state_dict, vit_up_state_dict = self._split_state_dict(state_dict)

        if backbone_state_dict:
            self.backbone.load_state_dict(backbone_state_dict, strict=False)
        self.backbone.compile()

        if vit_up_state_dict:
            migrated_vit_up_state_dict = migrate_vit_up_state_dict_keys(
                vit_up_state_dict
            )
            self.vit_up.load_state_dict(migrated_vit_up_state_dict, strict=False)

        self.vit_up = self.vit_up.eval()

        print(f"✓ Loaded {len(state_dict)} weight parameters")

    @staticmethod
    def _split_state_dict(
        state_dict: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        backbone_state_dict: Dict[str, Any] = {}
        vit_up_state_dict: Dict[str, Any] = {}

        for key, value in state_dict.items():
            if key.startswith("backbone."):
                backbone_state_dict[key.removeprefix("backbone.")] = value
            else:
                vit_up_state_dict[key] = value

        return backbone_state_dict, vit_up_state_dict

    def _create_vit_up_model(self, vit_up_config: Dict[str, Any]) -> ViTUp:
        """Create the ViTUp architecture from a run config tree."""
        vit_up_module = self._instantiate_config_tree(vit_up_config)
        if not isinstance(vit_up_module, ViTUp):
            raise TypeError(
                "Expected vit_up config to instantiate to a ViTUp instance, "
                f"got {type(vit_up_module)!r}."
            )
        return vit_up_module

    def _prepare_images(self, images: torch.Tensor | Image.Image) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = self._pil_image_transform(images).unsqueeze(0)
        elif not isinstance(images, torch.Tensor):
            raise TypeError(
                "images must be a torch.Tensor or PIL.Image.Image, "
                f"got {type(images)!r}."
            )

        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(
                "images must have shape (C, H, W) or (B, C, H, W). "
                f"Got {tuple(images.shape)}."
            )
        return images.to(device=self.device)

    def _autocast_context(self):
        return (
            torch.autocast(device_type=self.device_type, dtype=torch.bfloat16)
            if self.use_bfloat16 and self.device_type in {"cuda", "cpu"}
            else nullcontext()
        )

    @torch.no_grad()
    def set_images(self, images: torch.Tensor | Image.Image) -> None:
        """
        Set input images and precompute reusable ViT-Up cache data.

        Args:
            images: Full-size input image tensor in CHW/BCHW format, or a single
                PIL image. Tensors are expected to already use the model's image
                normalization. PIL images are converted and normalized without
                resizing.
        """
        if self.vit_up is None:
            raise RuntimeError("Model weights not loaded. Call load_weights first.")

        images = self._prepare_images(images)
        with self._autocast_context():
            self._cache_data = self.vit_up.compute_cache_data(
                pixel_values=images,
                backbone=self.backbone,
                hidden_layer_img_size=self.hidden_layer_img_size,
            )
        self._images = images

    def get_cache_data(self) -> Optional[dict[str, Any]]:
        """Return the cache data created by the most recent set_images call."""
        return self._cache_data

    def _clear_cache(self) -> None:
        self._images = None
        self._cache_data = None

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor | Image.Image | None = None,
        query_coords: Optional[torch.Tensor] = None,
        query_chunk_size: Optional[int] = None,
        return_all_layers: bool = False,
    ) -> torch.Tensor | list:
        """
        Extract ViTUp features for query coordinates.

        Args:
            images: Optional input images. When provided, updates the internal
                cache via set_images before running inference. If omitted, the
                previously cached images from set_images are used.
            query_coords: Normalized query coordinates (0-1), shape (B, N_queries, 2)
            query_chunk_size: Optional number of query points to process per chunk.
                Defaults to the wrapper chunk size, or all queries if unset.
            return_all_layers: If True, return features from all layers.
                             If False, return only final layer features.

        Returns:
            Features: If return_all_layers=False, returns (B, N_queries, D) tensor.
                     If return_all_layers=True, returns list of (B, N_queries, D) tensors.
        """
        if self.vit_up is None:
            raise RuntimeError("Model weights not loaded. Call load_weights first.")
        if query_coords is None:
            raise ValueError("query_coords must be provided.")
        if images is not None:
            self.set_images(images)
        if self._cache_data is None:
            raise RuntimeError("No image cache available. Call set_images(images) first.")
        if query_coords.ndim != 3 or query_coords.shape[-1] != 2:
            raise ValueError(
                "query_coords must have shape (B, N_queries, 2). "
                f"Got {tuple(query_coords.shape)}."
            )
        if self._images is not None and query_coords.shape[0] != self._images.shape[0]:
            raise ValueError(
                "query_coords batch size must match cached images batch size. "
                f"Got {query_coords.shape[0]} query batches for "
                f"{self._images.shape[0]} cached images."
            )

        query_chunk_size = (
            self.query_chunk_size if query_chunk_size is None else query_chunk_size
        )
        if query_chunk_size is None:
            query_chunk_size = query_coords.shape[1]

        query_coords = query_coords.to(device=self.device, dtype=self.dtype)

        with self._autocast_context():
            q_chunks = []
            for q_start in range(0, query_coords.shape[1], query_chunk_size):
                q_end = min(q_start + query_chunk_size, query_coords.shape[1])
                q_layers_chunk = self.vit_up(
                    pixel_values=None,
                    q_xy_normalized=query_coords[:, q_start:q_end, :],
                    cache_data=self._cache_data,
                )
                if not return_all_layers:
                    q_layers_chunk = q_layers_chunk[-1]
                q_chunks.append(q_layers_chunk)

        # Return based on flag
        if return_all_layers:
            return [
                torch.cat([chunk[layer_idx] for chunk in q_chunks], dim=1)
                for layer_idx in range(len(q_chunks[0]))
            ]
        else:
            # Return features from last layer
            return torch.cat(q_chunks, dim=1)

    def set_config(
        self,
        use_bfloat16: Optional[bool] = None,
        query_chunk_size: Optional[int] = None,
    ) -> None:
        """
        Update wrapper configuration.

        Args:
            use_bfloat16: Whether to use bf16 inference
            query_chunk_size: New query chunk size
        """
        if use_bfloat16 is not None:
            self.use_bfloat16 = use_bfloat16
            self.dtype = torch.bfloat16 if use_bfloat16 else torch.float32
            self.backbone = self.backbone.to(device=self.device, dtype=self.dtype)
            self.vit_up = self.vit_up.to(device=self.device, dtype=self.dtype)
            self._clear_cache()

        if query_chunk_size is not None:
            self.query_chunk_size = query_chunk_size

    @property
    def model_config(self) -> Dict[str, Any]:
        """Get current model configuration."""
        return {
            "model_name": self.model_name,
            "config_path": str(self.model_config_path),
            "backbone": self.backbone_model_name,
            "weights_repo": HF_REPO_ID,
            "weights_filename": self._model_spec["weights_filename"],
            "device": self.device,
            "use_bfloat16": self.use_bfloat16,
            "hidden_layer_img_size": self.hidden_layer_img_size,
            "query_chunk_size": self.query_chunk_size,
            "vit_up_loaded": self.vit_up is not None,
        }


def create_vit_up_wrapper(
    model_name: str = "vit_up_dinov3_base",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_bfloat16: bool = True,
    hidden_layer_img_size: int = 448,
    query_chunk_size: Optional[int] = None,
) -> ViTUpWrapper:
    """
    Convenience function to create ViTUp wrapper with standard configurations.

    Args:
        model_name: Which model variant ("vit_up_dinov3_base" or
            "vit_up_dinov3_splus"). Short aliases "base" and "splus" are accepted.
        device: Device to use

    Returns:
        Initialized ViTUpWrapper
    """
    return ViTUpWrapper(
        model_name=model_name,
        device=device,
        use_bfloat16=use_bfloat16,
        hidden_layer_img_size=hidden_layer_img_size,
        query_chunk_size=query_chunk_size,
    )


if __name__ == "__main__":
    # Example usage
    print("ViTUp Inference Wrapper")
    print("=" * 60)
    print()
    print("Example:")
    print("  from vit_up.inference.vit_up_wrapper import ViTUpWrapper")
    print()
    print("  # Load model from HuggingFace Hub")
    print('  wrapper = ViTUpWrapper("vit_up_dinov3_base")')
    print('  wrapper = ViTUpWrapper("vit_up_dinov3_splus")')
    print()
    print("  # Run inference")
    print("  images = torch.randn(1, 3, 448, 448)")
    print("  query_coords = torch.rand(1, 100, 2)")
    print("  features = wrapper(images, query_coords)")
