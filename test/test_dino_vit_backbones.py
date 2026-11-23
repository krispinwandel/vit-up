from pathlib import Path
from typing import Any, Callable, Type

import pytest
import torch
from PIL import Image
from torchvision.transforms import v2 as T

from vit_up.layers.backbones.dino_vit_base import DinoViTBackboneBase
from vit_up.layers.backbones.dinov2_vit import DINOv2ViT
from vit_up.layers.backbones.dinov3_vit import DINOv3ViT

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
IMAGE_PATH = ASSET_DIR / "parrot.png"


def _load_pixel_values(image_size: int = 224) -> torch.Tensor:
    image = Image.open(IMAGE_PATH).convert("RGB")
    transform = T.Compose(
        [
            T.Resize((image_size, image_size), antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform(image).unsqueeze(0)


def _load_hf_model(model_name: str) -> Any:
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoModel.from_pretrained(model_name).eval()
    except Exception as exc:
        pytest.skip(f"Could not load Hugging Face checkpoint {model_name!r}: {exc}")


def _load_backbone(
    backbone_cls: Type[DinoViTBackboneBase],
    model_name: str,
) -> DinoViTBackboneBase:
    try:
        return backbone_cls.init_from_hf(model_name, freeze_weights=True).eval()
    except Exception as exc:
        pytest.skip(f"Could not load nf_dino backbone {model_name!r}: {exc}")


def _load_timm_model(model_name: str) -> Any:
    timm = pytest.importorskip("timm")
    try:
        return timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=False,
        ).eval()
    except Exception as exc:
        pytest.skip(f"Could not load timm checkpoint {model_name!r}: {exc}")


def _compute_timm_last_hidden_state_bhwc_with_pos_embed_antialias(
    model_name: str,
    pixel_values: torch.Tensor,
    *,
    antialias: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> torch.Tensor:
    timm = pytest.importorskip("timm")
    vit_module = pytest.importorskip("timm.models.vision_transformer")
    original_resample: Callable[..., torch.Tensor] = vit_module.resample_abs_pos_embed

    def resample_abs_pos_embed(*args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs["antialias"] = antialias
        return original_resample(*args, **kwargs)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            vit_module,
            "resample_abs_pos_embed",
            resample_abs_pos_embed,
        )
        try:
            timm_model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
                dynamic_img_size=True,
                dynamic_img_pad=False,
            ).eval()
        except Exception as exc:
            pytest.skip(f"Could not load timm checkpoint {model_name!r}: {exc}")
        with torch.inference_mode():
            return _timm_last_hidden_state_bhwc(timm_model, pixel_values)


def _spatial_tokens_bhwc(
    token_states: torch.Tensor,
    *,
    n_prefix_tokens: int,
    height: int,
    width: int,
    patch_size: int,
) -> torch.Tensor:
    h_tokens = height // patch_size
    w_tokens = width // patch_size
    return token_states[:, n_prefix_tokens:, :].reshape(
        token_states.shape[0],
        h_tokens,
        w_tokens,
        token_states.shape[-1],
    )


def _timm_last_hidden_state_bhwc(
    timm_model: Any,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    token_states = timm_model.forward_features(pixel_values)
    n_prefix_tokens = int(getattr(timm_model, "num_prefix_tokens", 1))
    spatial_tokens = token_states[:, n_prefix_tokens:, :]
    feat_hw = timm_model.patch_embed.dynamic_feat_size(tuple(pixel_values.shape[-2:]))
    return spatial_tokens.reshape(
        pixel_values.shape[0],
        feat_hw[0],
        feat_hw[1],
        token_states.shape[-1],
    )


def _distance_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff_abs = (actual - expected).abs()
    expected_abs = expected.abs().clamp_min(torch.finfo(expected.dtype).eps)
    diff_rel = diff_abs / expected_abs
    rel_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    ).clamp_min(torch.finfo(expected.dtype).eps)
    return {
        "abs_max": diff_abs.max().item(),
        "abs_p99": torch.quantile(diff_abs.flatten(), 0.99).item(),
        "abs_mean": diff_abs.mean().item(),
        "rel_max": diff_rel.max().item(),
        "rel_p99": torch.quantile(diff_rel.flatten(), 0.99).item(),
        "rel_mean": diff_rel.mean().item(),
        "rel_l2": rel_l2.item(),
    }


def _format_distance_stats(stats: dict[str, float]) -> str:
    return (
        f"abs_max={stats['abs_max']:.6g}, "
        f"abs_p99={stats['abs_p99']:.6g}, "
        f"abs_mean={stats['abs_mean']:.6g}, "
        f"rel_max={stats['rel_max']:.6g}, "
        f"rel_p99={stats['rel_p99']:.6g}, "
        f"rel_mean={stats['rel_mean']:.6g}, "
        f"rel_l2={stats['rel_l2']:.6g}"
    )


@pytest.mark.parametrize(
    ("backbone_cls", "hf_model_name", "timm_model_name", "image_size"),
    [
        (
            DINOv2ViT,
            "facebook/dinov2-small",
            "vit_small_patch14_dinov2.lvd142m",
            518,
        ),
        (
            DINOv2ViT,
            "facebook/dinov2-with-registers-small",
            "vit_small_patch14_dinov2.lvd142m",
            518,
        ),
        (
            DINOv3ViT,
            "facebook/dinov3-vits16plus-pretrain-lvd1689m",
            "vit_small_plus_patch16_dinov3.lvd1689m",
            256,
        ),
    ],
)
def test_dino_vit_backbone_last_hidden_state_matches_hf_and_timm(
    backbone_cls: Type[DinoViTBackboneBase],
    hf_model_name: str,
    timm_model_name: str,
    image_size: int,
) -> None:
    torch.manual_seed(0)
    pixel_values = _load_pixel_values(image_size=image_size)

    hf_model = _load_hf_model(hf_model_name)
    backbone = _load_backbone(backbone_cls, hf_model_name)
    timm_model = _load_timm_model(timm_model_name)

    with torch.inference_mode():
        hidden_states_bhwc = backbone(pixel_values=pixel_values)
        hf_outputs = hf_model(pixel_values=pixel_values)
        hf_last_hidden_state_bhwc = _spatial_tokens_bhwc(
            hf_outputs.last_hidden_state,
            n_prefix_tokens=backbone.n_prefix_tokens,
            height=pixel_values.shape[-2],
            width=pixel_values.shape[-1],
            patch_size=backbone.get_patch_size(),
        )
        timm_last_hidden_state_bhwc = _timm_last_hidden_state_bhwc(
            timm_model,
            pixel_values,
        )

    # print stats
    hf_stats = _distance_stats(hidden_states_bhwc[-1], hf_last_hidden_state_bhwc)
    timm_stats = _distance_stats(hidden_states_bhwc[-1], timm_last_hidden_state_bhwc)
    print(
        f"{backbone_cls.__name__} vs Hugging Face: "
        f"{_format_distance_stats(hf_stats)}"
    )
    print(f"{backbone_cls.__name__} vs timm: " f"{_format_distance_stats(timm_stats)}")

    # DinoViTBackboneBase consumers expect hidden_states_bhwc[-1] to be the
    # post-final-norm spatial token map, matching last_hidden_state.
    torch.testing.assert_close(
        hidden_states_bhwc[-1],
        hf_last_hidden_state_bhwc,
        rtol=1e-5,
        atol=1e-5,
    )
    if "with-registers" not in hf_model_name:
        torch.testing.assert_close(
            hidden_states_bhwc[-1],
            timm_last_hidden_state_bhwc,
            rtol=1e-4,
            atol=1e-4,
        )
    else:
        print("comparison with timm for registers disabled")


@pytest.mark.parametrize(
    ("backbone_cls", "hf_model_name", "timm_model_name"),
    [
        (
            DINOv2ViT,
            "facebook/dinov2-small",
            "vit_small_patch14_dinov2.lvd142m",
        ),
        (
            DINOv2ViT,
            "facebook/dinov2-with-registers-small",
            "vit_small_patch14_dinov2.lvd142m",
        ),
        (
            DINOv3ViT,
            "facebook/dinov3-vits16plus-pretrain-lvd1689m",
            "vit_small_plus_patch16_dinov3.lvd1689m",
        ),
    ],
)
def test_dino_vit_backbone_448_interpolation_distance_report(
    backbone_cls: Type[DinoViTBackboneBase],
    hf_model_name: str,
    timm_model_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    pixel_values = _load_pixel_values(image_size=448)

    hf_model = _load_hf_model(hf_model_name)
    backbone = _load_backbone(backbone_cls, hf_model_name)
    timm_model = _load_timm_model(timm_model_name)

    with torch.inference_mode():
        hidden_states_bhwc = backbone(pixel_values=pixel_values)
        hf_outputs = hf_model(pixel_values=pixel_values)
        hf_last_hidden_state_bhwc = _spatial_tokens_bhwc(
            hf_outputs.last_hidden_state,
            n_prefix_tokens=backbone.n_prefix_tokens,
            height=pixel_values.shape[-2],
            width=pixel_values.shape[-1],
            patch_size=backbone.get_patch_size(),
        )
        timm_last_hidden_state_bhwc = _timm_last_hidden_state_bhwc(
            timm_model,
            pixel_values,
        )
    timm_no_antialias_last_hidden_state_bhwc = (
        _compute_timm_last_hidden_state_bhwc_with_pos_embed_antialias(
            timm_model_name,
            pixel_values,
            antialias=False,
            monkeypatch=monkeypatch,
        )
    )

    hf_stats = _distance_stats(hidden_states_bhwc[-1], hf_last_hidden_state_bhwc)
    timm_stats = _distance_stats(hidden_states_bhwc[-1], timm_last_hidden_state_bhwc)
    timm_no_antialias_stats = _distance_stats(
        hidden_states_bhwc[-1],
        timm_no_antialias_last_hidden_state_bhwc,
    )
    print(
        f"{backbone_cls.__name__} 448 vs Hugging Face: "
        f"{_format_distance_stats(hf_stats)}"
    )
    print(
        f"{backbone_cls.__name__} 448 vs timm: " f"{_format_distance_stats(timm_stats)}"
    )
    print(
        f"{backbone_cls.__name__} 448 vs timm antialias=False: "
        f"{_format_distance_stats(timm_no_antialias_stats)}"
    )

    if backbone_cls is DINOv2ViT:
        assert timm_stats["rel_l2"] < hf_stats["rel_l2"], (
            "DINOv2ViT should use timm-style antialiased positional embedding "
            "resize, making it closer to timm than to Hugging Face interpolation. "
            f"HF: {_format_distance_stats(hf_stats)}; "
            f"timm: {_format_distance_stats(timm_stats)}"
        )
        assert timm_stats["rel_l2"] < timm_no_antialias_stats["rel_l2"], (
            "DINOv2ViT should be closer to timm antialias=True than timm "
            "antialias=False at interpolated resolutions. "
            f"timm: {_format_distance_stats(timm_stats)}; "
            f"timm antialias=False: {_format_distance_stats(timm_no_antialias_stats)}"
        )
