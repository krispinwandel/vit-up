import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from vit_up.model import load_backbone
from .base import UpsamplerBase, as_size


def _pop_src_modules() -> dict:
    return {name: sys.modules.pop(name) for name in list(sys.modules) if name == "src" or name.startswith("src.")}


@contextmanager
def _naf_imports(repo_dir: Path):
    """NAF's hubconf imports a top-level `src` package; isolate it from other `src` packages."""
    old_path, old_modules = list(sys.path), _pop_src_modules()
    sys.path[:] = [p for p in old_path if not (Path(p or ".").resolve() / "src" / "__init__.py").is_file()]
    sys.path.insert(0, str(repo_dir))
    try:
        yield
    finally:
        sys.path[:] = old_path
        _pop_src_modules()
        sys.modules.update(old_modules)


def _load_naf(local_repo: str | None) -> torch.nn.Module:
    repo = Path(local_repo or Path(torch.hub.get_dir()) / "valeoai_NAF_main").expanduser()
    if repo.is_dir():
        with _naf_imports(repo):
            return torch.hub.load(str(repo), "naf", pretrained=True, device="cpu", source="local")
    if local_repo:
        raise FileNotFoundError(repo)
    _pop_src_modules()
    return torch.hub.load("valeoai/NAF", "naf", pretrained=True, device="cpu")


class NAFUpsampler(UpsamplerBase):
    """NAF (released checkpoint) on the last backbone layer, guided by the input image."""

    def __init__(self, backbone: str, name: str = "naf", local_repo: str | None = None, **kwargs):
        super().__init__(name=name)
        self.upsampler = _load_naf(local_repo).eval().requires_grad_(False)
        self.backbone = load_backbone(backbone).eval()
        self.backbone.compile()

    @torch.no_grad()
    def forward(self, images, output_size, input_size=None):
        images = self.resize(images, input_size)
        features = self.backbone_features(self.backbone, images)
        return self.to_bhwc(self.upsampler(images, features, as_size(output_size)))
