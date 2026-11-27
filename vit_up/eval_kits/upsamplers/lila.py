"""Released LILA decoder features (google-research/lila, CVPR 2026) as an eval-kit upsampler."""
from __future__ import annotations

from pathlib import Path
import subprocess

import torch

from .base import UpsamplerBase

LILA_REPO_URL = "https://github.com/google-research/google-research.git"
# Commit whose LILA code was reviewed for this integration.
LILA_COMMIT = "ee4298226f21649f45c0d0427f7c0f3f682d2a20"
# Hugging Face equivalents of LILA's frozen DINOv2 encoders (same weights).
ENCODERS = {"vits14": "facebook/dinov2-small", "vitb14": "facebook/dinov2-base", "vitl14": "facebook/dinov2-large"}


def _ensure_lila_code(local_repo: Path, commit: str) -> Path:
    """Sparse-checkout only the `lila/` directory of google-research at a pinned commit."""
    lila_dir = local_repo / "lila"
    if (lila_dir / "hubconf.py").is_file():
        return lila_dir
    local_repo.mkdir(parents=True, exist_ok=True)
    git = lambda *args: subprocess.run(["git", "-C", str(local_repo), *args], check=True)
    git("init", "-q")
    git("remote", "add", "origin", LILA_REPO_URL)
    git("sparse-checkout", "set", "lila")
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
    git("checkout", "-q", "FETCH_HEAD")
    return lila_dir


class LILAUpsampler(UpsamplerBase):
    """LILA's layer-normalized decoder features at the input resolution (its `sem_head`).

    The decoder runs on the frozen DINOv2 encoder of the same checkpoint, at the input's
    own aspect ratio, exactly as in LILA's evaluation scripts.
    """

    native_aspect = True      # takes any patch-aligned input; no square padding needed
    normalized_output = True  # LILA already layer-normalizes its decoder features

    def __init__(self, entrypoint: str = "lila_dinov2_vits14", name: str = "lila",
                 local_repo: str | None = None, commit: str = LILA_COMMIT, feature_dim: int | None = None,
                 **kwargs):
        super().__init__(name=name)
        del feature_dim, kwargs
        repo = Path(local_repo or Path(torch.hub.get_dir()) / "google-research_lila").expanduser()
        lila_dir = _ensure_lila_code(repo, commit)
        self.model = torch.hub.load(str(lila_dir), entrypoint, pretrained=True, source="local",
                                    trust_repo=True).eval().requires_grad_(False)
        self.patch_size = self.model.patch_size
        self.backbone_model_name = ENCODERS[entrypoint.split("_")[2]]

    @torch.no_grad()
    def forward(self, images, output_size=None, input_size=None):
        features, _ = self.model.sem_head(images, with_norm=True, with_enc_norm=True)
        return features.permute(0, 2, 3, 1).float()
