from __future__ import annotations

from typing import Optional, Tuple, Union, cast

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CanonicalParameterizedOrthonormalArtifactBasis2D(nn.Module):
    """
    Learns K continuous 2D Fourier frequencies on a canonical grid, usually 37x37
    for DINOv2 learned positional embeddings.

    The canonical sine/cosine basis is constructed on canonical_spatial_size,
    orthonormalized there, then bilinearly interpolated to the input feature
    resolution.

    This means that when target resolution is larger, e.g. 64x64, the learned
    canonical artifact patterns are spatially stretched rather than regenerated
    as higher-frequency sinusoids.

    Input:
        x: [B, C, H, W] or [B, S, C, H, W]

    Output:
        x_clean: same shape as x
        basis_hw: [2K + include_dc, H, W]
        coeffs: [B, C, 2K + include_dc] or [B, S, C, 2K + include_dc]
    """

    def __init__(
        self,
        canonical_spatial_size: Tuple[int, int] = (37, 37),
        num_frequencies: int = 8,
        max_frequency: Optional[Union[float, Tuple[float, float]]] = None,
        strength: float = 1.0,
        learnable_strength: bool = False,
        include_dc: bool = False,
        reorthonormalize_after_interpolation: bool = True,
        preserve_channel_norm: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        hc, wc = canonical_spatial_size
        if hc <= 0 or wc <= 0:
            raise ValueError(
                f"Invalid canonical_spatial_size={canonical_spatial_size}."
            )
        if num_frequencies <= 0:
            raise ValueError(
                f"num_frequencies must be positive, got {num_frequencies}."
            )

        self.canonical_spatial_size = (hc, wc)
        self.num_frequencies = int(num_frequencies)
        self.include_dc = bool(include_dc)
        self.reorthonormalize_after_interpolation = bool(
            reorthonormalize_after_interpolation
        )
        self.preserve_channel_norm = bool(preserve_channel_norm)
        self.eps = float(eps)

        if max_frequency is None:
            # Frequencies are defined on normalized canonical coordinates [-1, 1].
            # This is the maximum number of cycles over the normalized coordinate
            # domain. Conservative default for 37x37.
            max_frequency = (hc / 2.0, wc / 2.0)

        if isinstance(max_frequency, tuple):
            max_fx, max_fy = max_frequency
        else:
            max_fx = max_fy = float(max_frequency)

        self.register_buffer(
            "max_frequency",
            torch.tensor([float(max_fx), float(max_fy)], dtype=torch.float32),
        )

        # Learn canonical frequencies. These are not tied to the runtime H,W.
        #
        # Actual frequencies:
        #   freq = max_frequency * tanh(freq_logits)
        #
        # Shape: [K, 2], columns are (u, v).
        self.freq_logits = nn.Parameter(torch.empty(self.num_frequencies, 2))
        nn.init.normal_(self.freq_logits, mean=0.0, std=0.25)

        if learnable_strength:
            init = torch.tensor(float(strength)).clamp(1e-4, 1.0 - 1e-4)
            self.strength_logit = nn.Parameter(torch.logit(init))
            self.register_buffer("_fixed_strength", torch.tensor(float(strength)))
        else:
            self.strength_logit = None
            self.register_buffer("_fixed_strength", torch.tensor(float(strength)))

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, hc),
            torch.linspace(-1.0, 1.0, wc),
            indexing="ij",
        )
        self.register_buffer("canonical_grid_x", xx)
        self.register_buffer("canonical_grid_y", yy)

    @property
    def num_basis(self) -> int:
        return 2 * self.num_frequencies + int(self.include_dc)

    @property
    def strength(self) -> torch.Tensor:
        if self.strength_logit is None:
            return self._fixed_strength
        return torch.sigmoid(self.strength_logit)

    def get_frequencies(self) -> torch.Tensor:
        """
        Returns learned canonical frequencies.

        Shape:
            [K, 2], columns are (u, v).
        """
        max_frequency = self.max_frequency.to(
            device=self.freq_logits.device,
            dtype=self.freq_logits.dtype,
        )
        return max_frequency * torch.tanh(self.freq_logits)

    def get_canonical_basis_pre_orthogonalization(self) -> torch.Tensor:
        """
        Builds sine/cosine Fourier atoms on the canonical grid.

        Returns:
            [2K + include_dc, Hc, Wc]
        """
        freqs = self.get_frequencies()
        dtype = self.freq_logits.dtype
        device = self.freq_logits.device

        x = self.canonical_grid_x.to(device=device, dtype=dtype)
        y = self.canonical_grid_y.to(device=device, dtype=dtype)

        u = freqs[:, 0].view(-1, 1, 1)
        v = freqs[:, 1].view(-1, 1, 1)

        phase = 2.0 * math.pi * (u * x.unsqueeze(0) + v * y.unsqueeze(0))

        cos_basis = torch.cos(phase)
        sin_basis = torch.sin(phase)

        # [K, 2, Hc, Wc] -> [2K, Hc, Wc]
        basis = torch.stack([cos_basis, sin_basis], dim=1).flatten(0, 1)

        if self.include_dc:
            dc = torch.ones(
                1,
                self.canonical_spatial_size[0],
                self.canonical_spatial_size[1],
                device=device,
                dtype=dtype,
            )
            basis = torch.cat([dc, basis], dim=0)

        return basis

    def _orthonormalize(self, basis: torch.Tensor) -> torch.Tensor:
        """
        Orthonormalizes basis fields over flattened spatial dimensions.

        Input:
            basis: [K, H, W]

        Output:
            basis_orth: [K, H, W]
        """
        k, h, w = basis.shape
        if k > h * w:
            raise ValueError(
                f"Cannot orthonormalize {k} basis maps on spatial size {h}x{w}."
            )

        basis_nk = basis.reshape(k, h * w).transpose(0, 1)  # [HW, K]
        q, _ = torch.linalg.qr(basis_nk, mode="reduced")
        return q.transpose(0, 1).reshape(k, h, w)

    def get_canonical_basis(self) -> torch.Tensor:
        """
        Returns orthonormalized basis on canonical grid.

        Shape:
            [2K + include_dc, Hc, Wc]
        """
        basis = self.get_canonical_basis_pre_orthogonalization()
        return self._orthonormalize(basis)

    def get_basis(
        self,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Returns basis at target spatial size.

        If spatial_size is None, returns canonical basis.

        Important:
            For non-canonical H,W, this interpolates the canonical basis fields.
            Therefore the artifact pattern stretches with resolution.

        Shape:
            [2K + include_dc, H, W]
        """
        canonical_basis = self.get_canonical_basis()

        if spatial_size is None or spatial_size == self.canonical_spatial_size:
            return canonical_basis

        h, w = spatial_size

        basis = F.interpolate(
            canonical_basis.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

        # Interpolation changes norms and can break orthogonality.
        # Re-orthonormalizing gives a true projector at the target resolution,
        # while still preserving the stretched canonical shapes.
        if self.reorthonormalize_after_interpolation:
            basis = self._orthonormalize(basis)
        else:
            basis = basis / (
                basis.flatten(1)
                .norm(dim=-1, keepdim=True)
                .clamp_min(self.eps)
                .view(-1, 1, 1)
            )

        return basis

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape

        if x.ndim == 4:
            has_sample_dim = False
            b, c, h, w = x.shape
            x_flat = x
        elif x.ndim == 5:
            has_sample_dim = True
            b, s, c, h, w = x.shape
            x_flat = x.reshape(b * s, c, h, w)
        else:
            raise ValueError(
                f"Expected x with shape [B,C,H,W] or [B,S,C,H,W], got {x.shape}."
            )

        basis = self.get_basis((h, w)).to(device=x_flat.device, dtype=x_flat.dtype)
        strength = self.strength.to(device=x_flat.device, dtype=x_flat.dtype)

        coeffs = torch.einsum("bchw,khw->bck", x_flat, basis)
        projection = torch.einsum("bck,khw->bchw", coeffs, basis)

        x_clean = x_flat - strength * projection

        if self.preserve_channel_norm:
            old_norm = x_flat.norm(dim=(-2, -1), keepdim=True)
            new_norm = x_clean.norm(dim=(-2, -1), keepdim=True)
            x_clean = x_clean * (old_norm / (new_norm + self.eps))

        if has_sample_dim:
            b, s, c, h, w = x.shape
            x_clean = x_clean.reshape(original_shape)
            coeffs = coeffs.reshape(b, s, c, self.num_basis)

        return x_clean, basis, coeffs

    def orthogonality_error(
        self,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        basis = self.get_basis(spatial_size)
        k = basis.shape[0]
        gram = basis.flatten(1) @ basis.flatten(1).T
        eye = torch.eye(k, device=basis.device, dtype=basis.dtype)
        return (gram - eye).square().mean()

    def spectral_energy(
        self,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        basis = self.get_basis(spatial_size)
        fft = torch.fft.rfft2(basis.float(), dim=(-2, -1), norm="ortho")
        return fft.abs().square()


class LayerwiseCanonicalArtifactSmoother(nn.Module):
    """
    One canonical Fourier artifact smoother per ViT layer.

    Each block learns a canonical 37x37 basis and interpolates it to the
    corresponding hidden-state resolution during forward.
    """

    def __init__(
        self,
        num_layers: int,
        canonical_spatial_size: Tuple[int, int] = (37, 37),
        num_frequencies: int = 8,
        max_frequency: Optional[Union[float, Tuple[float, float]]] = None,
        strength: float = 1.0,
        learnable_strength: bool = False,
        include_dc: bool = False,
        reorthonormalize_after_interpolation: bool = True,
        preserve_channel_norm: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.num_layers = int(num_layers)

        self.blocks = nn.ModuleList(
            [
                CanonicalParameterizedOrthonormalArtifactBasis2D(
                    canonical_spatial_size=canonical_spatial_size,
                    num_frequencies=num_frequencies,
                    max_frequency=max_frequency,
                    strength=strength,
                    learnable_strength=learnable_strength,
                    include_dc=include_dc,
                    reorthonormalize_after_interpolation=(
                        reorthonormalize_after_interpolation
                    ),
                    preserve_channel_norm=preserve_channel_norm,
                    eps=eps,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, hidden_states):
        input_was_tensor = isinstance(hidden_states, torch.Tensor)

        if input_was_tensor:
            if hidden_states.ndim not in (5, 6):
                raise ValueError(
                    "Tensor hidden_states must be [L,B,C,H,W] or [L,B,S,C,H,W]. "
                    f"Got {hidden_states.shape}."
                )
            if hidden_states.shape[0] != self.num_layers:
                raise ValueError(
                    f"Expected L={self.num_layers}, got {hidden_states.shape[0]}."
                )
            hidden_list = [hidden_states[i] for i in range(self.num_layers)]
        else:
            hidden_list = list(hidden_states)
            if len(hidden_list) != self.num_layers:
                raise ValueError(
                    f"Expected {self.num_layers} hidden states, got {len(hidden_list)}."
                )

        smoothed = []
        bases = []

        for block, h in zip(self.blocks, hidden_list):
            h_smooth, basis, _ = block(h)
            smoothed.append(h_smooth)
            bases.append(basis)

        if input_was_tensor:
            smoothed = torch.stack(smoothed, dim=0)

        return smoothed, bases

    def get_canonical_frequencies(self):
        return [
            cast(
                CanonicalParameterizedOrthonormalArtifactBasis2D, block
            ).get_frequencies()
            for block in self.blocks
        ]

    def get_canonical_bases(self):
        return [
            cast(
                CanonicalParameterizedOrthonormalArtifactBasis2D, block
            ).get_canonical_basis()
            for block in self.blocks
        ]

    # def orthogonality_loss(self) -> torch.Tensor:
    #     losses = [block.orthogonality_error() for block in self.blocks]
    #     return torch.stack(losses).mean()
