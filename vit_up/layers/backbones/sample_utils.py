from typing import Optional, List, Tuple, Any, cast
import numpy as np
import math
import torch
import torch.nn.functional as F
from contextlib import nullcontext

# from vit_up.layers.backbones.dinov2_vit import DinoViTBackboneBase


def compute_backbone_hidden_states(
    backbone: Any,
    pixel_values: torch.Tensor,
    img_size: Optional[int] = None,
    window_size: int = 0,
) -> List[torch.Tensor]:
    if img_size is not None and int(img_size) <= 0:
        raise ValueError(f"img_size must be > 0 when provided. Got {img_size}.")

    backbone_input = pixel_values
    if img_size is not None and tuple(pixel_values.shape[-2:]) != (
        img_size,
        img_size,
    ):
        backbone_input = F.interpolate(
            pixel_values,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )

    use_autocast = backbone_input.device.type in (
        "cuda",
        "xpu",
    ) and backbone_input.dtype in (torch.float16, torch.bfloat16)
    autocast_ctx = (
        torch.autocast(
            dtype=backbone_input.dtype,
            device_type=backbone_input.device.type,
        )
        if use_autocast
        else nullcontext()
    )
    with autocast_ctx:
        out = backbone(
            pixel_values=backbone_input,
            window_size=window_size,
        )
    return cast(List[torch.Tensor], out)


# @staticmethod
def select_hidden_layers(
    hidden_states: List[torch.Tensor],
    layer_indices: List[int],
) -> List[torch.Tensor]:
    max_idx = len(hidden_states) - 1
    selected: List[torch.Tensor] = []
    for layer_idx in layer_indices:
        idx = int(layer_idx)
        if idx < 0 or idx > max_idx:
            raise ValueError(
                "layer_indices contains out-of-range values. "
                f"Expected index in [0, {max_idx}], got {idx}."
            )
        selected.append(hidden_states[idx])
    return selected


def _ordered_unique(values: list[int]) -> list[int]:
    seen = set()
    out = []
    for v in values:
        v = int(v)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _make_geometric_integer_radii(
    max_radius: int,
    n_radii: int,
    *,
    max_trials: int = 128,
) -> list[int]:
    """
    Make approximately log/geometric integer radii.

    Lower ratio => more even / less aggressively logarithmic.
    Higher ratio => more aggressively logarithmic.

    Returns exactly n_radii unique radii if possible.
    """
    if max_radius <= 0:
        return []
    if n_radii <= 0:
        return []
    if n_radii > max_radius:
        raise ValueError(
            f"Cannot make {n_radii} unique positive radii from [1, {max_radius}]."
        )
    if n_radii == 1:
        return [1]

    # Initial ratio so that 1, q, q^2, ..., q^(n_radii-1) reaches max_radius.
    q0 = max_radius ** (1.0 / float(n_radii - 1))

    best: list[int] = []

    # Try progressively smaller q. Smaller q gives denser/smoother early radii.
    # If q gets too small and duplicates appear near 1, fallback below fills.
    for trial in range(max_trials):
        alpha = 1.0 - 0.75 * trial / max(max_trials - 1, 1)
        q = 1.0 + alpha * (q0 - 1.0)

        radii = [
            int(round(q**i)) for i in range(n_radii * 4)  # oversample, then deduplicate
        ]
        radii = [r for r in radii if 1 <= r <= max_radius]
        radii = _ordered_unique(radii)

        if len(radii) > len(best):
            best = radii

        if len(radii) >= n_radii:
            return radii[:n_radii]

    # Deterministic fallback: keep geometric candidates first, then fill linearly.
    fallback = _ordered_unique(best + list(range(1, max_radius + 1)))
    return fallback[:n_radii]


# def make_diag_antithetic_log_offsets(
#     n_tokens_per_side: int,
#     n_samples: int,
#     device: torch.device,
#     *,
#     include_zero: bool = False,
#     random_phase: bool = False,
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#     """
#     Diagonal antithetic geometric/log-like cyclic roll offsets.

#     Returns:
#         offsets: LongTensor [n_samples, 2]
#         weights: None
#     """
#     N = int(n_tokens_per_side)
#     K = int(n_samples)

#     if N <= 0:
#         raise ValueError(f"n_tokens_per_side must be positive, got {N}.")
#     if K <= 0:
#         raise ValueError(f"n_samples must be positive, got {K}.")
#     if K > N:
#         raise ValueError(
#             f"Cannot return {K} unique diagonal offsets on a cyclic grid of size {N}."
#         )

#     values: list[int] = []

#     if include_zero:
#         values.append(0)

#     max_radius = N // 2

#     n_pairs_needed = (K - len(values) + 1) // 2
#     candidate_radii = _make_geometric_integer_radii(
#         max_radius=max_radius,
#         n_radii=n_pairs_needed,
#     )

#     used_offsets = set(values)

#     for r in candidate_radii:
#         if len(values) >= K:
#             break

#         r = int(r)

#         if 2 * r == N:
#             candidates = [r]
#         else:
#             candidates = [(-r) % N, r % N]

#         for v in candidates:
#             v = int(v) % N
#             if v not in used_offsets and len(values) < K:
#                 values.append(v)
#                 used_offsets.add(v)

#     # If half-period duplicate prevented enough offsets, fill with remaining radii.
#     if len(values) < K:
#         for r in range(1, max_radius + 1):
#             if len(values) >= K:
#                 break

#             if 2 * r == N:
#                 candidates = [r]
#             else:
#                 candidates = [(-r) % N, r % N]

#             for v in candidates:
#                 v = int(v) % N
#                 if v not in used_offsets and len(values) < K:
#                     values.append(v)
#                     used_offsets.add(v)

#     if len(values) != K:
#         raise RuntimeError(f"Constructed {len(values)} offsets, requested {K}.")

#     t = torch.tensor(values, device=device, dtype=torch.long)

#     if random_phase:
#         phase = torch.randint(0, N, (), device=device)
#         t = (t + phase) % N

#     return torch.stack([t, t], dim=-1), None

from typing import Optional, Tuple
import torch


def make_diag_antithetic_log_offsets(
    n_tokens_per_side: int,
    n_samples: int,
    device: torch.device,
    *,
    base_pe_size: int = 37,
    base_max_radius: float = 8.0,
    include_zero: bool = False,
    random_phase: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Diagonal antithetic log-spaced cyclic roll offsets.

    Radii are defined in the learned-PE coordinate system, then scaled to the
    current token grid:

        r_N = round(r_37 * N / base_pe_size)

    Returns:
        offsets: LongTensor [n_samples, 2]
        weights: None
    """
    N = int(n_tokens_per_side)
    K = int(n_samples)

    if N <= 0:
        raise ValueError(f"n_tokens_per_side must be positive, got {N}.")
    if K <= 0:
        raise ValueError(f"n_samples must be positive, got {K}.")
    if K > N:
        raise ValueError(
            f"Cannot return {K} unique offsets on cyclic grid of size {N}."
        )
    if base_pe_size <= 0:
        raise ValueError(f"base_pe_size must be positive, got {base_pe_size}.")
    if base_max_radius <= 0:
        raise ValueError(f"base_max_radius must be positive, got {base_max_radius}.")

    values: list[int] = []
    used_offsets: set[int] = set()

    if include_zero:
        values.append(0)
        used_offsets.add(0)

    n_radii = (K - len(values) + 1) // 2
    if n_radii <= 0:
        t = torch.tensor(values[:K], device=device, dtype=torch.long)
        return torch.stack([t, t], dim=-1), None

    # Log curve in the 37-grid coordinate system.
    # Example: n_radii=4, base_max_radius=8 -> approximately [1, 2, 4, 8].
    base_radii = torch.logspace(
        start=0.0,
        end=torch.log2(torch.tensor(float(base_max_radius))).item(),
        steps=n_radii,
        base=2.0,
        device=device,
    )

    # Scale from PE-grid coordinates to current token-grid coordinates.
    scale = float(N) / float(base_pe_size)
    radii = torch.round(base_radii * scale).long()

    # Avoid zero after scaling/rounding at small N.
    radii = torch.clamp(radii, min=1, max=N // 2)

    # Deduplicate radii while preserving order.
    unique_radii: list[int] = []
    seen_radii: set[int] = set()
    for r in radii.tolist():
        r = int(r)
        if r not in seen_radii:
            unique_radii.append(r)
            seen_radii.add(r)

    # Minimal fill: if scaling caused duplicates, add nearby unused radii.
    candidate_fill = list(range(1, N // 2 + 1))
    for r in candidate_fill:
        if len(unique_radii) >= n_radii:
            break
        if r not in seen_radii:
            unique_radii.append(r)
            seen_radii.add(r)

    for r in unique_radii:
        if len(values) >= K:
            break

        r = int(r)

        if 2 * r == N:
            candidates = [r]
        else:
            candidates = [(-r) % N, r % N]

        for v in candidates:
            v = int(v) % N
            if v not in used_offsets and len(values) < K:
                values.append(v)
                used_offsets.add(v)

    # Final fallback if N/2 duplicate caused one missing sample.
    for r in range(1, N // 2 + 1):
        if len(values) >= K:
            break

        candidates = [r] if 2 * r == N else [(-r) % N, r % N]
        for v in candidates:
            v = int(v) % N
            if v not in used_offsets and len(values) < K:
                values.append(v)
                used_offsets.add(v)

    if len(values) != K:
        raise RuntimeError(f"Constructed {len(values)} offsets, requested {K}.")

    t = torch.tensor(values, device=device, dtype=torch.long)
    # t = torch.tensor([-1 % N, 1], device=device, dtype=torch.long)

    # print("t:", t)

    if random_phase:
        phase = torch.randint(-N // 10, N // 10, (), device=device)
        t = (t + phase) % N

    return torch.stack([t, t], dim=-1), None


from typing import Optional, Tuple
import torch


def make_diag_antithetic_nearest37_offsets(
    n_tokens_per_side: int,
    n_samples: int,
    device: torch.device,
    *,
    base_pe_size: int = 37,
    include_zero: bool = False,
    random_phase: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Diagonal antithetic offsets using the k closest radii in learned-PE space.

    Idea:
        Define nearest integer radii in the 37x37 PE coordinate system:

            r_37 = 1, 2, 3, ...

        Then scale them to the current token grid only if the current grid is larger:

            scale = max(1, N / base_pe_size)
            r_N = round(r_37 * scale)

    This means:
        N <= 37: use local token radii directly, e.g. ±1, ±2, ±3, ±4.
        N >  37: enlarge the local window according to PE-coordinate scaling.

    Returns:
        offsets: LongTensor [n_samples, 2]
            Each row is a cyclic diagonal token offset (t, t).
        weights:
            None. Intended for uniform averaging.
    """
    N = int(n_tokens_per_side)
    K = int(n_samples)

    if N <= 0:
        raise ValueError(f"n_tokens_per_side must be positive, got {N}.")
    if K <= 0:
        raise ValueError(f"n_samples must be positive, got {K}.")
    if K > N:
        raise ValueError(
            f"Cannot return {K} unique diagonal offsets on cyclic grid of size {N}."
        )
    if base_pe_size <= 0:
        raise ValueError(f"base_pe_size must be positive, got {base_pe_size}.")

    values: list[int] = []
    used_offsets: set[int] = set()

    if include_zero:
        values.append(0)
        used_offsets.add(0)

    n_radii_needed = (K - len(values) + 1) // 2

    # Only scale up. For N <= 37, keep the nearest local token shifts.
    scale = max(1.0, float(N) / float(base_pe_size))

    # Generate more base radii than needed because rounding can create duplicates.
    max_radius = N // 2
    max_base_radius = max(
        base_pe_size, int(torch.ceil(torch.tensor(max_radius / scale)).item()) + 4
    )

    base_radii = torch.arange(
        1,
        max_base_radius + 1,
        device=device,
        dtype=torch.float32,
    )

    radii = torch.round(base_radii * scale).long()
    radii = torch.clamp(radii, min=1, max=max_radius)

    # Deduplicate radii while preserving closeness order in 37-space.
    unique_radii: list[int] = []
    seen_radii: set[int] = set()
    for r in radii.tolist():
        r = int(r)
        if r not in seen_radii:
            unique_radii.append(r)
            seen_radii.add(r)
        if len(unique_radii) >= n_radii_needed:
            break

    # Minimal fallback if rounding/clamping did not produce enough radii.
    for r in range(1, max_radius + 1):
        if len(unique_radii) >= n_radii_needed:
            break
        if r not in seen_radii:
            unique_radii.append(r)
            seen_radii.add(r)

    for r in unique_radii:
        if len(values) >= K:
            break

        if 2 * r == N:
            candidates = [r]
        else:
            candidates = [(-r) % N, r % N]

        for v in candidates:
            v = int(v) % N
            if v not in used_offsets and len(values) < K:
                values.append(v)
                used_offsets.add(v)

    # Final fallback in case N/2 produced only one antithetic offset.
    for r in range(1, max_radius + 1):
        if len(values) >= K:
            break

        candidates = [r] if 2 * r == N else [(-r) % N, r % N]
        for v in candidates:
            v = int(v) % N
            if v not in used_offsets and len(values) < K:
                values.append(v)
                used_offsets.add(v)

    if len(values) != K:
        raise RuntimeError(f"Constructed {len(values)} offsets, requested {K}.")

    t = torch.tensor(values, device=device, dtype=torch.long)
    print("t:", t)

    if random_phase:
        phase = torch.randint(0, N, (), device=device)
        t = (t + phase) % N

    print("t after phase:", t)

    return torch.stack([t, t], dim=-1), None


# def make_diag_antithetic_log_offsets(
#     n_tokens_per_side: int,
#     n_samples: int,
#     device: torch.device,
#     *,
#     include_zero: bool = False,
#     random_phase: bool = False,
# ) -> torch.Tensor:
#     """
#     Diagonal antithetic log-spaced cyclic roll offsets.

#     Returns:
#         offsets: LongTensor [M, 2], with M == n_samples.
#                  Each row is (t, t), where t is a cyclic token offset.

#     Example for N=16, n_samples=8, include_zero=False:
#         roughly [15, 1, 14, 2, 12, 4, 10, 6]
#         corresponding to [-1, +1, -2, +2, -4, +4, -6, +6] mod 16.
#     """
#     N = int(n_tokens_per_side)
#     K = int(n_samples)

#     if N <= 0:
#         raise ValueError(f"n_tokens_per_side must be positive, got {N}.")
#     if K <= 0:
#         raise ValueError(f"n_samples must be positive, got {K}.")
#     if K > N:
#         raise ValueError(
#             f"Cannot return {K} unique diagonal offsets on a cyclic grid of size {N}."
#         )

#     values = []

#     if include_zero:
#         values.append(0)

#     # Positive cyclic radii. We avoid generating both +N/2 and -N/2 because
#     # they are identical modulo N.
#     max_radius = N // 2

#     # Log-spaced candidate radii, then rounded to unique integers.
#     # This gives [1, 2, 4, 8] for N=16 before half-period handling.
#     n_pairs_needed = (K - len(values) + 1) // 2
#     n_radii = max(n_pairs_needed, 1)

#     log_radii = torch.logspace(
#         start=0.0,
#         end=(
#             float(torch.log2(torch.tensor(max_radius, dtype=torch.float32)).item())
#             if max_radius > 1
#             else 0.0
#         ),
#         steps=max(n_radii * 3, 8),  # oversample, then deduplicate
#         base=2.0,
#     )

#     candidate_radii = torch.round(log_radii).long().tolist()
#     candidate_radii = [r for r in candidate_radii if 1 <= r <= max_radius]

#     print(len(candidate_radii), candidate_radii)

#     # Add linear fallback radii to guarantee enough candidates.
#     candidate_radii += list(range(1, max_radius + 1))

#     seen_radii = set()
#     for r in candidate_radii:
#         if r in seen_radii:
#             continue
#         seen_radii.add(r)

#         if len(values) >= K:
#             break

#         if 2 * r == N:
#             # +r == -r mod N. Only one unique offset.
#             values.append(r)
#         else:
#             # Antithetic pair: -r, +r.
#             values.extend([(-r) % N, r % N])

#         # Preserve order, remove accidental duplicates, truncate later.
#         deduped = []
#         seen_values = set()
#         for v in values:
#             v = int(v) % N
#             if v not in seen_values:
#                 seen_values.add(v)
#                 deduped.append(v)
#         values = deduped

#     if len(values) < K:
#         raise RuntimeError(
#             f"Could only construct {len(values)} unique offsets, requested {K}."
#         )

#     values = values[:K]

#     t = torch.tensor(values, device=device, dtype=torch.long)

#     if random_phase:
#         phase = torch.randint(0, N, (), device=device)
#         t = (t + phase) % N

#     return torch.stack([t, t], dim=-1), None


def make_diag_antithetic_uniform_offsets(
    n_tokens_per_side: int,
    n_samples: int,
    device: torch.device,
    *,
    include_zero: bool = False,
    random_phase: bool = False,
) -> Tuple[torch.Tensor, None]:
    """
    Diagonal antithetic uniformly-spaced cyclic roll offsets.

    Returns:
        offsets: LongTensor [n_samples, 2]
                 Each row is (t, t), where t is a cyclic token offset.

    Example for N=16, n_samples=8, include_zero=False:
        radii ~= [1, 3, 5, 7]
        offsets = [-1, +1, -3, +3, -5, +5, -7, +7] mod 16
                = [15, 1, 13, 3, 11, 5, 9, 7]
    """
    N = int(n_tokens_per_side)
    K = int(n_samples)

    if N <= 0:
        raise ValueError(f"n_tokens_per_side must be positive, got {N}.")
    if K <= 0:
        raise ValueError(f"n_samples must be positive, got {K}.")
    if K > N:
        raise ValueError(
            f"Cannot return {K} unique diagonal offsets on a cyclic grid of size {N}."
        )

    values: List[int] = []

    if include_zero:
        values.append(0)

    remaining = K - len(values)
    if remaining <= 0:
        t = torch.tensor(values[:K], device=device, dtype=torch.long)
        return torch.stack([t, t], dim=-1), None

    max_radius = N // 2

    # Number of antithetic radius candidates needed.
    # Each radius usually contributes 2 samples: -r and +r.
    n_radii_needed = (remaining + 1) // 2

    # Prefer radii strictly below N/2, because r=N/2 has no distinct antithetic pair.
    usable_max_radius = max_radius - 1 if N % 2 == 0 else max_radius

    if usable_max_radius <= 0:
        # N=1 or degenerate tiny case.
        candidates = [0]
    else:
        # Midpoint-uniform radii in [1, usable_max_radius].
        # This avoids both over-emphasizing tiny shifts and hitting N/2 too early.
        raw = (
            (torch.arange(n_radii_needed, dtype=torch.float32) + 0.5)
            * usable_max_radius
            / n_radii_needed
        )
        candidates = torch.floor(raw).long().clamp(1, usable_max_radius).tolist()

        # Deduplicate while preserving order.
        deduped = []
        seen = set()
        for r in candidates:
            r = int(r)
            if r not in seen:
                seen.add(r)
                deduped.append(r)
        candidates = deduped

        # Fallback to guarantee enough unique radii.
        # Use increasing radii not already selected.
        for r in range(1, usable_max_radius + 1):
            if len(candidates) >= n_radii_needed:
                break
            if r not in seen:
                seen.add(r)
                candidates.append(r)

    for r in candidates:
        if len(values) >= K:
            break

        r = int(r) % N

        if r == 0:
            if 0 not in values:
                values.append(0)
            continue

        if 2 * r == N:
            # +r and -r are identical modulo N.
            if r not in values:
                values.append(r)
        else:
            neg = (-r) % N
            pos = r % N

            if neg not in values and len(values) < K:
                values.append(neg)
            if pos not in values and len(values) < K:
                values.append(pos)

    # If K is odd and include_zero=False, or if rounding/dedup left a gap,
    # fill remaining slots with uniform unused offsets.
    if len(values) < K:
        used = set(values)
        fill = (
            torch.floor((torch.arange(N, dtype=torch.float32) + 0.5) * N / N)
            .long()
            .tolist()
        )

        for v in fill:
            v = int(v) % N
            if v not in used:
                used.add(v)
                values.append(v)
                if len(values) == K:
                    break

    if len(values) != K:
        raise RuntimeError(f"Constructed {len(values)} offsets, requested {K}.")

    t = torch.tensor(values, device=device, dtype=torch.long)

    if random_phase:
        phase = torch.randint(0, N, (), device=device)
        t = (t + phase) % N

        # Re-deduplicate after phase is unnecessary because cyclic shift preserves uniqueness.

    return torch.stack([t, t], dim=-1), None


def make_diag_scaled_gauss_legendre_offsets(
    n_tokens_per_side: int,
    n_samples: int,
    device: torch.device,
    *,
    base_pe_size: int = 37,
    base_radius: float = 12.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns diagonal cyclic token-roll offsets and normalized Gauss-Legendre weights.

    The continuous Gauss-Legendre nodes xi_i in [-1, 1] are scaled to a
    resolution-dependent local PE window

        R_N = base_radius * n_tokens_per_side / base_pe_size

    and rounded to integer token offsets.

    Returns:
        offsets: LongTensor [n_samples, 2]
            Diagonal token offsets (t_i, t_i), modulo n_tokens_per_side.
        weights: Tensor [n_samples]
            Normalized quadrature weights, sum to 1.
    """
    N = int(n_tokens_per_side)
    K = int(n_samples)

    if N <= 0:
        raise ValueError(f"n_tokens_per_side must be positive, got {N}.")
    if K <= 0:
        raise ValueError(f"n_samples must be positive, got {K}.")
    if K > N:
        raise ValueError(
            f"n_samples={K} cannot exceed n_tokens_per_side={N} "
            "if unique diagonal offsets are expected."
        )
    if base_pe_size <= 0:
        raise ValueError(f"base_pe_size must be positive, got {base_pe_size}.")
    if base_radius <= 0:
        raise ValueError(f"base_radius must be positive, got {base_radius}.")

    # Continuous Gauss-Legendre nodes/weights on [-1, 1].
    nodes_np, weights_np = np.polynomial.legendre.leggauss(K)

    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)

    # Radius in current token-grid coordinates.
    radius = float(base_radius) * float(N) / float(base_pe_size)

    # Signed integer token offsets.
    signed_offsets = torch.round(radius * nodes).to(torch.long)

    # Convert signed offsets to cyclic offsets.
    t = signed_offsets % N

    # Diagonal offsets: (dy, dx) = (t, t).
    offsets = torch.stack([t, t], dim=-1)

    # Normalize for weighted averaging.
    weights = weights / weights.sum()

    return offsets, weights


from typing import Optional, Tuple
import numpy as np
import torch


def make_diag_multiscale_gauss_legendre_offsets(
    n_tokens_per_side: int,
    n_samples: int,
    device: torch.device,
    *,
    base_pe_size: int = 37,
    base_min_radius: float = 1.0,
    base_max_radius: float = 8.0,
    samples_per_scale: int = 2,
    random_phase: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Multi-scale diagonal Gauss-Legendre roll offsets.

    Uses several PE-coordinate window radii between base_min_radius and
    base_max_radius. Each window contributes samples_per_scale GL nodes.

    Returns:
        offsets: LongTensor [n_samples, 2]
        weights: FloatTensor [n_samples]
    """
    N = int(n_tokens_per_side)
    K = int(n_samples)

    if N <= 0:
        raise ValueError(f"n_tokens_per_side must be positive, got {N}.")
    if K <= 0:
        raise ValueError(f"n_samples must be positive, got {K}.")
    # if K > N:
    #     raise ValueError(f"Cannot return {K} unique offsets on cyclic grid size {N}.")
    if samples_per_scale <= 0:
        raise ValueError(
            f"samples_per_scale must be positive, got {samples_per_scale}."
        )
    if K % samples_per_scale != 0:
        raise ValueError(
            f"n_samples={K} must be divisible by samples_per_scale={samples_per_scale}."
        )

    n_scales = K // samples_per_scale

    # Log-spaced window radii in learned-PE coordinates.
    if n_scales == 1:
        base_radii = torch.tensor([base_max_radius], device=device, dtype=torch.float32)
    else:
        base_radii = torch.logspace(
            start=float(np.log2(base_min_radius)),
            end=float(np.log2(base_max_radius)),
            steps=n_scales,
            base=2.0,
            device=device,
        )

    # GL nodes/weights per local window.
    nodes_np, weights_np = np.polynomial.legendre.leggauss(samples_per_scale)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=torch.float32)
    gl_weights = torch.as_tensor(weights_np, device=device, dtype=torch.float32)

    scale = float(N) / float(base_pe_size)

    signed_offsets = []
    weights = []

    for R37 in base_radii:
        RN = float(R37.item()) * scale

        r = torch.round(RN * nodes).long()

        # Avoid zero offsets after rounding when using tiny windows.
        # For symmetric 2-point GL this usually only matters at very small N/R.
        for j in range(r.numel()):
            val = int(r[j].item())
            if val == 0:
                val = -1 if float(nodes[j].item()) < 0 else 1
            signed_offsets.append(val)

        # Weight each scale equally; within each scale use GL weights.
        w = gl_weights / gl_weights.sum()
        w = w / float(n_scales)
        weights.extend([float(x) for x in w.tolist()])

    t = torch.tensor(signed_offsets, device=device, dtype=torch.long) % N
    weights_t = torch.tensor(weights, device=device, dtype=torch.float32)
    weights_t = weights_t / weights_t.sum()

    if random_phase:
        phase = torch.randint(0, N, (), device=device)
        t = (t + phase) % N

    offsets = torch.stack([t, t], dim=-1)
    return offsets, weights_t


def _compute_sampled_gt_features(
    backbone: Any,
    pixel_values: torch.Tensor,
    layer_indices: List[int],
    img_size: Optional[int],
    window_size: int,
    num_samples: int,
    sample_upscale: float,
) -> List[torch.Tensor]:
    if pixel_values.ndim != 4:
        raise ValueError(
            "pixel_values must be 4D (B, C, H, W) for sampled GT features. "
            f"Got shape: {tuple(pixel_values.shape)}"
        )
    if sample_upscale <= 0:
        raise ValueError(f"sample_upscale must be > 0. Got {sample_upscale}.")

    h_in, w_in = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
    if img_size is None:
        if h_in != w_in:
            raise ValueError(
                "img_size is required when pixel_values are not square. "
                f"Got H={h_in}, W={w_in}."
            )
        base_img_size = h_in
    else:
        base_img_size = int(img_size)
        if base_img_size <= 0:
            raise ValueError(f"img_size must be > 0 when provided. Got {img_size}.")

    patch_size = backbone.get_patch_size()
    if base_img_size % patch_size != 0:
        raise ValueError(
            "img_size must be divisible by patch_size for sampled GT features. "
            f"Got img_size={base_img_size}, patch_size={patch_size}."
        )

    base_input = pixel_values
    if tuple(pixel_values.shape[-2:]) != (base_img_size, base_img_size):
        base_input = F.interpolate(
            pixel_values,
            size=(base_img_size, base_img_size),
            mode="bilinear",
            align_corners=False,
        )

    canvas_size = int(float(sample_upscale) * float(base_img_size))
    canvas_size = (canvas_size // patch_size) * patch_size
    canvas_size = max(canvas_size, base_img_size)

    n_img_tokens = base_img_size // patch_size
    n_canvas_tokens = canvas_size // patch_size
    max_offset = n_canvas_tokens - n_img_tokens

    running_avg_layers: Optional[List[torch.Tensor]] = None
    for sample_idx in range(num_samples):
        if max_offset > 0:
            x_offset = int(
                torch.randint(
                    low=0,
                    high=max_offset + 1,
                    size=(1,),
                    device=base_input.device,
                ).item()
            )
            y_offset = int(
                torch.randint(
                    low=0,
                    high=max_offset + 1,
                    size=(1,),
                    device=base_input.device,
                ).item()
            )
        else:
            x_offset = 0
            y_offset = 0

        x_px = x_offset * patch_size
        y_px = y_offset * patch_size

        canvas = torch.zeros(
            (
                base_input.shape[0],
                base_input.shape[1],
                canvas_size,
                canvas_size,
            ),
            device=base_input.device,
            dtype=base_input.dtype,
        )
        canvas[:, :, y_px : y_px + base_img_size, x_px : x_px + base_img_size] = (
            base_input
        )

        sampled_hidden_states = compute_backbone_hidden_states(
            backbone=backbone,
            pixel_values=canvas,
            img_size=canvas_size,
            window_size=window_size,
        )
        sampled_layers = select_hidden_layers(
            sampled_hidden_states,
            layer_indices,
        )

        sampled_crops: List[torch.Tensor] = []
        for layer_hwc in sampled_layers:
            if layer_hwc.ndim != 4:
                raise ValueError(
                    "Expected sampled hidden state with shape (B, H, W, C). "
                    f"Got shape {tuple(layer_hwc.shape)}."
                )
            sampled_crops.append(
                layer_hwc[
                    :,
                    y_offset : y_offset + n_img_tokens,
                    x_offset : x_offset + n_img_tokens,
                    :,
                ]
            )

        if running_avg_layers is None:
            running_avg_layers = [layer.clone() for layer in sampled_crops]
            continue

        alpha = 1.0 / float(sample_idx + 1)
        for layer_idx, layer_crop in enumerate(sampled_crops):
            running_avg_layers[layer_idx].add_(
                (layer_crop - running_avg_layers[layer_idx]) * alpha
            )

    return running_avg_layers if running_avg_layers is not None else []


def _compute_sampled_gt_features_deterministic(
    backbone: Any,
    pixel_values: torch.Tensor,
    layer_indices: List[int],
    img_size: Optional[int],
    window_size: int,
    n_iters: int = 3,
) -> List[torch.Tensor]:
    """Deterministic sampling: tile the input into k x k grids for k=1..n_iters,
    run the backbone on each tiled canvas, extract per-tile crops and average
    all tile crops across all iterations to produce final features.

    This gives a deterministic set of samples (no randomness) useful for
    repeatable positional-denoising experiments.
    """
    if pixel_values.ndim != 4:
        raise ValueError(
            "pixel_values must be 4D (B, C, H, W) for deterministic sampled GT features. "
            f"Got shape: {tuple(pixel_values.shape)}"
        )
    n_iters = int(n_iters)
    if n_iters <= 0:
        raise ValueError(f"n_iters must be >= 1. Got {n_iters}.")

    h_in, w_in = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
    if img_size is None:
        if h_in != w_in:
            raise ValueError(
                "img_size is required when pixel_values are not square. "
                f"Got H={h_in}, W={w_in}."
            )
        base_img_size = h_in
    else:
        base_img_size = int(img_size)
        if base_img_size <= 0:
            raise ValueError(f"img_size must be > 0 when provided. Got {img_size}.")

    patch_size = backbone.get_patch_size()
    if base_img_size % patch_size != 0:
        raise ValueError(
            "img_size must be divisible by patch_size for deterministic sampled GT features. "
            f"Got img_size={base_img_size}, patch_size={patch_size}."
        )

    base_input = pixel_values
    if tuple(pixel_values.shape[-2:]) != (base_img_size, base_img_size):
        base_input = F.interpolate(
            pixel_values,
            size=(base_img_size, base_img_size),
            mode="bilinear",
            align_corners=False,
        )

    n_img_tokens = base_img_size // patch_size

    sum_layers: Optional[List[torch.Tensor]] = None
    total_tiles = 0

    for k in range(1, n_iters + 1):
        # canvas is k x k tiled copies of the base image
        canvas_size = k * base_img_size
        canvas = base_input.repeat(1, 1, k, k)

        sampled_hidden_states = compute_backbone_hidden_states(
            backbone=backbone,
            pixel_values=canvas,
            img_size=canvas_size,
            window_size=window_size,
        )
        sampled_layers = select_hidden_layers(
            sampled_hidden_states,
            layer_indices,
        )

        for layer_idx, layer_hwc in enumerate(sampled_layers):
            if layer_hwc.ndim != 4:
                raise ValueError(
                    "Expected sampled hidden state with shape (B, H, W, C). "
                    f"Got shape {tuple(layer_hwc.shape)}."
                )

        # initialize accumulator on first pass
        if sum_layers is None:
            sum_layers = [
                torch.zeros(
                    (
                        layer.shape[0],
                        n_img_tokens,
                        n_img_tokens,
                        layer.shape[-1],
                    ),
                    device=layer.device,
                    dtype=layer.dtype,
                )
                for layer in sampled_layers
            ]

        # iterate over tile positions and accumulate per-tile crops
        for i in range(k):
            for j in range(k):
                for li, layer_hwc in enumerate(sampled_layers):
                    crop = layer_hwc[
                        :,
                        i * n_img_tokens : (i + 1) * n_img_tokens,
                        j * n_img_tokens : (j + 1) * n_img_tokens,
                        :,
                    ]
                    sum_layers[li].add_(crop)
                total_tiles += 1

    if sum_layers is None or total_tiles == 0:
        return []

    avg_layers = [s / float(total_tiles) for s in sum_layers]
    return avg_layers


def _compute_sampled_gt_features_deterministic_fixed_canvas(
    backbone: Any,
    pixel_values: torch.Tensor,
    layer_indices: List[int],
    img_size: Optional[int],
    window_size: int,
    n_iters: int = 3,
    sample_upscale: float = 1.5,
    use_adaptive_canvas_size: bool = False,
    n_samples_per_iter: int = 1,
) -> List[torch.Tensor]:
    if pixel_values.ndim != 4:
        raise ValueError(
            "pixel_values must be 4D (B, C, H, W) for deterministic sampled GT features. "
            f"Got shape: {tuple(pixel_values.shape)}"
        )
    n_iters = int(n_iters)
    if n_iters <= 0:
        raise ValueError(f"n_iters must be >= 1. Got {n_iters}.")
    if sample_upscale <= 0:
        raise ValueError(f"sample_upscale must be > 0. Got {sample_upscale}.")

    h_in, w_in = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
    if img_size is None:
        if h_in != w_in:
            raise ValueError(
                "img_size is required when pixel_values are not square. "
                f"Got H={h_in}, W={w_in}."
            )
        base_img_size = h_in
    else:
        base_img_size = int(img_size)
        if base_img_size <= 0:
            raise ValueError(f"img_size must be > 0 when provided. Got {img_size}.")

    patch_size = backbone.get_patch_size()
    if base_img_size % patch_size != 0:
        raise ValueError(
            "img_size must be divisible by patch_size for deterministic sampled GT features. "
            f"Got img_size={base_img_size}, patch_size={patch_size}."
        )

    base_input = pixel_values
    if tuple(pixel_values.shape[-2:]) != (base_img_size, base_img_size):
        base_input = F.interpolate(
            pixel_values,
            size=(base_img_size, base_img_size),
            mode="bilinear",
            align_corners=False,
        )

    h_canvas = int(float(sample_upscale) * float(base_img_size))
    w_canvas = int(float(sample_upscale) * float(base_img_size))

    n_img_tokens = base_img_size // patch_size

    running_num_layers: Optional[List[torch.Tensor]] = None
    running_den: Optional[List[torch.Tensor]] = None

    for k in range(1, n_iters + 1):
        # Compute canvas size per-iteration. Optionally use adaptive sizing.
        if use_adaptive_canvas_size:
            h_k = h_canvas * k + patch_size
            w_k = w_canvas * k + patch_size
        else:
            # Slightly shift token grid by using +patch_size after k-aligned quantization.
            h_k = (h_canvas // (patch_size * k)) * k * patch_size + patch_size
            w_k = (w_canvas // (patch_size * k)) * k * patch_size + patch_size

        repeated = base_input.repeat(1, 1, k, k)
        if tuple(repeated.shape[-2:]) != (h_k, w_k):
            repeated = F.interpolate(
                repeated,
                size=(h_k, w_k),
                mode="bilinear",
                align_corners=False,
            )

        # Paste repeated image into a same-size canvas with a sub-tile offset.
        # inner loop: run multiple random offsets per k-iteration
        accum_num_layers: Optional[List[torch.Tensor]] = None
        accum_den: Optional[List[torch.Tensor]] = None
        for s_idx in range(int(n_samples_per_iter)):
            max_off_y = int(h_k // k)
            max_off_x = int(w_k // k)
            off_y_px = int(
                torch.randint(
                    low=0,
                    high=max_off_y + 1,
                    size=(1,),
                    device=base_input.device,
                ).item()
            )
            off_x_px = int(
                torch.randint(
                    low=0,
                    high=max_off_x + 1,
                    size=(1,),
                    device=base_input.device,
                ).item()
            )
            # create a wrapped canvas by rolling the repeated tile so the
            # top-left of `repeated` appears at `(off_y_px, off_x_px)`;
            # this fills the formerly-empty regions with repeated content
            # instead of zeros.
            canvas = torch.roll(repeated, shifts=(off_y_px, off_x_px), dims=(2, 3))
            valid_h = h_k - off_y_px
            valid_w = w_k - off_x_px

            sampled_hidden_states = compute_backbone_hidden_states(
                backbone=backbone,
                pixel_values=canvas,
                img_size=None,
                window_size=window_size,
            )
            sampled_layers = select_hidden_layers(
                sampled_hidden_states,
                layer_indices,
            )

            grid_side = k * n_img_tokens
            coords_1d = (
                torch.arange(
                    grid_side, device=base_input.device, dtype=base_input.dtype
                )
                + 0.5
            ) / float(grid_side)
            coords_1d = coords_1d * 2.0 - 1.0
            grid_y, grid_x = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
            base_grid = torch.stack((grid_x, grid_y), dim=-1)

            first_layer = sampled_layers[0]
            if first_layer.ndim != 4:
                raise ValueError(
                    "Expected sampled hidden state with shape (B, H, W, C). "
                    f"Got shape {tuple(first_layer.shape)}."
                )

            bsz = int(first_layer.shape[0])
            h_tokens = int(first_layer.shape[1])
            w_tokens = int(first_layer.shape[2])

            # Shift sample grid directly in normalized canvas coordinates.
            dy = 2.0 * float(off_y_px) / float(h_k)
            dx = 2.0 * float(off_x_px) / float(w_k)

            sample_grid = base_grid.unsqueeze(0).repeat(bsz, 1, 1, 1)
            sample_grid[..., 0] = sample_grid[..., 0] + dx
            sample_grid[..., 1] = sample_grid[..., 1] + dy

            # Keep only valid sample points (inside pasted region) for tile-mean aggregation.
            x_valid_min = ((float(off_x_px) + 0.5) / float(w_k)) * 2.0 - 1.0
            y_valid_min = ((float(off_y_px) + 0.5) / float(h_k)) * 2.0 - 1.0
            grid_x = sample_grid[0, :, :, 0]
            grid_y = sample_grid[0, :, :, 1]
            valid_mask = (
                (grid_x >= x_valid_min)
                & (grid_x <= 1.0)
                & (grid_y >= y_valid_min)
                & (grid_y <= 1.0)
            )
            valid_tiles = valid_mask.reshape(
                k,
                n_img_tokens,
                k,
                n_img_tokens,
            ).permute(0, 2, 1, 3)
            valid_tiles = valid_tiles.reshape(
                k * k,
                n_img_tokens,
                n_img_tokens,
            )
            weights = valid_tiles.to(first_layer.dtype).unsqueeze(0).unsqueeze(-1)
            weight_denom = weights.sum(dim=1).clamp_min(1.0)

            # Accumulate numerators (weighted sums) for this sample
            sample_num_layers: List[torch.Tensor] = []
            for layer_hwc in sampled_layers:
                if layer_hwc.ndim != 4:
                    raise ValueError(
                        "Expected sampled hidden state with shape (B, H, W, C). "
                        f"Got shape {tuple(layer_hwc.shape)}."
                    )
                if (
                    int(layer_hwc.shape[0]) != bsz
                    or int(layer_hwc.shape[1]) != h_tokens
                    or int(layer_hwc.shape[2]) != w_tokens
                ):
                    raise ValueError(
                        "All selected layers must share the same (B, H, W) shape in "
                        "_compute_sampled_gt_features_deterministic_fixed_canvas. "
                        f"Expected ({bsz}, {h_tokens}, {w_tokens}, C), "
                        f"got {tuple(layer_hwc.shape)}."
                    )

                layer_nchw = layer_hwc.permute(0, 3, 1, 2)

                sampled_nchw = F.grid_sample(
                    layer_nchw,
                    sample_grid,
                    mode="nearest" if use_adaptive_canvas_size else "bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                sampled_bhwc = sampled_nchw.permute(0, 2, 3, 1)

                bsz = sampled_bhwc.shape[0]
                ch = sampled_bhwc.shape[-1]
                sampled_tiles = sampled_bhwc.reshape(
                    bsz,
                    k,
                    n_img_tokens,
                    k,
                    n_img_tokens,
                    ch,
                ).permute(0, 1, 3, 2, 4, 5)
                sampled_tiles = sampled_tiles.reshape(
                    bsz,
                    k * k,
                    n_img_tokens,
                    n_img_tokens,
                    ch,
                )
                # Weighted sum (numerator) for this sample
                weighted_sum = (sampled_tiles * weights).sum(dim=1)
                sample_num_layers.append(weighted_sum)

            # prepare batch-matched denominator for this sample
            weight_denom_b = weight_denom.repeat(bsz, 1, 1, 1)

            if accum_num_layers is None:
                accum_num_layers = [layer.clone() for layer in sample_num_layers]
                accum_den = [weight_denom_b.clone() for _ in sample_num_layers]
            else:
                assert accum_den is not None
                for li, layer_num in enumerate(sample_num_layers):
                    accum_num_layers[li].add_(layer_num)
                    accum_den[li].add_(weight_denom_b)

        # after n_samples_per_iter samples, add accumulators into running totals
        if accum_num_layers is None:
            continue
        assert accum_den is not None
        if running_num_layers is None:
            running_num_layers = [layer.clone() for layer in accum_num_layers]
            running_den = [den.clone() for den in accum_den]
        else:
            for li, layer_num in enumerate(accum_num_layers):
                running_num_layers[li].add_(layer_num)
                if running_den is None:
                    assert accum_den is not None
                    running_den = [den.clone() for den in accum_den]
                    break
                # accum_den is not None here due to previous assert
                running_den[li].add_(accum_den[li])

    if running_num_layers is None or running_den is None:
        return []

    final_layers = [
        num / den.clamp_min(1.0) for num, den in zip(running_num_layers, running_den)
    ]
    return final_layers
