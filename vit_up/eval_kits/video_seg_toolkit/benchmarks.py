# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from google-research/lila (eval_base.py, eval_vos.py, eval_vosknn.py):
# features come from a FeatureExtractor instead of a LILA model, the backbone and
# dense features can be evaluated separately, and the k-NN branches are averaged
# (see vos_knn). Optimizers, schedules, losses, resolutions and label propagation
# are otherwise unchanged.
"""DAVIS-2017 video object segmentation benchmarks of the LILA protocol: linear probing, k-NN label
propagation and a sensitivity sweep of the k-NN settings."""
from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .datasets import DavisVideos
from .features import FeatureExtractor
from .metrics import DavisMetric
from .probes import LinearCat


@dataclass
class Context:
    features: FeatureExtractor
    cfg: object  # benchmark config (DictConfig)
    output_dir: Path
    device: torch.device
    name: str = "video_seg"

    def encode(self, images: torch.Tensor, dense_size: tuple[int, int] | None = None):
        """LILA's EvalBase.encode: optional square resize, then encoder and decoder features."""
        images = images.to(self.device, non_blocking=True)
        if self.cfg.get("resize_input", False):
            size = self.cfg.resize_input_size
            size = (size, size) if isinstance(size, int) else tuple(size)
            images = F.interpolate(images, size, mode="bilinear", align_corners=False)
        return self.features(images, dense_size=dense_size)

    def save(self, name: str, payload: dict) -> None:
        path = self.output_dir / name
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {path}")


def _patch_resample(frames: torch.Tensor, resize_min: int, patch_size: int) -> torch.Tensor:
    """Resize so the shorter side is `resize_min` and the longer side is patch-aligned (LILA's PatchResample)."""
    h, w = frames.shape[-2:]
    if w < h:
        size = (int(round(int(h * resize_min / w) / patch_size) * patch_size), resize_min)
    else:
        size = (resize_min, int(round(int(w * resize_min / h) / patch_size) * patch_size))
    return F.interpolate(frames, size, mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------- DAVIS-2017

# Feature branches evaluated on DAVIS. LILA's protocol is `combined`; the single-branch modes
# separate the frozen backbone tokens from the model's dense features.
MODE_BRANCHES = {"backbone": ("enc",), "dense": ("dec",), "combined": ("enc", "dec")}


def _vos_modes(ctx: Context) -> list[str]:
    modes = list(ctx.cfg.get("modes") or MODE_BRANCHES)
    unknown = set(modes) - set(MODE_BRANCHES)
    if unknown:
        raise ValueError(f"Unknown DAVIS modes {sorted(unknown)}; choose from {list(MODE_BRANCHES)}.")
    if ctx.features.decoder is None:  # backbone baseline: only the encoder branch exists
        modes = [m for m in modes if m == "backbone"]
    if not modes:
        raise ValueError("No DAVIS mode can run for this model.")
    return modes


def _branch_weights(cfg, mode: str) -> dict:
    """Branch weights of a mode; LILA's alpha weights apply to the combined mode only."""
    if mode == "combined":
        return {"enc": cfg.alpha_enc, "dec": cfg.alpha_dec}
    return {kind: 1.0 for kind in MODE_BRANCHES[mode]}


def _report_davis(ctx: Context, name: str, metrics: dict) -> dict:
    result = {mode: metric.compute() for mode, metric in metrics.items()}
    print(f"{name}: running J&F " + ", ".join(f"{mode} {r['JF']:.4f}" for mode, r in result.items()))
    return result


def _seed_statistics(per_seed: list[dict]) -> dict:
    """Mean and sample standard deviation over seeds of every metric, plus the per-seed values."""
    keys = per_seed[0].keys()
    values = {k: np.array([m[k] for m in per_seed]) for k in keys}
    return {"mean": {k: float(v.mean()) for k, v in values.items()},
            "std": {k: float(v.std(ddof=1)) if len(v) > 1 else 0.0 for k, v in values.items()},
            "seeds": per_seed}


def vos_probe(ctx: Context) -> dict:
    """Fit linear probes on the first frame of each sequence and segment the remaining frames.

    Every mode trains `num_seeds` probes with different random initializations at once: one conv
    with num_seeds x num_classes outputs and the sum of the per-seed losses, so each seed's weights
    receive exactly the gradients (and AdamW updates) of a separate run. Reports mean and std.
    """
    cfg = ctx.cfg
    davis = DavisVideos(cfg.davis_root, cfg.get("split_file"))
    modes = _vos_modes(ctx)
    num_seeds = cfg.get("num_seeds", 1)
    metrics = {mode: [DavisMetric() for _ in range(num_seeds)] for mode in modes}
    patch_size = ctx.features.encoder.patch_size
    frames_done, result = 0, {}
    for name in tqdm(davis.videos[:cfg.get("max_sequences")], desc="DAVIS sequences"):
        frames, masks = davis.get_sequence(name)
        num_classes = int(masks.max()) + 1
        enc0, dec0 = ctx.encode(_patch_resample(frames[:1], cfg.resize_input_size, patch_size))
        target = masks[:1].to(ctx.device).long().expand(num_seeds, -1, -1)
        probes = {}
        for mode in modes:
            weights = _branch_weights(cfg, mode)
            probe = LinearCat(enc0.shape[1], dec0.shape[1] if "dec" in weights else None, num_seeds * num_classes,
                              alpha_enc=weights.get("enc", 0.0), alpha_dec=weights.get("dec", 0.0),
                              kernel_size=1, padding=0).to(ctx.device)
            optimizer = torch.optim.AdamW(probe.trainable_parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            for _ in tqdm(range(cfg.num_steps), desc=f"Fitting {name} ({mode}, {num_seeds} seeds)", leave=False):
                optimizer.zero_grad()
                logits = probe(enc0, dec0, target.shape[-2:]).view(num_seeds, num_classes, *target.shape[-2:])
                # Sum over seeds of each seed's mean cross-entropy: independent probes in one conv.
                loss = F.cross_entropy(logits, target, reduction="none").mean(dim=(1, 2)).sum()
                loss.backward()
                optimizer.step()
            probes[mode] = probe
        predictions = {mode: [] for mode in modes}
        with torch.no_grad():
            for frame in frames.split(1):
                enc, dec = ctx.encode(_patch_resample(frame, cfg.resize_input_size, patch_size))
                for mode, probe in probes.items():
                    logits = probe(enc, dec, masks.shape[-2:]).view(num_seeds, num_classes, *masks.shape[-2:])
                    predictions[mode].append(logits.argmax(1).cpu())  # (num_seeds, H, W)
                frames_done += 1
        for mode in modes:
            stacked = torch.stack(predictions[mode], dim=1).numpy()  # (num_seeds, T, H, W)
            for seed, metric in enumerate(metrics[mode]):
                metric.add_sequence(masks.numpy(), stacked[seed])
        result = {mode: _seed_statistics([m.compute() for m in seed_metrics]) for mode, seed_metrics in metrics.items()}
        print(f"{name}: running J&F " + ", ".join(
            f"{mode} {r['mean']['JF']:.4f} +- {r['std']['JF']:.4f}" for mode, r in result.items()))
    ctx.save("metrics.json", result)
    return result


def _pad_and_unfold(x: torch.Tensor, window: int) -> torch.Tensor:
    d = x.shape[-3]
    padded = F.pad(x, (window // 2,) * 4, mode="replicate")
    return F.unfold(padded, kernel_size=window).unflatten(1, (d, window ** 2))


def _propagate_labels(feats_current, feats_context, masks_context, window: int, num_nn: int):
    """Soft masks of the current frame from top-k matches inside a local window of every context frame."""
    h, w = feats_current.shape[-2:]
    feats_context = torch.stack(feats_context)
    masks_context = torch.stack(masks_context)
    q = feats_current.movedim(1, -1)
    k = feats_context.movedim(1, -1)
    padding = window // 2
    with torch.autocast("cuda", enabled=q.is_cuda):
        values = _pad_and_unfold(masks_context, window).permute(3, 0, 2, 1).flatten(1, 2)
        k_padded = F.pad(k, (0, 0, padding, padding, padding, padding), "replicate")
        sims = [(q * k_padded[:, u:u + h, v:v + w, :]).sum(dim=3)
                for u in range(window) for v in range(window)]
        sim = torch.stack(sims, dim=0).flatten(-2).permute(2, 1, 0).flatten(1, 2)
        topk, topk_idx = sim.topk(num_nn, -1)
        topk_values = values.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, values.shape[-1]))
        out = torch.einsum("nk,nkc->cn", F.softmax(topk, 1), topk_values)
    return out.unflatten(-1, (h, w)).float()


def _grid_size(height: int, width: int, grid_size: int) -> tuple[int, int]:
    """LILA's k-NN grid: the shorter side becomes `grid_size`, the longer side keeps the aspect ratio."""
    return (grid_size, int(width * grid_size / height)) if height < width else (int(height * grid_size / width), grid_size)


def _rescale_grid(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size, mode="bilinear", align_corners=False)


def vos_knn(ctx: Context) -> dict:
    """Propagate the first-frame masks by local k-NN over the first frame and the last predictions.

    Every mode keeps its own predicted context masks; the features are computed once per frame.
    """
    cfg = ctx.cfg
    davis = DavisVideos(cfg.davis_root, cfg.get("split_file"))
    modes = _vos_modes(ctx)
    metrics = {mode: DavisMetric() for mode in modes}
    kinds = sorted({kind for mode in modes for kind in MODE_BRANCHES[mode]})
    l2norm = lambda x: F.normalize(x, dim=1, p=2) / cfg.temp ** 0.5
    rescale = lambda x, ref: F.interpolate(x, ref.shape[-2:], mode="bilinear", align_corners=False)
    frames_done, result = 0, {}

    for name in tqdm(davis.videos[:cfg.get("max_sequences")], desc="DAVIS sequences"):
        frames, masks = davis.get_sequence(name)
        # LILA resamples with patch size 14 here regardless of the backbone.
        frames = _patch_resample(frames, cfg.resample_size, cfg.resample_patch_size)
        ref = F.one_hot(masks[:1].to(ctx.device).long(), int(masks[0].max()) + 1).movedim(-1, 1).float()

        with torch.no_grad():
            # One grid for every branch, derived from the frame: feature maps of different sizes
            # (e.g. 16 px backbone tokens of a 14 px-aligned frame) must share the mask grid.
            grid = _grid_size(*frames.shape[-2:], cfg.feat_grid_size)
            # Dense features are requested at the grid: implicit upsamplers are queried there,
            # fixed-resolution models (LILA) are resized to it as in LILA's evaluation.
            first = dict(zip(("enc", "dec"), ctx.encode(frames[:1], grid)))
            first = {k: l2norm(_rescale_grid(first[k], grid)) for k in kinds}
            ref_small = rescale(ref, first["enc"] if "enc" in first else first["dec"])
            context_feats = {k: [] for k in kinds}
            context_masks = {mode: [] for mode in modes}
            predictions = {mode: [] for mode in modes}
            for frame in tqdm(frames[1:], desc=name, leave=False):
                current = dict(zip(("enc", "dec"), ctx.encode(frame[None], grid)))
                feats = {k: l2norm(_rescale_grid(current[k], grid)) for k in kinds}
                for mode in modes:
                    masks_in = [ref_small[0]] + context_masks[mode]
                    mask_out = torch.zeros_like(masks_in[0])
                    total_weight = 0.0
                    for kind, weight in _branch_weights(cfg, mode).items():
                        if weight == 0:
                            continue
                        mask_out += weight * _propagate_labels(
                            feats[kind], [first[kind][0]] + context_feats[kind], masks_in,
                            cfg.attn_window, cfg.num_nn)
                        total_weight += weight
                    if cfg.get("normalize_branch_masks", True):
                        # LILA's released code sums the branches' soft masks and feeds the sum back
                        # as context, so with two branches the masks double every frame and overflow
                        # in fp16 after ~16 frames (released LILA: 36.6 J&F vs. 73.9 in the paper).
                        # The weighted average keeps unit-sum masks; each frame's argmax is unchanged.
                        mask_out = mask_out / total_weight
                    context_masks[mode] = (context_masks[mode] + [mask_out])[-cfg.context_size:]
                    predictions[mode].append(mask_out)
                for kind in kinds:
                    context_feats[kind] = (context_feats[kind] + [feats[kind][0]])[-cfg.context_size:]
                frames_done += 1
            for mode in modes:
                predicted = rescale(torch.stack(predictions[mode]), ref).argmax(1).cpu()
                metrics[mode].add_sequence(masks.numpy(), torch.cat([masks[:1].long(), predicted]).numpy())
        result = _report_davis(ctx, name, metrics)
    ctx.save("metrics.json", result)
    return result


def _sweep_settings(cfg) -> dict[str, dict]:
    """LILA's k-NN settings plus one-at-a-time variations of each parameter."""
    base = dict(cfg.base)
    settings = {"lila": base}
    for param, values in cfg.vary.items():
        if param not in base:
            raise ValueError(f"Cannot vary unknown k-NN parameter {param!r}; known: {sorted(base)}.")
        for value in values:
            settings[f"{param}_{str(value).replace('.', 'p')}"] = base | {param: value}
    return settings


_WINDOW_INDEX: dict = {}


def _window_index(h: int, w: int, window: int, device) -> torch.Tensor:
    """(h*w, window^2) flat key index per query pixel; replicate padding = clamped coordinates."""
    key = (h, w, window, str(device))
    if key not in _WINDOW_INDEX:
        r = torch.arange(window, device=device) - window // 2
        i = torch.arange(h, device=device)[:, None, None, None] + r[None, None, :, None]
        j = torch.arange(w, device=device)[None, :, None, None] + r[None, None, None, :]
        _WINDOW_INDEX[key] = (i.clamp(0, h - 1) * w + j.clamp(0, w - 1)).reshape(h * w, window * window)
    return _WINDOW_INDEX[key]


def _propagate_labels_fast(feats_current, feats_context, masks_context, window: int, num_nn: int):
    """_propagate_labels via one similarity matmul per context frame (about 20x faster).

    Same windows, replicate padding, candidate order and top-k; equal up to fp16 rounding of the
    final weighted average (same argmax on test data).
    """
    h, w = feats_current.shape[-2:]
    idx = _window_index(h, w, window, feats_current.device)
    q = feats_current[0].flatten(1)
    sims, vals = [], []
    for k, m in zip(feats_context, masks_context):
        sims.append((q.T @ k.flatten(1)).gather(1, idx))
        vals.append(m.flatten(1)[:, idx])
    sim = torch.stack(sims, 1).flatten(1)
    values = torch.stack(vals, 2).flatten(2, 3).permute(1, 2, 0)
    with torch.autocast("cuda", enabled=q.is_cuda):
        topk, topk_idx = sim.topk(num_nn, -1)
        top_values = values.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, values.shape[-1]))
        out = torch.einsum("nk,nkc->cn", F.softmax(topk, 1), top_values)
    return out.unflatten(-1, (h, w)).float()


def _knn_sequence(feats: dict, ref_small: torch.Tensor, weights: dict, setting: dict) -> list[torch.Tensor]:
    """Label propagation of one sequence for one setting; feats[kind] is (T, C, h, w)."""
    def prepare(x):
        if setting["center"]:
            x = x - x.mean(dim=(-2, -1), keepdim=True)  # remove each frame's common feature component
        return F.normalize(x, dim=1) / setting["temp"] ** 0.5

    prepared = {kind: prepare(frames) for kind, frames in feats.items()}
    context_feats = {kind: [] for kind in weights}
    context_masks, predictions = [], []
    for t in range(1, len(next(iter(prepared.values())))):
        masks_in = [ref_small[0]] + context_masks
        mask_out = torch.zeros_like(masks_in[0])
        for kind, weight in weights.items():
            mask_out += weight * _propagate_labels_fast(
                prepared[kind][t:t + 1], [prepared[kind][0]] + context_feats[kind], masks_in,
                setting["attn_window"], setting["num_nn"])
            context_feats[kind] = (context_feats[kind] + [prepared[kind][t]])[-setting["context_size"]:]
        mask_out = mask_out / sum(weights.values())
        context_masks = (context_masks + [mask_out])[-setting["context_size"]:]
        predictions.append(mask_out)
    return predictions


def vos_knn_sweep(ctx: Context) -> dict:
    """Sensitivity of DAVIS k-NN to its hand-set parameters, identical for every model.

    1) Dense features of a sequence subset are extracted once per grid size (cached in CPU memory).
    2) Every setting propagates labels on the cached features.
    """
    cfg = ctx.cfg
    davis = DavisVideos(cfg.davis_root, cfg.get("split_file"))
    names = davis.videos[cfg.get("sequence_offset", 0)::cfg.get("sequence_stride", 1)][:cfg.get("max_sequences")]
    modes = _vos_modes(ctx)
    settings = _sweep_settings(cfg)
    kinds = sorted({kind for mode in modes for kind in MODE_BRANCHES[mode]})
    grid_sizes = sorted({setting["feat_grid_size"] for setting in settings.values()})
    rescale = lambda x, size: F.interpolate(x, size, mode="bilinear", align_corners=False)

    cache, frames_done = {}, 0
    with torch.no_grad():
        for name in tqdm(names, desc="Extracting features", unit="seq"):
            frames, masks = davis.get_sequence(name)
            frames = _patch_resample(frames, cfg.resample_size, cfg.resample_patch_size)
            grids = {g: _grid_size(*frames.shape[-2:], g) for g in grid_sizes}
            feats = {g: {kind: [] for kind in kinds} for g in grid_sizes}
            for frame in tqdm(frames, desc=name, leave=False, unit="frame"):
                for g, size in grids.items():
                    current = dict(zip(("enc", "dec"), ctx.encode(frame[None], size)))
                    for kind in kinds:
                        feats[g][kind].append(_rescale_grid(current[kind], size)[0].half().cpu())
                frames_done += 1
            cache[name] = {"masks": masks, "grids": grids,
                           "feats": {g: {k: torch.stack(v) for k, v in by_kind.items()} for g, by_kind in feats.items()}}
    num_frames = sum(len(c["masks"]) for c in cache.values())
    print(f"Cached features of {len(cache)} sequences ({num_frames} frames) at grid sizes {grid_sizes}")

    result, timings = {}, {}
    progress = tqdm(settings.items(), desc="Settings", unit="setting")
    with torch.no_grad():
        for setting_name, setting in progress:
            start = time.perf_counter()
            metrics = {mode: DavisMetric() for mode in modes}
            for name in tqdm(names, desc=setting_name, leave=False, unit="seq"):
                entry = cache[name]
                masks, size = entry["masks"], entry["grids"][setting["feat_grid_size"]]
                ref = F.one_hot(masks[:1].to(ctx.device).long(), int(masks[0].max()) + 1).movedim(-1, 1).float()
                ref_small = rescale(ref, size)
                for mode in modes:
                    weights = {k: w for k, w in _branch_weights(cfg, mode).items() if w != 0}
                    feats = {k: entry["feats"][setting["feat_grid_size"]][k].to(ctx.device).float() for k in weights}
                    predictions = _knn_sequence(feats, ref_small, weights, setting)
                    predicted = rescale(torch.stack(predictions), masks.shape[-2:]).argmax(1).cpu()
                    metrics[mode].add_sequence(masks.numpy(), torch.cat([masks[:1].long(), predicted]).numpy())
            result[setting_name] = {mode: metric.compute() for mode, metric in metrics.items()}
            timings[setting_name] = time.perf_counter() - start
            summary = ", ".join(f"{mode} {r['JF'] * 100:.1f}" for mode, r in result[setting_name].items())
            progress.set_postfix_str(f"{setting_name}: {summary}")
            tqdm.write(f"{setting_name:22s} J&F {summary}  ({timings[setting_name]:.0f} s)")

    ctx.save("metrics.json", {"settings": {k: dict(v) for k, v in settings.items()}, "sequences": list(names),
                              "frames": num_frames, "results": result, "seconds": timings})
    return result


BENCHMARKS = {
    "vos_probe": vos_probe,
    "vos_knn": vos_knn,
    "vos_knn_sweep": vos_knn_sweep,
}
