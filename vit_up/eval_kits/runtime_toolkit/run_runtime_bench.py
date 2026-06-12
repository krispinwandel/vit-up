from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from vit_up.eval_kits.upsamplers.base import UpsamplerBase
from vit_up.eval_kits.runtime_toolkit.model_analytics import print_model_and_params


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if amp_dtype is None or device.type not in ("cuda", "xpu"):
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _format_stats(values: List[float]) -> str:
    return (
        f"mean={mean(values):.2f}, std={pstdev(values):.2f}, "
        f"min={min(values):.2f}, max={max(values):.2f}"
    )


def _bytes_to_mib(value: int) -> float:
    return value / float(1024**2)


def _reset_cuda_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        # Reset only this process's PyTorch allocator peak counters. This keeps the
        # measurement independent from other jobs that may be using the same GPU.
        torch.cuda.reset_peak_memory_stats(device)


def _get_cuda_peak_memory_mib(
    device: torch.device,
    baseline_allocated_bytes: int,
    baseline_reserved_bytes: int,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if device.type != "cuda":
        return None, None, None, None

    # `allocated` is live tensor memory; `reserved` is memory held by PyTorch's
    # caching allocator and is the closer estimate of this process's VRAM footprint.
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    # The delta fields isolate memory added by this iteration from persistent
    # model/input allocations that were already present before timing started.
    peak_allocated_delta_bytes = max(
        0,
        peak_allocated_bytes - baseline_allocated_bytes,
    )
    peak_reserved_delta_bytes = max(0, peak_reserved_bytes - baseline_reserved_bytes)
    return (
        _bytes_to_mib(peak_allocated_bytes),
        _bytes_to_mib(peak_allocated_delta_bytes),
        _bytes_to_mib(peak_reserved_bytes),
        _bytes_to_mib(peak_reserved_delta_bytes),
    )


def _call_model_forward(
    model: UpsamplerBase,
    pixel_values_bchw: torch.Tensor,
    output_size: int,
    input_size: Optional[int],
    cache_data: Optional[Any],
) -> torch.Tensor:
    return model(
        pixel_values_bchw=pixel_values_bchw,
        output_size=output_size,
        input_size=input_size,
        cache_data=cache_data,
    )


def _run_one_iter(
    model: UpsamplerBase,
    image: torch.Tensor,
    output_size: int,
    input_size: Optional[int],
    amp_dtype: Optional[torch.dtype],
) -> Tuple[
    float,
    float,
    float,
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    with torch.inference_mode(), _autocast_context(image.device, amp_dtype):
        _sync_if_needed(image.device)
        # Capture persistent memory before the iteration, then reset peak counters
        # so the subsequent peak reflects only the cache + forward work below.
        baseline_allocated_bytes = (
            torch.cuda.memory_allocated(image.device)
            if image.device.type == "cuda"
            else 0
        )
        baseline_reserved_bytes = (
            torch.cuda.memory_reserved(image.device)
            if image.device.type == "cuda"
            else 0
        )
        _reset_cuda_peak_memory_stats(image.device)

        t0 = time.perf_counter()
        cache_data = model.pre_compute_cache(
            pixel_values_bchw=image,
            output_size=output_size,
            input_size=input_size,
        )
        _sync_if_needed(image.device)
        t1 = time.perf_counter()

        _call_model_forward(
            model=model,
            pixel_values_bchw=image,
            output_size=output_size,
            input_size=input_size,
            cache_data=cache_data,
        )
        _sync_if_needed(image.device)
        t2 = time.perf_counter()

        (
            peak_allocated_mib,
            peak_allocated_delta_mib,
            peak_reserved_mib,
            peak_reserved_delta_mib,
        ) = _get_cuda_peak_memory_mib(
            image.device,
            baseline_allocated_bytes=baseline_allocated_bytes,
            baseline_reserved_bytes=baseline_reserved_bytes,
        )

    cache_ms = (t1 - t0) * 1000.0
    forward_ms = (t2 - t1) * 1000.0
    total_ms = (t2 - t0) * 1000.0
    return (
        cache_ms,
        forward_ms,
        total_ms,
        peak_allocated_mib,
        peak_allocated_delta_mib,
        peak_reserved_mib,
        peak_reserved_delta_mib,
    )


def _normalize_output_sizes(values: Iterable[Any]) -> List[int]:
    sizes: List[int] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError("output_sizes entries must be ints or [H, W] pairs.")
            if int(value[0]) != int(value[1]):
                raise ValueError(
                    f"Non-square output size provided: {value}. "
                    "ViT-Up runtime benchmark expects square outputs."
                )
            sizes.append(int(value[0]))
        else:
            sizes.append(int(value))
    return sizes


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


@hydra.main(
    config_path="../config/runtime",
    config_name="runtime_bench",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if cfg.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative.")
    if cfg.bench_iters <= 0:
        raise ValueError("bench_iters must be positive.")

    output_sizes = _normalize_output_sizes(cfg.output_sizes)
    if not output_sizes:
        raise ValueError("output_sizes must contain at least one size.")

    image_size = int(cfg.image_size)

    device = torch.device(cfg.device)
    tensor_dtype = _dtype_from_name(str(cfg.dtype))
    amp_dtype = None if tensor_dtype == torch.float32 else tensor_dtype
    print("Loading model...")
    model = instantiate(cfg.model)
    if not isinstance(model, UpsamplerBase):
        raise TypeError("cfg.model did not instantiate to UpsamplerBase.")
    model = model.to(device=device, dtype=tensor_dtype).eval()

    print_model_and_params(model)
    if bool(cfg.print_model_params_only):
        return

    image = torch.randn(
        int(cfg.batch_size),
        3,
        int(image_size),
        int(image_size),
        device=device,
        dtype=tensor_dtype,
    )

    run_dir = Path(HydraConfig.get().run.dir)
    results: Dict[str, Any] = {
        "model": str(OmegaConf.select(cfg, "model.name", default="model")),
        "device": str(device),
        "dtype": str(cfg.dtype),
        "batch_size": int(cfg.batch_size),
        "image_size": int(image_size),
        "warmup_iters": int(cfg.warmup_iters),
        "bench_iters": int(cfg.bench_iters),
        "output_sizes": output_sizes,
        "memory_unit": "MiB",
        "memory_measurement": (
            "torch.cuda max_memory_allocated/max_memory_reserved for this process; "
            "background processes are not included. Reserved memory is the closer "
            "PyTorch allocator estimate of VRAM held by this process."
        ),
        "results": {},
    }

    print("Benchmark configuration:")
    print(f"  model={results['model']}")
    print(f"  device={device}")
    print(f"  dtype={cfg.dtype}")
    print(f"  batch_size={cfg.batch_size}")
    print(f"  image_shape={tuple(image.shape)}")
    print(f"  warmup_iters={cfg.warmup_iters}")
    print(f"  bench_iters={cfg.bench_iters}")

    for output_size in output_sizes:
        print(f"\nOutput size {output_size}x{output_size}")

        for _ in range(int(cfg.warmup_iters)):
            _run_one_iter(
                model=model,
                image=image,
                output_size=int(output_size),
                input_size=cfg.image_size,
                amp_dtype=amp_dtype,
            )

        cache_times: List[float] = []
        forward_times: List[float] = []
        total_times: List[float] = []
        peak_allocated_mib_values: List[float] = []
        peak_allocated_delta_mib_values: List[float] = []
        peak_reserved_mib_values: List[float] = []
        peak_reserved_delta_mib_values: List[float] = []
        for _ in tqdm(range(int(cfg.bench_iters))):
            (
                cache_ms,
                forward_ms,
                total_ms,
                peak_allocated_mib,
                peak_allocated_delta_mib,
                peak_reserved_mib,
                peak_reserved_delta_mib,
            ) = _run_one_iter(
                model=model,
                image=image,
                output_size=int(output_size),
                input_size=cfg.image_size,
                amp_dtype=amp_dtype,
            )
            cache_times.append(cache_ms)
            forward_times.append(forward_ms)
            total_times.append(total_ms)
            if peak_allocated_mib is not None:
                peak_allocated_mib_values.append(peak_allocated_mib)
            if peak_allocated_delta_mib is not None:
                peak_allocated_delta_mib_values.append(peak_allocated_delta_mib)
            if peak_reserved_mib is not None:
                peak_reserved_mib_values.append(peak_reserved_mib)
            if peak_reserved_delta_mib is not None:
                peak_reserved_delta_mib_values.append(peak_reserved_delta_mib)

        summary = {
            "cache_ms": _format_stats(cache_times),
            "forward_ms": _format_stats(forward_times),
            "total_ms": _format_stats(total_times),
        }
        result_entry: Dict[str, Any] = {
            "cache_ms": cache_times,
            "forward_ms": forward_times,
            "total_ms": total_times,
            "summary": summary,
        }
        if peak_allocated_mib_values:
            # Omit CUDA memory fields on non-CUDA devices instead of writing null
            # arrays, keeping CPU/XPU result JSON compact and backwards-friendly.
            result_entry["peak_allocated_mib"] = peak_allocated_mib_values
            result_entry["peak_allocated_delta_mib"] = peak_allocated_delta_mib_values
            result_entry["peak_reserved_mib"] = peak_reserved_mib_values
            result_entry["peak_reserved_delta_mib"] = peak_reserved_delta_mib_values
            summary["peak_allocated_mib"] = _format_stats(peak_allocated_mib_values)
            summary["peak_allocated_delta_mib"] = _format_stats(
                peak_allocated_delta_mib_values
            )
            summary["peak_reserved_mib"] = _format_stats(peak_reserved_mib_values)
            summary["peak_reserved_delta_mib"] = _format_stats(
                peak_reserved_delta_mib_values
            )
        results["results"][str(output_size)] = result_entry

        print("Results (milliseconds):")
        print(f"  cache/backbone: {_format_stats(cache_times)}")
        print(f"  forward/model : {_format_stats(forward_times)}")
        print(f"  total_forward : {_format_stats(total_times)}")
        if peak_allocated_mib_values:
            print("Peak CUDA memory (MiB, current process only):")
            print(f"  allocated       : {_format_stats(peak_allocated_mib_values)}")
            print(
                "  allocated delta : "
                f"{_format_stats(peak_allocated_delta_mib_values)}"
            )
            print(f"  reserved        : {_format_stats(peak_reserved_mib_values)}")
            print(
                f"  reserved delta  : {_format_stats(peak_reserved_delta_mib_values)}"
            )

    if cfg.save_results:
        results_path = run_dir / str(cfg.results_filename)
        _write_json(results_path, results)
        print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
