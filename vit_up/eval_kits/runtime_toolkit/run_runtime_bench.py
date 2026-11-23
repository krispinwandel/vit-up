"""Latency and peak memory from an image to dense features, including the backbone."""
import json
import time
from pathlib import Path
from statistics import mean, median, pstdev

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    """Unique parameters, split into the frozen feature extractor and the upsampler."""
    parameters = {id(p): p for p in model.parameters()}
    backbone = {id(p) for name, module in model.named_modules()
                if name.split(".")[-1] in ("backbone", "extractor") for p in module.parameters()}
    count = lambda ids: sum(parameters[i].numel() for i in ids)  # noqa: E731
    return {"total": count(parameters), "backbone": count(backbone & parameters.keys()),
            "upsampler": count(parameters.keys() - backbone)}


def measure(model, image, output_size, amp_dtype) -> tuple[float, float]:
    """One image-to-features pass: milliseconds and peak additional allocated MiB."""
    device = image.device
    autocast = torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None)
    with torch.inference_mode(), autocast:
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        output = model(image, output_size=output_size, input_size=image.shape[-1]).float()
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000.0
        peak = (torch.cuda.max_memory_allocated(device) - baseline) / 2 ** 20
    if tuple(output.shape[:3]) != (image.shape[0], output_size, output_size):
        raise ValueError(f"Expected a {output_size}x{output_size} feature grid, got {tuple(output.shape)}.")
    return elapsed, peak


def sample(model, image, output_size, amp_dtype, min_iters, min_seconds) -> list[tuple[float, float]]:
    """Measure until both `min_iters` iterations and `min_seconds` of measured time are reached."""
    samples = []
    while len(samples) < min_iters or sum(s[0] for s in samples) < 1000 * min_seconds:
        samples.append(measure(model, image, output_size, amp_dtype))
    return samples


@hydra.main(config_path="../config/runtime", config_name="runtime_bench", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.num_threads)
    amp_dtype = DTYPES[cfg.dtype]
    model = instantiate(cfg.model).to(device=device, dtype=torch.float32).eval()
    image = torch.randn(cfg.batch_size, 3, cfg.image_size, cfg.image_size, device=device)

    results = {"model": OmegaConf.to_container(cfg.model, resolve=True), "device": torch.cuda.get_device_name(device),
               "torch": torch.__version__, "dtype": cfg.dtype, "batch_size": cfg.batch_size,
               "image_size": cfg.image_size, "parameters": parameter_counts(model), "results": {}}
    print(json.dumps(results["parameters"], indent=2))
    for output_size in cfg.output_sizes:
        measure(model, image, output_size, amp_dtype)  # compilation and lazy initialization
        repeats = []
        for _ in range(cfg.bench_repeats):
            sample(model, image, output_size, amp_dtype, cfg.warmup_iters, cfg.warmup_seconds)
            repeats.append(sample(model, image, output_size, amp_dtype, cfg.bench_iters, cfg.bench_seconds))
        times = [t for repeat in repeats for t, _ in repeat]
        medians = [median(t for t, _ in repeat) for repeat in repeats]
        results["results"][str(output_size)] = {
            "ms_mean": mean(times), "ms_median": median(times), "ms_std": pstdev(times),
            "ms_min": min(times), "ms_max": max(times), "samples": len(times),
            "repeat_medians_ms": medians, "repeat_relative_spread": (max(medians) - min(medians)) / median(medians),
            "peak_allocated_mib_max": max(m for repeat in repeats for _, m in repeat),
        }
        print(f"{output_size}x{output_size}: {json.dumps(results['results'][str(output_size)])}")
    path = Path(HydraConfig.get().run.dir) / "runtime_bench.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
