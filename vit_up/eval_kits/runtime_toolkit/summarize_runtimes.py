from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

from omegaconf import OmegaConf


def _config_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "config"
        / "runtime"
        / "runtime_bench.yaml"
    )


def _load_runtime_defaults() -> Tuple[Optional[str], Optional[str]]:
    config_path = _config_path()
    if not config_path.is_file():
        return None, None
    cfg = OmegaConf.load(config_path)
    return cfg.get("mnt_dir"), cfg.get("results_filename")


def _parse_models(raw: str) -> List[str]:
    models = [item.strip() for item in raw.split(",") if item.strip()]
    if not models:
        raise ValueError("No models were provided. Use --models or model=m1,m2.")
    return models


def _find_latest_results_file(
    model_root: Path, results_filename: str
) -> Optional[Path]:
    candidates = list(model_root.rglob(results_filename))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _summarize_results(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = payload.get("results", {})
    rows: List[Dict[str, Any]] = []
    for size_key, entry in results.items():
        total_ms_values = entry.get("total_ms", [])
        peak_allocated_values = entry.get("peak_allocated_mib")
        row = {
            "output_size": int(size_key),
            "total_ms_mean": mean(total_ms_values) if total_ms_values else None,
            "peak_allocated_mib_mean": (
                mean(peak_allocated_values) if peak_allocated_values else None
            ),
        }
        rows.append(row)
    rows.sort(key=lambda item: item["output_size"])
    return rows


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _extract_model_override(argv: Iterable[str]) -> Optional[str]:
    for arg in argv:
        if arg.startswith("model="):
            return arg.split("=", 1)[1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize runtime benchmarks from the latest Hydra run per model."
        )
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model names (e.g., dinov3/jafar,dinov3/other)",
    )
    parser.add_argument(
        "--mnt-dir",
        help="Override runtime mnt_dir (defaults to runtime_bench.yaml).",
    )
    parser.add_argument(
        "--results-filename",
        default=None,
        help="Override results filename (defaults to runtime_bench.yaml).",
    )
    args, unknown = parser.parse_known_args()

    model_override = _extract_model_override(unknown)
    models_raw = model_override or args.models
    if not models_raw:
        raise SystemExit("Provide models via --models or model=m1,m2.")
    models = _parse_models(models_raw)

    mnt_dir, results_filename = _load_runtime_defaults()
    mnt_dir = args.mnt_dir or mnt_dir
    results_filename = args.results_filename or results_filename or "runtime_bench.json"
    if not mnt_dir:
        raise SystemExit("mnt_dir is unknown. Pass --mnt-dir to continue.")

    runtime_root = Path(mnt_dir) / "output" / "runtime"
    rows: List[Tuple[str, int, Optional[float], Optional[float]]] = []
    latest_paths: Dict[str, Optional[Path]] = {}

    for model in models:
        model_root = runtime_root / model
        results_path = _find_latest_results_file(model_root, results_filename)
        latest_paths[model] = results_path
        if results_path is None:
            continue
        payload = _load_json(results_path)
        for row in _summarize_results(payload):
            rows.append(
                (
                    model,
                    row["output_size"],
                    row["total_ms_mean"],
                    row["peak_allocated_mib_mean"],
                )
            )

    for model, path in latest_paths.items():
        if path is None:
            print(f"{model}: no results found under {runtime_root}/{model}")
        else:
            print(f"{model}: {path}")

    if not rows:
        print("No runtime data found to summarize.")
        return

    print("\n| Model | Output | Total ms (mean) | Peak alloc MiB (mean) |")
    print("| --- | --- | --- | --- |")
    for model, output_size, total_ms_mean, peak_alloc_mean in sorted(
        rows, key=lambda item: (item[0], item[1])
    ):
        print(
            "| {model} | {output} | {total_ms} | {peak} |".format(
                model=model,
                output=output_size,
                total_ms=_format_value(total_ms_mean),
                peak=_format_value(peak_alloc_mean),
            )
        )


if __name__ == "__main__":
    main()
