"""Compare k-NN sensitivity sweeps of several models: J&F per setting and the resulting ranking.

    python -m vit_up.eval_kits.video_seg_toolkit.sweep_table \
        vit_up_d2s=<run dir>/metrics.json vit_up_d3=<run dir>/metrics.json lila=<run dir>/metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


def main(arguments: list[str]) -> None:
    runs = {}
    for argument in arguments:
        label, _, path = argument.partition("=")
        runs[label] = json.loads(Path(path).read_text())
    mode = "dense"
    settings = list(next(iter(runs.values()))["results"])
    labels = list(runs)
    width = max(len(s) for s in settings)
    print(f"{'setting':{width}s}  " + "  ".join(f"{label:>12s}" for label in labels) + "  ranking (J&F, " + mode + ")")
    for setting in settings:
        scores = {label: runs[label]["results"].get(setting, {}).get(mode, {}).get("JF") for label in labels}
        cells = "  ".join(f"{s * 100:12.1f}" if s is not None else f"{'–':>12s}" for s in scores.values())
        ranked = " > ".join(label for label, s in sorted(scores.items(), key=lambda kv: -(kv[1] or -1)) if s is not None)
        print(f"{setting:{width}s}  {cells}  {ranked}")


if __name__ == "__main__":
    main(sys.argv[1:])
