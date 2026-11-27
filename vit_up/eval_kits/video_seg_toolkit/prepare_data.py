"""Download DAVIS-2017 trainval 480p (~0.8 GB) to <data-root>/DAVIS2017.

    python -m vit_up.eval_kits.video_seg_toolkit.prepare_data davis --data-root data
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import zipfile

DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
def download(url: str, dest: Path) -> Path:
    """Download with curl (resumable); skips files that are already complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = dest.with_name(dest.name + ".done")
    if dest.exists() and done.exists():
        print(f"[skip] {dest}")
        return dest
    print(f"[download] {url} -> {dest}")
    subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(dest), url], check=True)
    done.touch()
    return dest


def extract_zip(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():  # refuse paths that escape the target directory
            if not (target / member).resolve().is_relative_to(target.resolve()):
                raise ValueError(f"Unsafe path in {archive}: {member}")
        zf.extractall(target)


def prepare_davis(data_root: Path) -> None:
    target = data_root / "DAVIS2017"
    if (target / "ImageSets" / "2017" / "val.txt").exists():
        print(f"[skip] {target} exists")
        return
    archive = download(DAVIS_URL, data_root / "downloads" / "DAVIS-2017-trainval-480p.zip")
    staging = data_root / "downloads" / "davis_extract"
    extract_zip(archive, staging)
    shutil.move(str(staging / "DAVIS"), str(target))
    shutil.rmtree(staging)
    print(f"DAVIS-2017 ready at {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", choices=["davis"])
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    prepare_davis(args.data_root)


if __name__ == "__main__":
    main()
