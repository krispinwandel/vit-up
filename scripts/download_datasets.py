#!/usr/bin/env python3
"""Download datasets following JAFAR docs/datasets.md."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import urllib.parse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageOps
from tqdm import tqdm

COPY_CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_EXTRACT_WORKERS = 8
NAVI_DOWNSAMPLE_MIN_SIZE = 1024
NAVI_VERSION = "navi_v1"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _cgroup_cpu_count() -> Optional[int]:
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota_text, period_text = cpu_max.read_text().strip().split()[:2]
    except (FileNotFoundError, ValueError):
        quota_text = period_text = ""
    if quota_text and quota_text != "max":
        try:
            quota = int(quota_text)
            period = int(period_text)
        except ValueError:
            quota = period = 0
        if quota > 0 and period > 0:
            return max(1, quota // period)

    quota = _read_int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    period = _read_int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    if quota is not None and period is not None and quota > 0 and period > 0:
        return max(1, quota // period)
    return None


def _affinity_cpu_count() -> Optional[int]:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return None


def _available_workers(max_workers: Optional[int] = None) -> int:
    if max_workers is not None:
        return max(1, max_workers)

    cpu_count = _cgroup_cpu_count() or _affinity_cpu_count() or os.cpu_count() or 1
    return max(1, min(cpu_count - 1, DEFAULT_MAX_EXTRACT_WORKERS))


def _is_safe_member_path(extract_to: Path, member_name: str) -> bool:
    target_path = (extract_to / member_name).resolve()
    extract_root = extract_to.resolve()
    return target_path == extract_root or extract_root in target_path.parents


def _content_length(headers: Message) -> Optional[int]:
    value = headers.get("Content-Length")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _content_range_total(headers: Message) -> Optional[int]:
    value = headers.get("Content-Range")
    if value is None or "/" not in value:
        return None
    total = value.rsplit("/", 1)[1]
    if total == "*":
        return None
    try:
        return int(total)
    except ValueError:
        return None


def _remote_size(
    url: str, opener: Optional[urllib.request.OpenerDirector] = None
) -> Optional[int]:
    request = urllib.request.Request(url, method="HEAD")
    open_url = urllib.request.urlopen if opener is None else opener.open
    try:
        with open_url(request) as resp:
            return _content_length(resp.headers)
    except Exception:
        return None


def _download(
    url: str, dest: Path, opener: Optional[urllib.request.OpenerDirector] = None
) -> None:
    _ensure_dir(dest.parent)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    remote_size = _remote_size(url, opener=opener)

    if dest.exists():
        local_size = dest.stat().st_size
        if remote_size is None:
            print(f"[skip] {dest} already exists")
            return
        if local_size == remote_size:
            print(f"[skip] {dest} already downloaded")
            return
        if local_size < remote_size:
            if not tmp_dest.exists() or local_size > tmp_dest.stat().st_size:
                dest.replace(tmp_dest)
        else:
            print(f"[download] {dest} is larger than remote file; redownloading")
            dest.unlink()

    resume_size = tmp_dest.stat().st_size if tmp_dest.exists() else 0
    if remote_size is not None and resume_size == remote_size:
        tmp_dest.replace(dest)
        print(f"[skip] {dest} already downloaded")
        return
    if remote_size is not None and resume_size > remote_size:
        print(
            f"[download] partial file is larger than remote file; restarting {dest.name}"
        )
        tmp_dest.unlink()
        resume_size = 0

    print(f"[download] {url} -> {dest}")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )
    if resume_size:
        request.add_header("Range", f"bytes={resume_size}-")

    open_url = urllib.request.urlopen if opener is None else opener.open
    try:
        resp_ctx = open_url(request)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and tmp_dest.exists():
            tmp_dest.replace(dest)
            print(f"[skip] {dest} already downloaded")
            return
        if exc.code == 406:
            print(f"[download] server rejected request; retrying with curl: {dest}")
            _download_with_curl(url, tmp_dest, resume=resume_size > 0)
            if remote_size is not None and tmp_dest.stat().st_size != remote_size:
                raise RuntimeError(
                    f"Incomplete download for {dest}: got {tmp_dest.stat().st_size} "
                    f"bytes, expected {remote_size}"
                )
            tmp_dest.replace(dest)
            return
        raise

    with resp_ctx as resp:
        status = getattr(resp, "status", resp.getcode())
        did_resume = resume_size > 0 and status == 206
        if resume_size and not did_resume:
            print(f"[download] server did not resume {dest.name}; restarting")
            resume_size = 0

        total = _content_range_total(resp.headers)
        if total is None:
            content_length = _content_length(resp.headers)
            total = resume_size + content_length if content_length is not None else None

        mode = "ab" if did_resume else "wb"
        initial = resume_size if did_resume else 0
        with (
            tqdm(
                total=total,
                initial=initial,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            ) as progress,
            open(tmp_dest, mode) as f,
        ):
            while True:
                chunk = resp.read(COPY_CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                progress.update(len(chunk))

    if remote_size is not None and tmp_dest.stat().st_size != remote_size:
        raise RuntimeError(
            f"Incomplete download for {dest}: got {tmp_dest.stat().st_size} bytes, "
            f"expected {remote_size}"
        )
    tmp_dest.replace(dest)


def _extract_marker(extract_to: Path, archive_path: Path) -> Path:
    return extract_to / f".{archive_path.name}.extracted"


def _archive_extracted(extract_to: Path, archive_name: str) -> bool:
    return _extract_marker(extract_to, Path(archive_name)).exists()


def _can_use_unzip() -> bool:
    try:
        # Redirect output to DEVNULL to avoid printing version info
        subprocess.run(
            ["unzip", "-v"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _can_use_tar() -> bool:
    try:
        subprocess.run(
            ["tar", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _require_unzip() -> None:
    if not _can_use_unzip():
        raise RuntimeError("Required utility 'unzip' is not installed")


def _require_tar() -> None:
    if not _can_use_tar():
        raise RuntimeError("Required utility 'tar' is not installed")


def _can_use_curl() -> bool:
    try:
        subprocess.run(
            ["curl", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _download_with_curl(url: str, dest: Path, resume: bool) -> None:
    if not _can_use_curl():
        raise RuntimeError("curl is required to download this URL")

    cmd = ["curl", "-L", "--fail", "-o", str(dest)]
    if resume:
        cmd += ["-C", "-"]
    cmd.append(url)
    subprocess.run(cmd, check=True)


def _list_zip_members(zip_path: Path) -> list[str]:
    _require_unzip()
    result = subprocess.run(
        ["unzip", "-Z", "-1", str(zip_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _list_tar_members(tar_path: Path) -> list[str]:
    _require_tar()
    result = subprocess.run(
        ["tar", "-taf", str(tar_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _extract_zip(
    zip_path: Path, extract_to: Path, workers: Optional[int] = None
) -> None:
    _ensure_dir(extract_to)

    members = _list_zip_members(zip_path)
    for member in members:
        if not _is_safe_member_path(extract_to, member):
            raise ValueError(f"Unsafe path in zip: {member}")

    marker = _extract_marker(extract_to, zip_path)
    if marker.exists():
        print(f"[skip] {zip_path.name} already extracted")
        return

    print(f"[unzip] {zip_path.name} -> {extract_to}")
    # Using -n to keep already-complete files and resume missing members.
    subprocess.run(
        ["unzip", "-n", "-q", str(zip_path), "-d", str(extract_to)],
        check=True,
    )
    marker.touch()


def _move_archive_to_partial(
    archive_path: Path,
    url: str,
    opener: Optional[urllib.request.OpenerDirector] = None,
) -> None:
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    if not archive_path.exists():
        return
    if (
        partial_path.exists()
        and partial_path.stat().st_size >= archive_path.stat().st_size
    ):
        archive_path.unlink()
    else:
        archive_path.replace(partial_path)

    remote_size = _remote_size(url, opener=opener)
    if (
        remote_size is not None
        and partial_path.exists()
        and partial_path.stat().st_size >= remote_size
    ):
        partial_path.unlink()


def _download_and_extract_zip(
    url: str,
    zip_path: Path,
    extract_to: Path,
    workers: Optional[int] = None,
) -> None:
    _download(url, zip_path)
    try:
        _extract_zip(zip_path, extract_to, workers=workers)
    except (EOFError, RuntimeError, subprocess.CalledProcessError):
        print(f"[resume] {zip_path} is incomplete or corrupt; resuming download")
        _move_archive_to_partial(zip_path, url)
        _download(url, zip_path)
        _extract_zip(zip_path, extract_to, workers=workers)


def _safe_extract_tar(tar_path: Path, extract_to: Path, mode: str = "r:*") -> None:
    _ensure_dir(extract_to)
    marker = _extract_marker(extract_to, tar_path)
    members = _list_tar_members(tar_path)
    for member in members:
        if not _is_safe_member_path(extract_to, member):
            raise ValueError(f"Unsafe path in tar: {member}")

    if marker.exists():
        print(f"[skip] {tar_path.name} already extracted")
        return

    print(f"[extract] {tar_path.name} -> {extract_to}")
    subprocess.run(
        ["tar", "--no-same-owner", "-xaf", str(tar_path), "-C", str(extract_to)],
        check=True,
    )
    marker.touch()


def _download_and_extract_tar(
    url: str, tar_path: Path, extract_to: Path, mode: str = "r:*"
) -> None:
    _download(url, tar_path)
    try:
        _safe_extract_tar(tar_path, extract_to, mode=mode)
    except (EOFError, RuntimeError, subprocess.CalledProcessError):
        print(f"[resume] {tar_path} is incomplete or corrupt; resuming download")
        _move_archive_to_partial(tar_path, url)
        _download(url, tar_path)
        _safe_extract_tar(tar_path, extract_to, mode=mode)


def _resize_navi_image(path: Path, interp: int, min_size: int) -> None:
    if path.name.startswith("downsampled_"):
        return
    output_path = path.with_name(f"downsampled_{path.name}")
    if output_path.exists():
        return

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        if min(width, height) <= 0:
            return
        scale = float(min_size) / float(min(width, height))
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, interp)
        image.save(output_path)


def _parallel_map(paths: Iterable[Path], workers: int, desc: str, fn) -> None:
    path_list = list(paths)
    if not path_list:
        print(f"[skip] {desc}: no files found")
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(
            tqdm(
                executor.map(fn, path_list),
                total=len(path_list),
                desc=desc,
            )
        )


def _downsample_navi(data_root: Path, workers: Optional[int] = None) -> None:
    workers = max(1, workers or _available_workers())

    rgb_paths = data_root.glob("*/*/images/*.jpg")
    depth_paths = data_root.glob("*/*/depth/*.png")

    print("[process] NAVI downsample rgb")
    _parallel_map(
        rgb_paths,
        workers,
        "NAVI rgb",
        lambda path: _resize_navi_image(path, Image.BICUBIC, NAVI_DOWNSAMPLE_MIN_SIZE),
    )

    print("[process] NAVI downsample depth")
    _parallel_map(
        depth_paths,
        workers,
        "NAVI depth",
        lambda path: _resize_navi_image(path, Image.NEAREST, NAVI_DOWNSAMPLE_MIN_SIZE),
    )


def _download_cocostuff(data_dir: Path, workers: Optional[int] = None) -> None:
    cocostuff_dir = data_dir / "COCOStuff"
    downloads_dir = cocostuff_dir / "downloads"
    images_dir = cocostuff_dir / "dataset" / "images"
    ann_dir = cocostuff_dir / "dataset" / "annotations"

    _ensure_dir(downloads_dir)
    _ensure_dir(images_dir)
    _ensure_dir(ann_dir)

    train_zip = downloads_dir / "train2017.zip"
    val_zip = downloads_dir / "val2017.zip"
    stuff_zip = downloads_dir / "stuffthingmaps_trainval2017.zip"

    print("[unzip] COCO images and annotations")
    if _archive_extracted(images_dir, train_zip.name):
        print("[skip] COCO train2017 images already extracted")
    else:
        _download_and_extract_zip(
            "http://images.cocodataset.org/zips/train2017.zip",
            train_zip,
            images_dir,
            workers=workers,
        )

    if _archive_extracted(images_dir, val_zip.name):
        print("[skip] COCO val2017 images already extracted")
    else:
        _download_and_extract_zip(
            "http://images.cocodataset.org/zips/val2017.zip",
            val_zip,
            images_dir,
            workers=workers,
        )

    if _archive_extracted(ann_dir, stuff_zip.name):
        print("[skip] COCO stuffthingmaps annotations already extracted")
    else:
        _download_and_extract_zip(
            "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip",
            stuff_zip,
            ann_dir,
            workers=workers,
        )

    curated_tar = downloads_dir / "COCOStuff164kCurated.tar.gz"
    curated_dir = cocostuff_dir / "curated"
    if curated_dir.exists() and _archive_extracted(downloads_dir, curated_tar.name):
        print("[skip] COCOStuff curated already extracted")
    else:
        print("[extract] COCOStuff curated")
        _download_and_extract_tar(
            "https://www.robots.ox.ac.uk/~xuji/datasets/COCOStuff164kCurated.tar.gz",
            curated_tar,
            downloads_dir,
        )
        extracted_root = downloads_dir / "COCO" / "CocoStuff164k" / "curated"
        if not extracted_root.exists():
            raise RuntimeError(
                f"COCOStuff curated folder not found at {extracted_root}"
            )
        if curated_dir.exists():
            shutil.rmtree(curated_dir)
        shutil.move(str(extracted_root), curated_dir)
        if curated_tar.exists():
            curated_tar.unlink()

    print("[done] COCOStuff")


def _login_cityscapes(username: str, password: str) -> urllib.request.OpenerDirector:
    cookie_jar = urllib.request.HTTPCookieProcessor()
    opener = urllib.request.build_opener(cookie_jar)

    login_url = "https://www.cityscapes-dataset.com/login/"
    post_data = urllib.parse.urlencode(
        {
            "username": username,
            "password": password,
            "submit": "Login",
        }
    ).encode("utf-8")

    print("[login] Cityscapes")
    request = urllib.request.Request(login_url, data=post_data)
    with opener.open(request) as resp:
        _ = resp.read()
    return opener


def _download_cityscapes(data_dir: Path, username: str, password: str) -> None:
    if not username or not password:
        print("[skip] Cityscapes requires --city_un and --city_pwd")
        return

    city_dir = data_dir / "cityscapes"
    _ensure_dir(city_dir)

    opener = _login_cityscapes(username, password)

    packages = {
        "leftImg8bit_trainvaltest.zip": "https://www.cityscapes-dataset.com/file-handling/?packageID=1",
        "gtFine_trainvaltest.zip": "https://www.cityscapes-dataset.com/file-handling/?packageID=3",
    }

    for fname, url in packages.items():
        dest = city_dir / fname
        _download(url, dest, opener=opener)

    print("[done] Cityscapes")


def _download_voc(data_dir: Path) -> None:
    voc_dir = data_dir / "VOCdevkit" / "VOC2012"
    voc_tar = data_dir / "VOCtrainval_11-May-2012.tar"
    if voc_dir.exists() and _archive_extracted(data_dir, voc_tar.name):
        print("[skip] VOC already extracted")
        return

    print("[extract] VOC")
    _download_and_extract_tar(
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        voc_tar,
        data_dir,
    )
    if voc_tar.exists():
        voc_tar.unlink()
    print("[done] VOC")


def _download_ade20k(data_dir: Path, workers: Optional[int] = None) -> None:
    ade_dir = data_dir / "ADEChallengeData2016"
    ade_zip = data_dir / "ADEChallengeData2016.zip"
    if ade_dir.exists() and _archive_extracted(data_dir, ade_zip.name):
        print("[skip] ADE20K already extracted")
        return

    print("[extract] ADE20K")
    _download_and_extract_zip(
        "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
        ade_zip,
        data_dir,
        workers=workers,
    )
    if ade_zip.exists():
        ade_zip.unlink()
    print("[done] ADE20K")


def _download_spair(data_dir: Path) -> None:
    spair_dir = data_dir / "SPair-71k"
    spair_tar = data_dir / "SPair-71k.tar.gz"
    if spair_dir.exists() and _archive_extracted(data_dir, spair_tar.name):
        print("[skip] SPair-71k already extracted")
        return

    print("[extract] SPair-71k")
    _download_and_extract_tar(
        "http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz",
        spair_tar,
        data_dir,
    )
    if spair_tar.exists():
        spair_tar.unlink()
    print("[done] SPair-71k")


def _download_navi(data_dir: Path, workers: Optional[int] = None) -> None:
    navi_root = data_dir / NAVI_VERSION
    navi_tar = data_dir / f"{NAVI_VERSION}.tar.gz"
    if navi_root.exists() and not _archive_extracted(data_dir, navi_tar.name):
        if _navi_extracted(data_dir):
            _extract_marker(data_dir, navi_tar).touch()
    if navi_root.exists() and _archive_extracted(data_dir, navi_tar.name):
        print("[skip] NAVI already extracted")
    else:
        print("[extract] NAVI")
        _download_and_extract_tar(
            f"http://storage.googleapis.com/gresearch/navi-dataset/{NAVI_VERSION}.tar.gz",
            navi_tar,
            data_dir,
        )
        if navi_tar.exists():
            navi_tar.unlink()

    if not navi_root.exists():
        raise RuntimeError(f"NAVI folder not found at {navi_root}")

    _downsample_navi(navi_root, workers=workers)
    print("[done] NAVI")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sync_tree(
    src: Path,
    dst: Path,
    ignore_dirs: Optional[set[str]] = None,
    ignore_suffixes: Optional[set[str]] = None,
) -> None:
    if not src.exists():
        return

    ignore_dirs = ignore_dirs or set()
    ignore_suffixes = ignore_suffixes or set()
    ignore_suffixes.add(".extracted")
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        rel_root = root_path.relative_to(src)
        if any(part in ignore_dirs for part in rel_root.parts):
            dirs[:] = []
            continue

        target_root = dst / rel_root
        _ensure_dir(target_root)

        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for name in files:
            src_file = root_path / name
            if any(name.endswith(suffix) for suffix in ignore_suffixes):
                continue
            dst_file = target_root / name
            if dst_file.exists() and dst_file.stat().st_size == src_file.stat().st_size:
                continue
            shutil.copy2(src_file, dst_file)


def _staging_dir(path: Optional[str]) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    return (_project_root() / "tmp" / "data").resolve()


def _cleanup_staging(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _non_empty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _coco_ready(data_dir: Path) -> bool:
    coco_root = data_dir / "COCOStuff"
    required_dirs = [
        coco_root / "dataset" / "images" / "train2017",
        coco_root / "dataset" / "images" / "val2017",
        coco_root / "dataset" / "annotations" / "train2017",
        coco_root / "dataset" / "annotations" / "val2017",
        coco_root / "curated",
    ]
    return all(path.is_dir() for path in required_dirs)


def _cityscapes_ready(data_dir: Path) -> bool:
    city_root = data_dir / "cityscapes"
    required_files = [
        city_root / "leftImg8bit_trainvaltest.zip",
        city_root / "gtFine_trainvaltest.zip",
    ]
    return all(_non_empty_file(path) for path in required_files)


def _voc_ready(data_dir: Path) -> bool:
    voc_root = data_dir / "VOCdevkit" / "VOC2012"
    required_dirs = [
        voc_root / "Annotations",
        voc_root / "JPEGImages",
        voc_root / "ImageSets",
    ]
    return all(path.is_dir() for path in required_dirs)


def _ade_ready(data_dir: Path) -> bool:
    ade_root = data_dir / "ADEChallengeData2016"
    required_dirs = [
        ade_root / "images" / "training",
        ade_root / "images" / "validation",
        ade_root / "annotations" / "training",
        ade_root / "annotations" / "validation",
    ]
    return all(path.is_dir() for path in required_dirs)


def _spair_ready(data_dir: Path) -> bool:
    spair_root = data_dir / "SPair-71k"
    required_dirs = [
        spair_root / "JPEGImages",
        spair_root / "ImageAnnotation",
        spair_root / "PairAnnotation",
    ]
    return all(path.is_dir() for path in required_dirs)


def _navi_ready(data_dir: Path) -> bool:
    navi_root = data_dir / NAVI_VERSION
    if not navi_root.is_dir():
        return False
    rgb_sample = next(navi_root.glob("*/*/images/downsampled_*.jpg"), None)
    depth_sample = next(navi_root.glob("*/*/depth/downsampled_*.png"), None)
    return rgb_sample is not None and depth_sample is not None


def _navi_extracted(data_dir: Path) -> bool:
    navi_root = data_dir / NAVI_VERSION
    if not navi_root.is_dir():
        return False
    rgb_sample = next(navi_root.glob("*/*/images/*.jpg"), None)
    depth_sample = next(navi_root.glob("*/*/depth/*.png"), None)
    return rgb_sample is not None and depth_sample is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download datasets used in this repo.")
    parser.add_argument("--data_dir", default="data", help="Root data directory")
    parser.add_argument(
        "--staging_dir",
        default=None,
        help="Local extraction directory (default: <project>/tmp/data)",
    )
    parser.add_argument("--city_un", default="", help="Cityscapes username")
    parser.add_argument("--city_pwd", default="", help="Cityscapes password")
    parser.add_argument(
        "--extract_workers",
        type=int,
        default=None,
        help=(
            "ZIP extraction workers. Defaults to detected container CPUs minus one, "
            f"capped at {DEFAULT_MAX_EXTRACT_WORKERS}."
        ),
    )
    parser.add_argument(
        "--ds",
        nargs="*",
        default=None,
        choices=["coco", "cityscapes", "voc", "ade", "spair", "navi"],
        help="Datasets to process (default: all)",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["coco", "cityscapes", "voc", "ade", "spair", "navi"],
        help="Datasets to skip",
    )
    parser.add_argument(
        "--no_cleanup_tmp",
        action="store_false",
        default=True,
        dest="cleanup_tmp",
        help="Disable cleanup of staging dataset directories after syncing to data_dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    staging_dir = _staging_dir(args.staging_dir)
    _ensure_dir(data_dir)
    _ensure_dir(staging_dir)
    extract_workers = _available_workers(args.extract_workers)

    print(f"[info] data_dir={data_dir}")
    print(f"[info] staging_dir={staging_dir}")
    print(f"[info] extract_workers={extract_workers}")
    print("[note] ImageNet must be downloaded manually to data/imagenet")

    all_datasets = {"coco", "cityscapes", "voc", "ade", "spair", "navi"}
    selected = set(args.ds) if args.ds else set(all_datasets)
    selected -= set(args.skip)

    if "coco" in selected:
        coco_target = data_dir / "COCOStuff"
        did_sync = False
        if data_dir != staging_dir and _coco_ready(data_dir):
            print("[skip] COCOStuff already present in data_dir")
        else:
            _download_cocostuff(staging_dir, workers=extract_workers)
            if data_dir != staging_dir:
                _sync_tree(
                    staging_dir / "COCOStuff",
                    coco_target,
                    ignore_dirs={"downloads"},
                    ignore_suffixes={".zip", ".tar", ".tar.gz", ".tgz"},
                )
                did_sync = True
        if args.cleanup_tmp and did_sync:
            _cleanup_staging(staging_dir / "COCOStuff")

    if "cityscapes" in selected:
        city_target = data_dir / "cityscapes"
        did_sync = False
        if data_dir != staging_dir and _cityscapes_ready(data_dir):
            print("[skip] Cityscapes already present in data_dir")
        else:
            _download_cityscapes(staging_dir, args.city_un, args.city_pwd)
            if data_dir != staging_dir:
                _sync_tree(
                    staging_dir / "cityscapes",
                    city_target,
                )
                did_sync = True
        if args.cleanup_tmp and did_sync:
            _cleanup_staging(staging_dir / "cityscapes")

    if "voc" in selected:
        voc_target = data_dir / "VOCdevkit" / "VOC2012"
        did_sync = False
        if data_dir != staging_dir and _voc_ready(data_dir):
            print("[skip] VOC already present in data_dir")
        else:
            _download_voc(staging_dir)
            if data_dir != staging_dir:
                _sync_tree(
                    staging_dir / "VOCdevkit",
                    data_dir / "VOCdevkit",
                    ignore_suffixes={".zip", ".tar", ".tar.gz", ".tgz"},
                )
                did_sync = True
        if args.cleanup_tmp and did_sync:
            _cleanup_staging(staging_dir / "VOCdevkit")

    if "ade" in selected:
        ade_target = data_dir / "ADEChallengeData2016"
        did_sync = False
        if data_dir != staging_dir and _ade_ready(data_dir):
            print("[skip] ADE20K already present in data_dir")
        else:
            _download_ade20k(staging_dir, workers=extract_workers)
            if data_dir != staging_dir:
                _sync_tree(
                    staging_dir / "ADEChallengeData2016",
                    ade_target,
                    ignore_suffixes={".zip", ".tar", ".tar.gz", ".tgz"},
                )
                did_sync = True
        if args.cleanup_tmp and did_sync:
            _cleanup_staging(staging_dir / "ADEChallengeData2016")

    if "spair" in selected:
        spair_target = data_dir / "SPair-71k"
        did_sync = False
        if data_dir != staging_dir and _spair_ready(data_dir):
            print("[skip] SPair-71k already present in data_dir")
        else:
            _download_spair(staging_dir)
            if data_dir != staging_dir:
                _sync_tree(
                    staging_dir / "SPair-71k",
                    spair_target,
                    ignore_suffixes={".zip", ".tar", ".tar.gz", ".tgz"},
                )
                did_sync = True
        if args.cleanup_tmp and did_sync:
            _cleanup_staging(staging_dir / "SPair-71k")

    if "navi" in selected:
        navi_target = data_dir / NAVI_VERSION
        did_sync = False
        if data_dir != staging_dir:
            if _navi_ready(data_dir):
                print("[skip] NAVI already present in data_dir")
            elif navi_target.exists():
                print("[process] NAVI downsample in data_dir")
                _downsample_navi(navi_target, workers=extract_workers)
            else:
                _download_navi(staging_dir, workers=extract_workers)
                _sync_tree(
                    staging_dir / NAVI_VERSION,
                    navi_target,
                    ignore_suffixes={".zip", ".tar", ".tar.gz", ".tgz"},
                )
                did_sync = True
        else:
            _download_navi(staging_dir, workers=extract_workers)
        if args.cleanup_tmp and did_sync:
            _cleanup_staging(staging_dir / NAVI_VERSION)

    print("[done] All requested datasets")


if __name__ == "__main__":
    main()
