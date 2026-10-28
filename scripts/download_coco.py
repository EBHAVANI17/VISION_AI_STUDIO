"""Download COCO 2017 images and annotations from the official CDN."""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "http://images.cocodataset.org"
URLS: dict[str, str] = {
    "val2017": f"{BASE}/zips/val2017.zip",
    "train2017": f"{BASE}/zips/train2017.zip",
    "annotations": f"{BASE}/annotations/annotations_trainval2017.zip",
    "stuff": f"{BASE}/annotations/stuff_annotations_trainval2017.zip",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/coco"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val2017"],
        choices=["val2017", "train2017"],
        help="Image splits to download (val2017 is ~1 GB, train2017 ~18 GB).",
    )
    parser.add_argument(
        "--annotations", action="store_true", help="caption/instance annotations (~241 MB)"
    )
    parser.add_argument("--stuff", action="store_true", help="COCO-Stuff annotations (~521 MB)")
    parser.add_argument("--keep-zips", action="store_true")
    args = parser.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)
    wanted = list(args.splits)
    if args.annotations:
        wanted.append("annotations")
    if args.stuff:
        wanted.append("stuff")

    for name in wanted:
        zip_path = args.dest / f"{name}.zip"
        print(f"[{name}] {URLS[name]}")
        _download(URLS[name], zip_path)
        print(f"[{name}] extracting into {args.dest}")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(args.dest)
        if not args.keep_zips:
            zip_path.unlink()

    print(f"done: {args.dest}")
    print("  images:      val2017/ (or train2017/)")
    print("  captions:    annotations/captions_val2017.json")
    print("  instances:   annotations/instances_val2017.json")
    return 0


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already present: {dest.name}")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        written = 0
        next_report = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            if total:
                percent = int(written * 100 / total)
                if percent >= next_report:
                    sys.stdout.write(
                        f"\r  {dest.name}: {percent:3d}% ({written / 1e6:.0f}/{total / 1e6:.0f} MB)"
                    )
                    sys.stdout.flush()
                    next_report = percent // 10 * 10 + 10
    sys.stdout.write("\r" + " " * 72 + "\r")
    part.replace(dest)
    print(f"  downloaded {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    raise SystemExit(main())
