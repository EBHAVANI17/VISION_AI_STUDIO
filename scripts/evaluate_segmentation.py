"""Semantic segmentation metrics (mIoU, per-class IoU, pixel accuracy).

Two evaluation tracks:

* ``coco-voc``  DeepLabV3 COCO/VOC weights on COCO val2017. Instance masks are rasterized into
  the 21 VOC classes; unmapped COCO categories count as background.
* ``ade``       SegFormer ADE20K weights on the ADE20K validation split (images + label PNGs).

Both modes resize inputs to ``--max-edge`` before inference and resize the ground truth with
nearest-neighbour, so numbers are only comparable across runs with the same setting.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from vision_ai_studio.image import load_image
from vision_ai_studio.models.registry import get_segmenter, resolve_device

VOC_LABELS: tuple[str, ...] = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

COCO_TO_VOC: dict[str, str] = {
    "airplane": "aeroplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "dining table": "diningtable",
    "dog": "dog",
    "horse": "horse",
    "motorcycle": "motorbike",
    "person": "person",
    "potted plant": "pottedplant",
    "sheep": "sheep",
    "couch": "sofa",
    "sofa": "sofa",
    "train": "train",
    "tv": "tvmonitor",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["coco-voc", "ade"], required=True)
    parser.add_argument("--seg", choices=["voc", "ade"], default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=500, help="Images to evaluate; 0 = all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-edge", type=int, default=640)
    parser.add_argument("--coco-root", type=Path, default=Path("data/coco"))
    parser.add_argument("--images", type=Path, help="ADE20K validation images directory.")
    parser.add_argument("--masks", type=Path, help="ADE20K validation annotation PNGs.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    segmenter_name = args.seg or ("voc" if args.mode == "coco-voc" else "ade")
    if args.mode == "coco-voc" and segmenter_name != "voc":
        parser.error("coco-voc mode needs --seg voc: the ADE label space does not match COCO.")
    if args.mode == "ade" and segmenter_name != "ade":
        parser.error("ade mode needs --seg ade.")

    segmenter = get_segmenter(segmenter_name, resolve_device(args.device))
    labels = list(getattr(segmenter, "labels", ())) or list(VOC_LABELS)
    output = args.output or Path("artifacts/eval") / f"segmentation_{args.mode}.json"

    if args.mode == "coco-voc":
        return _run_coco(args, segmenter, labels, output)
    return _run_ade(args, segmenter, labels, output)


def _run_coco(args: argparse.Namespace, segmenter, labels: list[str], output: Path) -> int:
    try:
        from pycocotools.coco import COCO
    except ImportError:
        print("Install evaluation extras: pip install -e '.[eval]'", file=sys.stderr)
        return 1

    annotations = args.coco_root / "annotations" / "instances_val2017.json"
    images_dir = args.coco_root / "val2017"
    if not annotations.is_file() or not images_dir.is_dir():
        print(
            f"Missing {annotations} or {images_dir}; run scripts/download_coco.py.", file=sys.stderr
        )
        return 1

    coco = COCO(str(annotations))
    categories = {
        category["id"]: category["name"].strip().lower()
        for category in coco.loadCats(coco.getCatIds())
    }
    name_to_voc = {
        name: VOC_LABELS.index(voc_name)
        for name in set(categories.values())
        if (voc_name := COCO_TO_VOC.get(name))
    }
    print(f"mapped COCO categories: {sorted(name_to_voc)}")

    image_ids = sorted(coco.getImgIds())
    if args.limit and args.limit < len(image_ids):
        image_ids = sorted(random.Random(args.seed).sample(image_ids, args.limit))

    num_classes = len(labels)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    evaluated = 0
    skipped = 0
    for index, image_id in enumerate(image_ids, start=1):
        info = coco.loadImgs(image_id)[0]
        path = images_dir / info["file_name"]
        if not path.is_file():
            skipped += 1
            continue

        image = load_image(path, max_edge=args.max_edge)
        ground_truth = _coco_ground_truth(coco, info, categories, name_to_voc)
        ground_truth = _resize_mask(ground_truth, image.size)
        prediction = segmenter.predict(image).mask
        _accumulate(confusion, ground_truth, prediction, num_classes)
        evaluated += 1

        if evaluated % 25 == 0 or index == len(image_ids):
            print(
                f"\r  evaluated {evaluated}/{len(image_ids)} (skipped {skipped})",
                end="",
                flush=True,
            )
    print()

    report = _report(confusion, labels)
    report.update(
        {
            "mode": "coco-voc",
            "segmenter": segmenter.name,
            "version": segmenter.version,
            "images": evaluated,
            "skipped": skipped,
            "max_edge": args.max_edge,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "caveats": [
                "COCO's 80 thing categories map onto 20 VOC classes; unmapped categories count as background.",
                "Crowd annotations are ignored.",
                "Overlapping instances resolve to the smaller object.",
                f"Metrics at max-edge {args.max_edge}px.",
            ],
        }
    )
    _finish(report, output)
    return 0


def _coco_ground_truth(
    coco,
    info: dict,
    categories: dict[int, str],
    name_to_voc: dict[str, int],
) -> np.ndarray:
    height, width = int(info["height"]), int(info["width"])
    ground_truth = np.zeros((height, width), dtype=np.int32)
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=info["id"]))
    annotations.sort(key=lambda ann: float(ann.get("area", 0)), reverse=True)
    for annotation in annotations:
        mask = coco.annToMask(annotation).astype(bool)
        if annotation.get("iscrowd"):
            ground_truth[mask] = -1
            continue
        category = categories.get(annotation["category_id"], "")
        voc_id = name_to_voc.get(category)
        if voc_id is not None:
            ground_truth[mask] = voc_id
    return ground_truth


def _run_ade(args: argparse.Namespace, segmenter, labels: list[str], output: Path) -> int:
    if not args.images or not args.masks:
        print("--images and --masks are required for ADE mode.", file=sys.stderr)
        return 1
    pairs = [
        (image_path, args.masks / f"{image_path.stem}.png")
        for image_path in sorted(args.images.glob("*.jpg"))
    ]
    pairs = [(image_path, mask_path) for image_path, mask_path in pairs if mask_path.is_file()]
    if not pairs:
        print(f"No image/mask pairs under {args.images} and {args.masks}.", file=sys.stderr)
        return 1
    if args.limit and args.limit < len(pairs):
        pairs = sorted(random.Random(args.seed).sample(pairs, args.limit))
    return _evaluate_ade_pairs(pairs, segmenter, labels, args, output)


def _evaluate_ade_pairs(pairs, segmenter, labels: list[str], args, output: Path) -> int:
    num_classes = len(labels)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        image = load_image(image_path, max_edge=args.max_edge)
        raw = np.asarray(Image.open(mask_path))
        if raw.ndim != 2:
            raise ValueError(f"{mask_path} is not a single-channel PNG")
        ground_truth = raw.astype(np.int32) - 1
        ground_truth[(raw <= 0) | (raw > num_classes)] = -1
        ground_truth = _resize_mask(ground_truth, image.size)
        prediction = segmenter.predict(image).mask
        _accumulate(confusion, ground_truth, prediction, num_classes)
        if index % 25 == 0 or index == len(pairs):
            print(f"\r  evaluated {index}/{len(pairs)}", end="", flush=True)
    print()

    report = _report(confusion, labels)
    report.update(
        {
            "mode": "ade",
            "segmenter": segmenter.name,
            "version": segmenter.version,
            "images": len(pairs),
            "skipped": 0,
            "max_edge": args.max_edge,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "caveats": [
                "ADE20K PNG labels are shifted by one: raw 1..150 -> model classes 0..149; raw 0/255 is ignored.",
                f"Metrics at max-edge {args.max_edge}px.",
            ],
        }
    )
    _finish(report, output)
    return 0


def _accumulate(
    confusion: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    num_classes: int,
) -> None:
    if prediction.shape != ground_truth.shape:
        prediction = _resize_mask(prediction, (ground_truth.shape[1], ground_truth.shape[0]))
    valid = ground_truth >= 0
    if not valid.any():
        return
    truth = ground_truth[valid].astype(np.int64)
    predicted = np.clip(prediction[valid], 0, num_classes - 1).astype(np.int64)
    counts = np.bincount(truth * num_classes + predicted, minlength=num_classes**2)
    confusion += counts.reshape(num_classes, num_classes)


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if mask.shape == (size[1], size[0]):
        return mask
    resized = Image.fromarray(mask.astype(np.int32), mode="I").resize(
        size, Image.Resampling.NEAREST
    )
    return np.asarray(resized, dtype=np.int32)


def _report(confusion: np.ndarray, labels: list[str]) -> dict:
    true_positive = np.diag(confusion).astype(np.float64)
    false_positive = confusion.sum(axis=0) - true_positive
    false_negative = confusion.sum(axis=1) - true_positive
    denominator = true_positive + false_positive + false_negative
    iou = np.divide(
        true_positive,
        denominator,
        out=np.full_like(true_positive, np.nan),
        where=denominator > 0,
    )
    present = confusion.sum(axis=1) > 0
    foreground = present.copy()
    foreground[0] = False

    classes = [
        {
            "class_id": index,
            "label": label,
            "iou": _nan_to_none(iou[index]),
            "gt_pixels": int(confusion[index].sum()),
            "pred_pixels": int(confusion[:, index].sum()),
        }
        for index, label in enumerate(labels)
    ]
    total = confusion.sum()
    return {
        "miou": _nanmean(iou[present]),
        "miou_foreground": _nanmean(iou[foreground]),
        "pixel_accuracy": float(true_positive.sum() / total) if total else None,
        "classes": classes,
    }


def _finish(report: dict, output: Path) -> None:
    print()
    print(f"{'class':<16}{'IoU':>8}{'GT pixels':>12}{'pred pixels':>14}")
    for row in report["classes"]:
        iou = "n/a" if row["iou"] is None else f"{row['iou']:.3f}"
        print(f"{row['label']:<16}{iou:>8}{row['gt_pixels']:>12,}{row['pred_pixels']:>14,}")
    print()
    print(f"mIoU (all present classes): {_fmt(report['miou'])}")
    print(f"mIoU (foreground classes):  {_fmt(report['miou_foreground'])}")
    print(f"pixel accuracy:             {_fmt(report['pixel_accuracy'])}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {output}")


def _nanmean(values: np.ndarray) -> float | None:
    finite = values[~np.isnan(values)]
    return float(finite.mean()) if finite.size else None


def _nan_to_none(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
