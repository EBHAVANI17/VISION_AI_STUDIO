"""Caption metrics on COCO val2017: BLEU-1..4, METEOR, ROUGE-L, CIDEr (SPICE optional).

Requires ``pip install -e '.[eval]'`` plus a Java runtime for METEOR/SPICE.
Raw predictions are written next to the report so scores can be recomputed without re-running
the model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from vision_ai_studio.image import load_image
from vision_ai_studio.models.registry import get_captioner, resolve_device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=Path("data/coco"))
    parser.add_argument("--caption", choices=["blip-base", "blip-large"], default="blip-base")
    parser.add_argument("--lora", default=None, help="Optional HF LoRA adapter repository.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=500, help="Images to score; 0 = all 5000.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument("--spice", action="store_true", help="Also compute SPICE (needs Java).")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        from pycocotools.coco import COCO
    except ImportError:
        print("Install evaluation extras: pip install -e '.[eval]'", file=sys.stderr)
        return 1

    annotations = args.coco_root / "annotations" / "captions_val2017.json"
    images_dir = args.coco_root / "val2017"
    if not annotations.is_file() or not images_dir.is_dir():
        print(
            f"Missing {annotations} or {images_dir}; run scripts/download_coco.py.", file=sys.stderr
        )
        return 1

    coco = COCO(str(annotations))
    image_ids = sorted(coco.getImgIds())
    if args.limit and args.limit < len(image_ids):
        image_ids = sorted(random.Random(args.seed).sample(image_ids, args.limit))

    captioner = get_captioner(
        args.caption,
        resolve_device(args.device),
        None,
        args.lora,
    )

    references: dict[int, list[str]] = {}
    hypotheses: dict[int, list[str]] = {}
    skipped = 0
    for index, image_id in enumerate(image_ids, start=1):
        info = coco.loadImgs(image_id)[0]
        path = images_dir / info["file_name"]
        if not path.is_file():
            skipped += 1
            continue
        captions = [
            annotation["caption"].strip()
            for annotation in coco.loadAnns(coco.getAnnIds(imgIds=image_id))
            if annotation["caption"].strip()
        ]
        if not captions:
            skipped += 1
            continue
        image = load_image(path, max_edge=args.max_edge)
        references[image_id] = captions
        hypotheses[image_id] = [captioner.caption(image, beam=args.beam)]
        if index % 25 == 0 or index == len(image_ids):
            print(
                f"\r  captioned {len(hypotheses)}/{len(image_ids)} (skipped {skipped})",
                end="",
                flush=True,
            )
    print()

    if not hypotheses:
        print("No captions generated.", file=sys.stderr)
        return 1

    output = args.output or Path("artifacts/eval") / f"captioning_{args.caption}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output.with_name(output.stem + "_predictions.json")
    predictions_path.write_text(
        json.dumps(
            {str(image_id): caps[0] for image_id, caps in hypotheses.items()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    scores, errors = _score(references, hypotheses, spice=args.spice)
    report = {
        "mode": "coco-captions",
        "captioner": args.caption,
        "version": captioner.version,
        "lora": args.lora,
        "images": len(hypotheses),
        "skipped": skipped,
        "beam": args.beam,
        "max_edge": args.max_edge,
        "scores": scores,
        "errors": errors,
        "predictions": str(predictions_path),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caveats": [
            "Five reference captions per image from COCO val2017.",
            "METEOR and SPICE require Java; failures are reported per metric.",
            f"Captions generated at max-edge {args.max_edge}px.",
        ],
    }

    print()
    for name, value in scores.items():
        print(f"{name:<10} {value:.4f}")
    for name, message in errors.items():
        print(f"{name:<10} skipped ({message})")

    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    return 0


def _score(
    references: dict[int, list[str]],
    hypotheses: dict[int, list[str]],
    *,
    spice: bool,
) -> tuple[dict[str, float], dict[str, str]]:
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError as exc:
        return {}, {"all": f"pycocoevalcap unavailable: {exc}"}

    def run(name: str, scorer, keys: list[str] | None = None) -> list[float] | None:
        try:
            value, _ = scorer.compute_score(references, hypotheses)
            values = value if isinstance(value, list) else [value]
            values = [float(item) for item in values]
            if keys:
                for key, item in zip(keys, values, strict=True):
                    scores[key] = item
            else:
                scores[name] = values[0]
            return values
        except Exception as exc:
            errors[name] = str(exc)[:200]
            return None

    run("BLEU", Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"])
    run("METEOR", Meteor())
    run("ROUGE-L", Rouge())
    run("CIDEr", Cider())
    if spice:
        try:
            from pycocoevalcap.spice.spice import Spice

            run("SPICE", Spice())
        except ImportError as exc:
            errors["SPICE"] = f"unavailable: {exc}"
    return scores, errors


if __name__ == "__main__":
    raise SystemExit(main())
