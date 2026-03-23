from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
import yaml

from data.dataset import DEFAULT_FILL, IMAGE_SUFFIXES
from model.detector import VSTDet
from utils.torch_utils import format_device_name, model_summary, resolve_device, seed_everything


PALETTE = [
    (255, 99, 71),
    (65, 105, 225),
    (60, 179, 113),
    (255, 165, 0),
    (186, 85, 211),
    (220, 20, 60),
]


def load_config_defaults(config_path: str | None) -> dict:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")
    return data


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=None, help="Path to a YAML experiment config.")
    bootstrap_args, _ = bootstrap.parse_known_args()
    config_defaults = load_config_defaults(bootstrap_args.config)

    parser = argparse.ArgumentParser(
        parents=[bootstrap],
        description="Run inference on images and save visualization overlays.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint.")
    parser.add_argument("--source", required=True, help="Image file or directory to run inference on.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square inference size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--output",
        default=None,
        help="Directory for rendered predictions. Defaults to <checkpoint_dir>/detect.",
    )
    parser.add_argument(
        "--hide-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not draw class labels on the saved image.",
    )
    parser.add_argument(
        "--hide-scores",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not draw confidence scores on the saved image.",
    )
    parser.add_argument(
        "--save-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a predictions.json file alongside the rendered images.",
    )
    parser.set_defaults(**config_defaults)
    return parser.parse_args()


def build_model_from_checkpoint(
    checkpoint: dict[str, object],
    device: torch.device,
) -> tuple[VSTDet, list[str]]:
    checkpoint_names = checkpoint.get("names")
    if not isinstance(checkpoint_names, list):
        raise ValueError("Checkpoint is missing class names; cannot label detections.")

    model = VSTDet(
        num_classes=len(checkpoint_names),
        variant=str(checkpoint.get("variant", "small")),
        backbone_name=str(checkpoint.get("backbone_name", "efficientnet_v2_s")),
        pretrained_backbone=False,
        neck_name=str(checkpoint.get("neck_name", "bifusion")),
        head_depth=int(checkpoint.get("head_depth", 2)),
        use_detail_branch=bool(checkpoint.get("use_detail_branch", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])  # type: ignore[arg-type]
    return model, [str(name) for name in checkpoint_names]


def resolve_sources(source: str) -> tuple[list[Path], Path | None]:
    source_path = Path(source)
    if source_path.is_file():
        return [source_path], source_path.parent
    if source_path.is_dir():
        paths = sorted(
            path for path in source_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise FileNotFoundError(f"No supported images found in directory: {source_path}")
        return paths, source_path
    raise FileNotFoundError(f"Source does not exist: {source}")


def preprocess_image(
    image: Image.Image,
    size: int,
) -> tuple[torch.Tensor, float, int, int]:
    width, height = image.size
    scale = min(size / width, size / height)
    new_width = max(int(round(width * scale)), 1)
    new_height = max(int(round(height * scale)), 1)
    resized = image.resize((new_width, new_height), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), DEFAULT_FILL)
    pad_left = (size - new_width) // 2
    pad_top = (size - new_height) // 2
    canvas.paste(resized, (pad_left, pad_top))
    tensor = TF.pil_to_tensor(canvas).float() / 255.0
    return tensor, scale, pad_left, pad_top


def scale_boxes_to_original(
    boxes: torch.Tensor,
    scale: float,
    pad_left: int,
    pad_top: int,
    width: int,
    height: int,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    scaled = boxes.clone()
    scaled[:, [0, 2]] = (scaled[:, [0, 2]] - pad_left) / max(scale, 1e-6)
    scaled[:, [1, 3]] = (scaled[:, [1, 3]] - pad_top) / max(scale, 1e-6)
    scaled[:, 0::2] = scaled[:, 0::2].clamp_(0, width)
    scaled[:, 1::2] = scaled[:, 1::2].clamp_(0, height)
    return scaled


def label_color(class_id: int) -> tuple[int, int, int]:
    return PALETTE[class_id % len(PALETTE)]


def draw_prediction_overlay(
    image: Image.Image,
    prediction: dict[str, torch.Tensor],
    class_names: list[str],
    hide_labels: bool,
    hide_scores: bool,
) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    line_width = max(2, round(min(rendered.size) * 0.004))

    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()

    for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        x1, y1, x2, y2 = box
        color = label_color(int(label))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        caption_parts: list[str] = []
        if not hide_labels:
            caption_parts.append(class_names[int(label)])
        if not hide_scores:
            caption_parts.append(f"{float(score):.2f}")
        if not caption_parts:
            continue

        caption = " ".join(caption_parts)
        text_bbox = draw.textbbox((x1, y1), caption)
        text_height = text_bbox[3] - text_bbox[1]
        text_y1 = max(0, y1 - text_height - 4)
        text_y2 = text_y1 + text_height + 4
        text_x2 = x1 + (text_bbox[2] - text_bbox[0]) + 6
        draw.rectangle((x1, text_y1, text_x2, text_y2), fill=color)
        draw.text((x1 + 3, text_y1 + 2), caption, fill=(255, 255, 255))
    return rendered


def serialize_prediction(
    image_path: Path,
    prediction: dict[str, torch.Tensor],
    class_names: list[str],
) -> dict[str, object]:
    entries = []
    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        entries.append(
            {
                "class_id": int(label),
                "class_name": class_names[int(label)],
                "score": float(score),
                "box_xyxy": [round(float(value), 2) for value in box],
            }
        )
    return {"image": str(image_path), "detections": entries}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output) if args.output else checkpoint_path.parent / "detect"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths, source_root = resolve_sources(args.source)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, class_names = build_model_from_checkpoint(checkpoint, device)

    print(
        "detecting",
        f"checkpoint={checkpoint_path}",
        f"source={args.source}",
        f"images={len(image_paths)}",
        f"imgsz={args.imgsz}",
        f"conf_threshold={args.conf_threshold}",
        f"nms_iou={args.nms_iou}",
    )
    print("device", f"name={format_device_name(device)}", f"type={device.type}")
    print("model", model_summary(model))

    serialized_predictions: list[dict[str, object]] = []
    total_detections = 0

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_tensor, scale, pad_left, pad_top = preprocess_image(image, args.imgsz)
        prediction = model.predict(
            image_tensor.unsqueeze(0).to(device),
            conf_threshold=args.conf_threshold,
            nms_iou=args.nms_iou,
            max_det=args.max_det,
        )[0]
        prediction["boxes"] = scale_boxes_to_original(
            prediction["boxes"],
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            width=width,
            height=height,
        )
        total_detections += int(prediction["scores"].numel())

        rendered = draw_prediction_overlay(
            image=image,
            prediction=prediction,
            class_names=class_names,
            hide_labels=args.hide_labels,
            hide_scores=args.hide_scores,
        )

        if source_root is not None:
            relative_path = image_path.relative_to(source_root)
            save_path = output_dir / relative_path
        else:
            save_path = output_dir / image_path.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(save_path)

        if args.save_json:
            serialized_predictions.append(
                serialize_prediction(image_path=image_path, prediction=prediction, class_names=class_names)
            )

    if args.save_json:
        (output_dir / "predictions.json").write_text(
            json.dumps(serialized_predictions, indent=2),
            encoding="utf-8",
        )

    print(
        "saved",
        f"output_dir={output_dir}",
        f"rendered_images={len(image_paths)}",
        f"total_detections={total_detections}",
    )


if __name__ == "__main__":
    main()
