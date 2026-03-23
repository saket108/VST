from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
import yaml

from data.dataset import YoloDetectionDataset, collate_fn
from model.detector import VSTDet
from utils.evaluator import evaluate_detection_metrics, format_metrics_table
from utils.plots import plot_per_class_metrics
from utils.reporting import append_results_summary
from utils.torch_utils import format_device_name, model_summary, resolve_device, seed_everything


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
        description="Evaluate a saved VSTDet checkpoint on a YOLO-format split.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint.")
    parser.add_argument("--data", required=True, help="Path to dataset YAML in YOLO format.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640, help="Square evaluation size.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf-threshold", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--output",
        default=None,
        help="Directory for evaluation artifacts. Defaults to the checkpoint folder.",
    )
    parser.set_defaults(**config_defaults)
    return parser.parse_args()


def build_loader(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, list[str]]:
    dataset = YoloDetectionDataset(
        args.data,
        split=args.split,
        image_size=args.imgsz,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
    )
    return loader, dataset.config.names


def build_model_from_checkpoint(
    checkpoint: dict[str, object],
    device: torch.device,
) -> tuple[VSTDet, list[str] | None]:
    checkpoint_names = checkpoint.get("names")
    names = checkpoint_names if isinstance(checkpoint_names, list) else None
    num_classes = int(checkpoint.get("num_classes", len(names) if names is not None else 0))
    if num_classes <= 0:
        raise ValueError("Checkpoint is missing a valid num_classes value.")

    model = VSTDet(
        num_classes=num_classes,
        variant=str(checkpoint.get("variant", "small")),
        backbone_name=str(checkpoint.get("backbone_name", "efficientnet_v2_s")),
        pretrained_backbone=False,
        neck_name=str(checkpoint.get("neck_name", "bifusion")),
        head_depth=int(checkpoint.get("head_depth", 2)),
        use_detail_branch=bool(checkpoint.get("use_detail_branch", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])  # type: ignore[arg-type]
    return model, names


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output) if args.output else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "evaluating",
        f"checkpoint={checkpoint_path}",
        f"split={args.split}",
        f"imgsz={args.imgsz}",
        f"conf_threshold={args.conf_threshold}",
        f"nms_iou={args.nms_iou}",
        f"max_det={args.max_det}",
    )
    print("device", f"name={format_device_name(device)}", f"type={device.type}")

    loader, dataset_names = build_loader(args, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, checkpoint_names = build_model_from_checkpoint(checkpoint, device)
    print("model", model_summary(model))

    if checkpoint_names is not None and checkpoint_names != dataset_names:
        print("warning", "checkpoint class names differ from dataset YAML; using dataset names for reporting")
    if model.num_classes != len(dataset_names):
        raise ValueError(
            f"Checkpoint expects {model.num_classes} classes but dataset defines {len(dataset_names)}."
        )

    report = evaluate_detection_metrics(
        model=model,
        loader=loader,
        device=device,
        num_classes=model.num_classes,
        class_names=dataset_names,
        conf_threshold=args.conf_threshold,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        show_progress=True,
        progress_desc=f"{args.split.title()} eval",
    )

    table = format_metrics_table(report)
    print("\nEvaluation report:")
    print(table)

    text_path = output_dir / f"eval_{args.split}_metrics.txt"
    json_path = output_dir / f"eval_{args.split}_metrics.json"
    text_path.write_text("Evaluation report:\n" + table + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_per_class_metrics(report, output_dir / f"eval_{args.split}_per_class_metrics.png")
    summary = report["summary"]
    append_results_summary(
        output_dir.parent / "results_summary.csv",
        {
            "kind": "eval",
            "run_name": output_dir.name,
            "output_dir": output_dir,
            "config": args.config,
            "data": args.data,
            "checkpoint": checkpoint_path,
            "split": args.split,
            "device": device.type,
            "variant": model.variant,
            "backbone": model.backbone_name,
            "neck": model.neck_name,
            "head_depth": model.head_depth,
            "detail_branch": model.use_detail_branch,
            "imgsz": args.imgsz,
            "batch_size": args.batch_size,
            "precision": summary["precision"],
            "recall": summary["recall"],
            "map50": summary["map50"],
            "map50_95": summary["map50_95"],
        },
    )
    print(f"\nSaved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
