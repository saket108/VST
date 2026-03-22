from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
import yaml

from data.dataset import YoloDetectionDataset, collate_fn
from model.detector import VSTDet
from training.engine import append_history, save_checkpoint, train_one_epoch, validate
from training.losses import DetectionLoss
from utils.evaluator import format_metrics_table, format_summary_row


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
        description="Train VSTDet with a research-style project layout.",
    )
    parser.add_argument("--data", required=True, help="Path to dataset YAML in YOLO format.")
    parser.add_argument("--variant", choices=["tiny", "small", "base"], default="small")
    parser.add_argument(
        "--backbone",
        choices=["custom", "mobilenet_v3_large", "efficientnet_v2_s", "convnext_tiny"],
        default="efficientnet_v2_s",
        help="Feature extractor used before the custom neck and head.",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet-pretrained torchvision weights when available.",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Freeze backbone parameters for the first N epochs.",
    )
    parser.add_argument("--imgsz", type=int, default=896, help="Square training size.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="runs/research_vstdet")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=300)
    parser.set_defaults(**config_defaults)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, list[str]]:
    train_set = YoloDetectionDataset(args.data, split="train", image_size=args.imgsz, augment=True)
    val_set = YoloDetectionDataset(args.data, split="val", image_size=args.imgsz, augment=False)
    pin_memory = torch.device(args.device).type == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader, train_set.config.names


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, names = build_loaders(args)
    model = VSTDet(
        num_classes=len(names),
        variant=args.variant,
        backbone_name=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)
    criterion = DetectionLoss(num_classes=len(names), strides=model.strides)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_map = 0.0
    history_path = output_dir / "history.csv"
    train_start = time.time()
    last_val_report: dict[str, object] | None = None

    print(
        "training",
        f"variant={args.variant}",
        f"backbone={args.backbone}",
        f"pretrained_backbone={args.pretrained_backbone}",
        f"freeze_backbone_epochs={args.freeze_backbone_epochs}",
    )
    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.set_backbone_trainable(epoch > args.freeze_backbone_epochs)
        print("\n      Epoch    GPU_mem   box_loss   cls_loss   ctr_loss  Instances       Size")
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            clip_grad=args.clip_grad,
            epoch=epoch,
            epochs=args.epochs,
        )
        scheduler.step()

        row: dict[str, float | int] = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **train_metrics,
        }

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics, val_report = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                class_names=names,
                conf_threshold=args.conf_threshold,
                nms_iou=args.nms_iou,
                max_det=args.max_det,
            )
            row.update(val_metrics)
            last_val_report = val_report
            print("                 Class     Images  Instances  Precision     Recall      mAP50  mAP50-95")
            print(format_summary_row(val_report))

            if val_metrics["map50_95"] >= best_map:
                best_map = val_metrics["map50_95"]
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_map,
                    names,
                )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_map,
                names,
            )

        append_history(history_path, row)
        epoch_time = time.time() - start
        summary = " ".join(
            f"{key}={value:.4f}" for key, value in row.items() if isinstance(value, float)
        )
        print(f"epoch {epoch:03d} {summary} time={epoch_time:.1f}s")

    total_hours = (time.time() - train_start) / 3600.0
    print(f"\n{args.epochs} epochs completed in {total_hours:.3f} hours.")
    print(f"Results saved to {output_dir}")
    if last_val_report is not None:
        print("\nFinal validation report:")
        print(format_metrics_table(last_val_report))


if __name__ == "__main__":
    main()
