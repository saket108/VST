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

from data.dataset import AugmentConfig, YoloDetectionDataset, collate_fn
from model.detector import VSTDet
from training.engine import append_history, save_checkpoint, train_one_epoch, validate
from training.losses import DetectionLoss
from utils.evaluator import format_metrics_table


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
    parser.add_argument(
        "--detail-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the high-resolution detail branch experimental model variant.",
    )
    parser.add_argument("--imgsz", type=int, default=896, help="Square training size.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=0.1,
        help="Multiplier applied to the backbone learning rate relative to --lr.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="runs/research_vstdet")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume training from a checkpoint path saved by this project.",
    )
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--copy-paste-mode", choices=["flip"], default="flip")
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--erasing", type=float, default=0.0)
    parser.add_argument(
        "--center-radius",
        type=float,
        default=1.5,
        help="Center-sampling radius used by the detector target assignment.",
    )
    parser.add_argument(
        "--topk-candidates",
        type=int,
        default=0,
        help="Add top-k center-prior positives per ground truth during target assignment.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
        help="Enable automatic mixed precision on CUDA for faster training.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.001,
        help="Confidence threshold used during validation decoding and metric computation.",
    )
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=300)
    parser.set_defaults(**config_defaults)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_augment_config(args: argparse.Namespace) -> AugmentConfig:
    return AugmentConfig(
        copy_paste=args.copy_paste,
        copy_paste_mode=args.copy_paste_mode,
        mosaic=args.mosaic,
        mixup=args.mixup,
        degrees=args.degrees,
        flipud=args.flipud,
        fliplr=args.fliplr,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        scale=args.scale,
        translate=args.translate,
        erasing=args.erasing,
    )


def build_optimizer(
    model: VSTDet,
    lr: float,
    backbone_lr_scale: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_params = list(model.backbone.parameters())
    backbone_param_ids = {id(parameter) for parameter in backbone_params}
    other_params = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_param_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_lr_scale},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, list[str]]:
    train_set = YoloDetectionDataset(
        args.data,
        split="train",
        image_size=args.imgsz,
        augment=True,
        augment_config=build_augment_config(args),
    )
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


def load_checkpoint(
    path: str | Path,
    model: VSTDet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_metric = float(checkpoint.get("best_metric", 0.0))
    return start_epoch, best_metric


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
        use_detail_branch=args.detail_branch,
    ).to(device)
    criterion = DetectionLoss(
        num_classes=len(names),
        strides=model.strides,
        center_radius=args.center_radius,
        topk_candidates=args.topk_candidates,
    )

    optimizer = build_optimizer(
        model=model,
        lr=args.lr,
        backbone_lr_scale=args.backbone_lr_scale,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    start_epoch = 1
    best_map = 0.0
    history_path = output_dir / "history.csv"
    train_start = time.time()
    last_val_report: dict[str, object] | None = None

    if args.resume:
        start_epoch, best_map = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    print(
        "training",
        f"variant={args.variant}",
        f"backbone={args.backbone}",
        f"pretrained_backbone={args.pretrained_backbone}",
        f"freeze_backbone_epochs={args.freeze_backbone_epochs}",
        f"detail_branch={args.detail_branch}",
        f"backbone_lr_scale={args.backbone_lr_scale}",
        f"center_radius={args.center_radius}",
        f"topk_candidates={args.topk_candidates}",
        f"mosaic={args.mosaic}",
        f"mixup={args.mixup}",
        f"copy_paste={args.copy_paste}",
        f"amp={amp_enabled}",
    )
    if args.resume:
        print(f"resuming from {args.resume} at epoch {start_epoch}")
    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs + 1):
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
            amp_enabled=amp_enabled,
        )
        current_lr = max(float(group["lr"]) for group in optimizer.param_groups)
        scheduler.step()

        row: dict[str, float | int] = {
            "epoch": epoch,
            "lr": current_lr,
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
                stage_label="Epoch val",
            )
            row.update(val_metrics)
            last_val_report = val_report

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
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_map,
            names,
        )

        append_history(history_path, row)
        _ = time.time() - start

    total_hours = (time.time() - train_start) / 3600.0
    print(f"\n{args.epochs} epochs completed in {total_hours:.3f} hours.")
    print(f"Results saved to {output_dir}")
    if last_val_report is not None:
        print("\nFinal checkpoint val:")
        print(format_metrics_table(last_val_report))


if __name__ == "__main__":
    main()
