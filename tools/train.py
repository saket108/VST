from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

from data.dataset import AugmentConfig, YoloDetectionDataset, collate_fn
from model.detector import VSTDet
from training import EpochCallbackState, TrainEndState, build_default_callbacks
from training.engine import train_one_epoch, validate
from training.losses import DetectionLoss
from utils.autobatch import estimate_autobatch_size
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
        "--neck",
        choices=["bifusion", "cafpn"],
        default="bifusion",
        help="Feature pyramid fusion module.",
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
    parser.add_argument(
        "--autobatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Probe a safe GPU batch size before building the dataloaders.",
    )
    parser.add_argument(
        "--autobatch-max",
        type=int,
        default=64,
        help="Highest candidate batch size to test when --autobatch is enabled.",
    )
    parser.add_argument("--head-depth", type=int, default=2)
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
        "--assigner",
        choices=["fcos", "atss"],
        default="fcos",
        help="Target assignment strategy used during training.",
    )
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
        "--atss-topk",
        type=int,
        default=9,
        help="Top-k candidates per level used by the ATSS assigner.",
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
    parser.add_argument(
        "--class-aware-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Oversample images containing selected classes using a weighted sampler.",
    )
    parser.add_argument(
        "--sample-classes",
        default="",
        help="Comma-separated class names or ids to boost when class-aware sampling is enabled.",
    )
    parser.add_argument(
        "--sample-boost-factor",
        type=float,
        default=4.0,
        help="Extra sampling weight added per target-class instance.",
    )
    parser.set_defaults(**config_defaults)
    return parser.parse_args()

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


def build_datasets(
    args: argparse.Namespace,
) -> tuple[YoloDetectionDataset, YoloDetectionDataset, list[str]]:
    train_set = YoloDetectionDataset(
        args.data,
        split="train",
        image_size=args.imgsz,
        augment=True,
        augment_config=build_augment_config(args),
    )
    val_set = YoloDetectionDataset(args.data, split="val", image_size=args.imgsz, augment=False)
    return train_set, val_set, train_set.config.names


def build_loaders(
    args: argparse.Namespace,
    train_set: YoloDetectionDataset,
    val_set: YoloDetectionDataset,
    train_sampler: WeightedRandomSampler | None = None,
) -> tuple[DataLoader, DataLoader]:
    pin_memory = resolve_device(args.device).type == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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
    return train_loader, val_loader


def resolve_sample_class_ids(names: list[str], spec: str) -> list[int]:
    resolved: list[int] = []
    if spec.strip():
        name_to_id = {name.lower(): index for index, name in enumerate(names)}
        for token in spec.split(","):
            item = token.strip()
            if not item:
                continue
            if item.isdigit():
                class_id = int(item)
                if not 0 <= class_id < len(names):
                    raise ValueError(f"sample class id {class_id} is out of range for {len(names)} classes.")
                resolved.append(class_id)
                continue
            lookup = name_to_id.get(item.lower())
            if lookup is None:
                raise ValueError(f"Unknown sample class '{item}'. Expected one of: {', '.join(names)}")
            resolved.append(lookup)
    else:
        default_targets = {"scratch", "missing-head"}
        resolved = [index for index, name in enumerate(names) if name in default_targets]
    return sorted(set(resolved))


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
    seed_everything(args.seed)

    device = resolve_device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set, names = build_datasets(args)
    model = VSTDet(
        num_classes=len(names),
        variant=args.variant,
        backbone_name=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
        neck_name=args.neck,
        head_depth=args.head_depth,
        use_detail_branch=args.detail_branch,
    ).to(device)
    criterion = DetectionLoss(
        num_classes=len(names),
        strides=model.strides,
        assigner=args.assigner,
        center_radius=args.center_radius,
        topk_candidates=args.topk_candidates,
        atss_topk=args.atss_topk,
    )
    amp_enabled = args.amp and device.type == "cuda"
    train_sampler: WeightedRandomSampler | None = None

    if args.autobatch:
        if device.type != "cuda":
            print("autobatch skipped: CUDA is required for GPU memory probing.")
        else:
            probe_trainable_backbone = args.freeze_backbone_epochs < args.epochs
            result = estimate_autobatch_size(
                model=model,
                criterion=criterion,
                dataset=train_set,
                collate=collate_fn,
                device=device,
                max_batch_size=args.autobatch_max,
                amp_enabled=amp_enabled,
                trainable_backbone=probe_trainable_backbone,
            )
            args.batch_size = result.batch_size
            attempts = ", ".join(
                f"{batch}:{'ok' if success else 'oom'}" for batch, success in result.tried
            )
            print(
                "autobatch",
                f"recommended_batch_size={args.batch_size}",
                f"trainable_backbone={probe_trainable_backbone}",
                f"reserved_memory={result.device_memory}",
            )
            print("autobatch attempts", attempts)

    if args.class_aware_sampling:
        sample_class_ids = resolve_sample_class_ids(names, args.sample_classes)
        if sample_class_ids:
            weights = train_set.build_sampling_weights(
                target_classes=sample_class_ids,
                boost_factor=args.sample_boost_factor,
            )
            train_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            boosted_images = int((weights > 1.0).sum().item())
            sample_class_names = ", ".join(names[class_id] for class_id in sample_class_ids)
            print(
                "class_aware_sampling",
                f"classes={sample_class_names}",
                f"boost_factor={args.sample_boost_factor}",
                f"boosted_images={boosted_images}/{len(weights)}",
            )
        else:
            print("class_aware_sampling skipped: no matching classes were resolved.")

    train_loader, val_loader = build_loaders(args, train_set, val_set, train_sampler=train_sampler)

    optimizer = build_optimizer(
        model=model,
        lr=args.lr,
        backbone_lr_scale=args.backbone_lr_scale,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    callbacks = build_default_callbacks(save_every=args.save_every)

    start_epoch = 1
    best_map = 0.0
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
        f"neck={args.neck}",
        f"pretrained_backbone={args.pretrained_backbone}",
        f"freeze_backbone_epochs={args.freeze_backbone_epochs}",
        f"head_depth={args.head_depth}",
        f"detail_branch={args.detail_branch}",
        f"backbone_lr_scale={args.backbone_lr_scale}",
        f"assigner={args.assigner}",
        f"center_radius={args.center_radius}",
        f"topk_candidates={args.topk_candidates}",
        f"atss_topk={args.atss_topk}",
        f"mosaic={args.mosaic}",
        f"mixup={args.mixup}",
        f"copy_paste={args.copy_paste}",
        f"class_aware_sampling={args.class_aware_sampling}",
        f"batch_size={args.batch_size}",
        f"amp={amp_enabled}",
    )
    print("device", f"name={format_device_name(device)}", f"type={device.type}")
    print("model", model_summary(model))
    if args.resume:
        print(f"resuming from {args.resume} at epoch {start_epoch}")
    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        model.set_backbone_trainable(epoch > args.freeze_backbone_epochs)
        print("\n      Epoch    GPU_mem   box_loss   cls_loss   ctr_loss  Instances       Size")
        current_val_report: dict[str, object] | None = None
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
            current_val_report = val_report

        callback_state = EpochCallbackState(
            epoch=epoch,
            epochs=args.epochs,
            output_dir=output_dir,
            row=row,
            val_report=current_val_report,
            best_metric=best_map,
            names=names,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        callbacks.on_epoch_end(callback_state)
        best_map = callback_state.best_metric
        _ = time.time() - start

    total_hours = (time.time() - train_start) / 3600.0
    callbacks.on_train_end(
        TrainEndState(
            epochs=args.epochs,
            output_dir=output_dir,
            total_hours=total_hours,
            last_val_report=last_val_report,
        )
    )


if __name__ == "__main__":
    main()
