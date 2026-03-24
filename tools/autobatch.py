from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from data.dataset import AugmentConfig, YoloDetectionDataset, collate_fn
from model.detector import VSTDet
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
        description="Probe a safe training batch size for VSTDet on the current device.",
    )
    parser.add_argument("--data", required=True, help="Path to dataset YAML in YOLO format.")
    parser.add_argument("--variant", choices=["tiny", "small", "base"], default="small")
    parser.add_argument(
        "--backbone",
        choices=["custom", "mobilenet_v3_large", "efficientnet_v2_s", "convnext_tiny"],
        default="efficientnet_v2_s",
    )
    parser.add_argument("--neck", choices=["bifusion", "cafpn", "cafpn_lite", "cafpn_p2"], default="bifusion")
    parser.add_argument(
        "--pretrained-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument(
        "--detail-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--head-depth", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
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
    parser.add_argument("--assigner", choices=["fcos", "atss"], default="fcos")
    parser.add_argument("--center-radius", type=float, default=1.5)
    parser.add_argument("--topk-candidates", type=int, default=0)
    parser.add_argument("--atss-topk", type=int, default=9)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = resolve_device(args.device)
    amp_enabled = args.amp and device.type == "cuda"

    dataset = YoloDetectionDataset(
        args.data,
        split="train",
        image_size=args.imgsz,
        augment=True,
        augment_config=build_augment_config(args),
    )
    model = VSTDet(
        num_classes=dataset.config.num_classes,
        variant=args.variant,
        backbone_name=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
        neck_name=args.neck,
        head_depth=args.head_depth,
        use_detail_branch=args.detail_branch,
    ).to(device)
    criterion = DetectionLoss(
        num_classes=dataset.config.num_classes,
        strides=model.strides,
        assigner=args.assigner,
        center_radius=args.center_radius,
        topk_candidates=args.topk_candidates,
        atss_topk=args.atss_topk,
    )

    print(
        "autobatch",
        f"variant={args.variant}",
        f"backbone={args.backbone}",
        f"neck={args.neck}",
        f"imgsz={args.imgsz}",
        f"max_batch_size={args.max_batch_size}",
        f"amp={amp_enabled}",
    )
    print("device", f"name={format_device_name(device)}", f"type={device.type}")
    print("model", model_summary(model))

    if device.type != "cuda":
        print("autobatch skipped: CUDA is required for GPU memory probing.")
        return

    result = estimate_autobatch_size(
        model=model,
        criterion=criterion,
        dataset=dataset,
        collate=collate_fn,
        device=device,
        max_batch_size=args.max_batch_size,
        amp_enabled=amp_enabled,
        trainable_backbone=True,
    )
    attempts = ", ".join(f"{batch}:{'ok' if success else 'oom'}" for batch, success in result.tried)
    print(
        "autobatch result",
        f"recommended_batch_size={result.batch_size}",
        f"trainable_backbone={result.trainable_backbone}",
        f"reserved_memory={result.device_memory}",
    )
    print("attempts", attempts)


if __name__ == "__main__":
    main()
