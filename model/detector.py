from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import batched_nms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_V2_S_Weights,
    MobileNet_V3_Large_Weights,
    convnext_tiny,
    efficientnet_v2_s,
    mobilenet_v3_large,
)
from torchvision.models.feature_extraction import create_feature_extractor

from .blocks import ContextBridge, ConvBNAct, EdgeResidual, Scale, WeightedFeatureFusion
from utils.box_ops import distance_to_boxes
from utils.points import build_points


@dataclass(frozen=True)
class VariantConfig:
    channels: tuple[int, int, int, int]
    depths: tuple[int, int, int, int]
    head_channels: int


@dataclass(frozen=True)
class BackboneSpec:
    builder: Callable[..., nn.Module]
    default_weights: object
    return_nodes: dict[str, str]
    channels: tuple[int, int, int, int]


VARIANTS: dict[str, VariantConfig] = {
    "tiny": VariantConfig((64, 96, 160, 224), (2, 2, 4, 2), 96),
    "small": VariantConfig((64, 128, 192, 256), (2, 3, 5, 2), 128),
    "base": VariantConfig((96, 160, 256, 320), (3, 4, 6, 3), 160),
}

TORCHVISION_BACKBONES: dict[str, BackboneSpec] = {
    "mobilenet_v3_large": BackboneSpec(
        builder=mobilenet_v3_large,
        default_weights=MobileNet_V3_Large_Weights.DEFAULT,
        return_nodes={
            "features.3.add": "c2",
            "features.6.add": "c3",
            "features.12.add": "c4",
            "features.16": "c5",
        },
        channels=(24, 40, 112, 960),
    ),
    "efficientnet_v2_s": BackboneSpec(
        builder=efficientnet_v2_s,
        default_weights=EfficientNet_V2_S_Weights.DEFAULT,
        return_nodes={
            "features.2.3.add": "c2",
            "features.3.3.add": "c3",
            "features.5.8.add": "c4",
            "features.6.14.add": "c5",
        },
        channels=(48, 64, 160, 256),
    ),
    "convnext_tiny": BackboneSpec(
        builder=convnext_tiny,
        default_weights=ConvNeXt_Tiny_Weights.DEFAULT,
        return_nodes={
            "features.1.2.add": "c2",
            "features.3.2.add": "c3",
            "features.5.8.add": "c4",
            "features.7.2.add": "c5",
        },
        channels=(96, 192, 384, 768),
    ),
}


class VSTBackbone(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int], depths: tuple[int, ...]) -> None:
        super().__init__()
        c2, c3, c4, c5 = channels
        d2, d3, d4, d5 = depths

        self.stem = nn.Sequential(
            ConvBNAct(3, c2 // 2, stride=2),
            ConvBNAct(c2 // 2, c2 // 2, groups=c2 // 2),
            ConvBNAct(c2 // 2, c2 // 2, kernel_size=1),
        )
        self.stage2 = self._make_stage(c2 // 2, c2, d2)
        self.stage3 = self._make_stage(c2, c3, d3)
        self.stage4 = self._make_stage(c3, c4, d4)
        self.stage5 = self._make_stage(c4, c5, d5)

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, depth: int) -> nn.Sequential:
        blocks: list[nn.Module] = [EdgeResidual(in_channels, out_channels, stride=2)]
        for _ in range(depth - 1):
            blocks.append(EdgeResidual(out_channels, out_channels))
        blocks.append(ContextBridge(out_channels))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c2, c3, c4, c5


class TorchvisionBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool = True) -> None:
        super().__init__()
        if name not in TORCHVISION_BACKBONES:
            raise ValueError(
                f"Unknown backbone '{name}'. Expected one of {sorted(TORCHVISION_BACKBONES)}."
            )

        spec = TORCHVISION_BACKBONES[name]
        weights = spec.default_weights if pretrained else None
        backbone = spec.builder(weights=weights)
        self.extractor = create_feature_extractor(backbone, return_nodes=spec.return_nodes)
        self.channels = spec.channels
        self.name = name
        self.pretrained = pretrained

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if weights is not None:
            transforms = weights.transforms()
            mean = list(transforms.mean)
            std = list(transforms.std)

        self.register_buffer(
            "pixel_mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = (x - self.pixel_mean) / self.pixel_std
        features = self.extractor(x)
        return features["c2"], features["c3"], features["c4"], features["c5"]


class BiFusionNeck(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int, int], out_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [ConvBNAct(ch, out_channels, kernel_size=1) for ch in in_channels]
        )
        self.downsample = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels, stride=2) for _ in range(3)]
        )

        self.topdown_p4 = WeightedFeatureFusion(out_channels, inputs=2)
        self.topdown_p3 = WeightedFeatureFusion(out_channels, inputs=2)
        self.topdown_p2 = WeightedFeatureFusion(out_channels, inputs=2)

        self.bottomup_p3 = WeightedFeatureFusion(out_channels, inputs=2)
        self.bottomup_p4 = WeightedFeatureFusion(out_channels, inputs=2)
        self.bottomup_p5 = WeightedFeatureFusion(out_channels, inputs=2)

        self.refine = nn.ModuleList([ContextBridge(out_channels) for _ in range(4)])

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        p2, p3, p4, p5 = [layer(x) for layer, x in zip(self.lateral, features)]

        p4_td = self.topdown_p4([p4, F.interpolate(p5, size=p4.shape[-2:], mode="nearest")])
        p3_td = self.topdown_p3([p3, F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")])
        p2_td = self.topdown_p2([p2, F.interpolate(p3_td, size=p2.shape[-2:], mode="nearest")])

        p3_out = self.bottomup_p3([p3_td, self.downsample[0](p2_td)])
        p4_out = self.bottomup_p4([p4_td, self.downsample[1](p3_out)])
        p5_out = self.bottomup_p5([p5, self.downsample[2](p4_out)])

        outputs = [p2_td, p3_out, p4_out, p5_out]
        return tuple(block(feature) for block, feature in zip(self.refine, outputs))


class HeadTower(nn.Sequential):
    def __init__(self, channels: int, depth: int = 2) -> None:
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(ConvBNAct(channels, channels, groups=channels))
            layers.append(ConvBNAct(channels, channels, kernel_size=1))
        super().__init__(*layers)


class DetectionHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, levels: int) -> None:
        super().__init__()
        self.cls_tower = HeadTower(channels)
        self.reg_tower = HeadTower(channels)
        self.cls_pred = nn.Conv2d(channels, num_classes, 3, padding=1)
        self.box_pred = nn.Conv2d(channels, 4, 3, padding=1)
        self.center_pred = nn.Conv2d(channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale() for _ in range(levels)])
        self._init_biases()

    def _init_biases(self, prior_prob: float = 0.01) -> None:
        cls_bias = math.log(prior_prob / (1.0 - prior_prob))
        nn.init.constant_(self.cls_pred.bias, cls_bias)
        nn.init.constant_(self.center_pred.bias, 0.0)
        nn.init.constant_(self.box_pred.bias, 1.0)

    def forward(self, features: tuple[torch.Tensor, ...]) -> dict[str, list[torch.Tensor]]:
        outputs = {"cls": [], "box": [], "center": []}
        for feature, scale in zip(features, self.scales):
            cls_feat = self.cls_tower(feature)
            reg_feat = self.reg_tower(feature)
            outputs["cls"].append(self.cls_pred(cls_feat))
            outputs["center"].append(self.center_pred(reg_feat))
            # Keep regression in float32 even under AMP. Large positive logits can overflow
            # in float16 before softplus, which then corrupts GIoU with non-finite boxes.
            with torch.autocast(device_type=feature.device.type, enabled=False):
                reg_logits = self.box_pred(reg_feat.float())
                reg_distances = F.softplus(scale(reg_logits)).clamp(max=1e4)
            outputs["box"].append(reg_distances)
        return outputs


class VSTDet(nn.Module):
    strides = (4, 8, 16, 32)

    def __init__(
        self,
        num_classes: int,
        variant: str = "small",
        backbone_name: str = "efficientnet_v2_s",
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Expected one of {sorted(VARIANTS)}.")

        cfg = VARIANTS[variant]
        self.num_classes = num_classes
        self.variant = variant
        self.backbone_name = backbone_name
        self.pretrained_backbone = pretrained_backbone

        if backbone_name == "custom":
            self.backbone = VSTBackbone(cfg.channels, cfg.depths)
            neck_in_channels = cfg.channels
        else:
            self.backbone = TorchvisionBackbone(backbone_name, pretrained=pretrained_backbone)
            neck_in_channels = self.backbone.channels

        self.neck = BiFusionNeck(neck_in_channels, cfg.head_channels)
        self.head = DetectionHead(cfg.head_channels, num_classes, levels=len(self.strides))

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def forward(self, images: torch.Tensor) -> dict[str, list[torch.Tensor] | tuple[int, ...]]:
        features = self.backbone(images)
        pyramid = self.neck(features)
        outputs = self.head(pyramid)
        outputs["strides"] = self.strides
        return outputs

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        conf_threshold: float = 0.05,
        nms_iou: float = 0.6,
        max_det: int = 300,
    ) -> list[dict[str, torch.Tensor]]:
        was_training = self.training
        self.eval()
        outputs = self(images)
        predictions = decode_predictions(
            outputs,
            image_size=images.shape[-2:],
            conf_threshold=conf_threshold,
            nms_iou=nms_iou,
            max_det=max_det,
        )
        if was_training:
            self.train()
        return predictions


def decode_predictions(
    outputs: dict[str, list[torch.Tensor] | tuple[int, ...]],
    image_size: tuple[int, int],
    conf_threshold: float,
    nms_iou: float,
    max_det: int,
) -> list[dict[str, torch.Tensor]]:
    cls_levels = outputs["cls"]  # type: ignore[index]
    box_levels = outputs["box"]  # type: ignore[index]
    center_levels = outputs["center"]  # type: ignore[index]
    strides = outputs["strides"]  # type: ignore[index]

    batch_size = cls_levels[0].shape[0]
    height, width = image_size
    results: list[dict[str, torch.Tensor]] = []

    for batch_index in range(batch_size):
        image_boxes: list[torch.Tensor] = []
        image_scores: list[torch.Tensor] = []
        image_labels: list[torch.Tensor] = []

        for cls_map, box_map, center_map, stride in zip(
            cls_levels, box_levels, center_levels, strides
        ):
            _, num_classes, feat_h, feat_w = cls_map.shape
            points = build_points(feat_h, feat_w, stride, cls_map.device, cls_map.dtype)

            cls_scores = cls_map[batch_index].permute(1, 2, 0).reshape(-1, num_classes).sigmoid()
            center_scores = center_map[batch_index].permute(1, 2, 0).reshape(-1).sigmoid()
            box_distances = (
                box_map[batch_index].permute(1, 2, 0).reshape(-1, 4) * stride
            )

            scores, labels = cls_scores.max(dim=1)
            scores = torch.sqrt(scores * center_scores)
            keep = scores > conf_threshold
            if not keep.any():
                continue

            boxes = distance_to_boxes(points[keep], box_distances[keep])
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)

            image_boxes.append(boxes)
            image_scores.append(scores[keep])
            image_labels.append(labels[keep])

        if not image_boxes:
            device = cls_levels[0].device
            results.append(
                {
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros((0,), device=device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=device),
                }
            )
            continue

        boxes = torch.cat(image_boxes, dim=0)
        scores = torch.cat(image_scores, dim=0)
        labels = torch.cat(image_labels, dim=0)

        keep = batched_nms(boxes, scores, labels, nms_iou)
        keep = keep[:max_det]
        results.append(
            {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
        )

    return results
