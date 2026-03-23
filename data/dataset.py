from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_FILL = (114, 114, 114)


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    train: object
    val: object
    test: object | None
    names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class AugmentConfig:
    copy_paste: float = 0.0
    copy_paste_mode: str = "flip"
    mosaic: float = 0.0
    mixup: float = 0.0
    degrees: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    scale: float = 0.35
    translate: float = 0.1
    erasing: float = 0.0


def load_dataset_config(path: str | Path) -> DatasetConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    root = Path(data.get("path", config_path.parent)).expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()

    names = data.get("names")
    if isinstance(names, dict):
        ordered = [name for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
    elif isinstance(names, list):
        ordered = [str(name) for name in names]
    else:
        count = int(data["nc"])
        ordered = [f"class_{index}" for index in range(count)]

    return DatasetConfig(
        root=root,
        train=data["train"],
        val=data["val"],
        test=data.get("test"),
        names=ordered,
    )


def resolve_split_paths(root: Path, entry: object) -> list[Path]:
    if isinstance(entry, list):
        paths: list[Path] = []
        for item in entry:
            paths.extend(resolve_split_paths(root, item))
        return sorted(set(paths))

    split_path = Path(str(entry))
    if not split_path.is_absolute():
        split_path = (root / split_path).resolve()

    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        items: list[Path] = []
        with split_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                path = Path(line)
                if not path.is_absolute():
                    path = (root / path).resolve()
                items.append(path)
        return items

    if split_path.is_file():
        return [split_path]

    if split_path.is_dir():
        return sorted(
            path
            for path in split_path.rglob("*")
            if path.suffix.lower() in IMAGE_SUFFIXES
        )

    raise FileNotFoundError(f"Could not resolve split path: {entry}")


def image_to_label_path(image_path: Path) -> Path:
    parts = image_path.parts
    if "images" in parts:
        index = parts.index("images")
        return Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def load_yolo_labels(label_path: Path, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not label_path.exists():
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

    boxes = []
    labels = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = parts
            cls_id = int(float(cls))
            cx = float(cx) * width
            cy = float(cy) * height
            bw = float(bw) * width
            bh = float(bh) * height
            x1 = cx - bw * 0.5
            y1 = cy - bh * 0.5
            x2 = cx + bw * 0.5
            y2 = cy + bh * 0.5
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_id)

    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def load_label_class_ids(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []

    class_ids: list[int] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                class_ids.append(int(float(parts[0])))
            except ValueError:
                continue
    return class_ids


def resize_and_pad(
    image: Image.Image,
    boxes: torch.Tensor,
    canvas_width: int,
    canvas_height: int,
    fill: tuple[int, int, int] = DEFAULT_FILL,
) -> tuple[Image.Image, torch.Tensor]:
    width, height = image.size
    scale = min(canvas_width / width, canvas_height / height)
    new_width = max(int(round(width * scale)), 1)
    new_height = max(int(round(height * scale)), 1)

    resized = image.resize((new_width, new_height), Image.BILINEAR)
    canvas = Image.new("RGB", (canvas_width, canvas_height), fill)
    pad_left = (canvas_width - new_width) // 2
    pad_top = (canvas_height - new_height) // 2
    canvas.paste(resized, (pad_left, pad_top))

    boxes = boxes.clone()
    if boxes.numel() > 0:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_top
    return canvas, boxes


def letterbox(
    image: Image.Image,
    boxes: torch.Tensor,
    size: int,
    fill: tuple[int, int, int] = DEFAULT_FILL,
) -> tuple[torch.Tensor, torch.Tensor]:
    canvas, boxes = resize_and_pad(image, boxes, size, size, fill=fill)
    image_tensor = TF.pil_to_tensor(canvas).float() / 255.0
    return image_tensor, boxes


def clip_and_filter_boxes(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    size: int,
    min_size: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes, labels

    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp_(0, size)
    boxes[:, 1::2] = boxes[:, 1::2].clamp_(0, size)
    wh = boxes[:, 2:] - boxes[:, :2]
    keep = (wh[:, 0] >= min_size) & (wh[:, 1] >= min_size)
    return boxes[keep], labels[keep]


def random_affine_letterbox(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    size: int,
    fill: tuple[int, int, int] = (114, 114, 114),
    scale_range: tuple[float, float] = (0.75, 1.35),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width, height = image.size
    base_scale = min(size / width, size / height)
    scale = base_scale * random.uniform(*scale_range)
    new_width = max(int(round(width * scale)), 1)
    new_height = max(int(round(height * scale)), 1)

    resized = image.resize((new_width, new_height), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)

    if new_width <= size:
        pad_left = random.randint(0, size - new_width)
    else:
        pad_left = -random.randint(0, new_width - size)

    if new_height <= size:
        pad_top = random.randint(0, size - new_height)
    else:
        pad_top = -random.randint(0, new_height - size)

    canvas.paste(resized, (pad_left, pad_top))

    boxes = boxes.clone()
    if boxes.numel() > 0:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_top
        boxes, labels = clip_and_filter_boxes(boxes, labels, size)

    image_tensor = TF.pil_to_tensor(canvas).float() / 255.0
    return image_tensor, boxes, labels


def apply_hsv_augment(image: Image.Image, cfg: AugmentConfig) -> Image.Image:
    if cfg.hsv_h <= 0.0 and cfg.hsv_s <= 0.0 and cfg.hsv_v <= 0.0:
        return image

    hsv = np.array(image.convert("HSV"), dtype=np.uint8)
    if cfg.hsv_h > 0.0:
        hue_shift = int(random.uniform(-cfg.hsv_h, cfg.hsv_h) * 255)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue_shift) % 256
    if cfg.hsv_s > 0.0:
        sat_gain = 1.0 + random.uniform(-cfg.hsv_s, cfg.hsv_s)
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * sat_gain, 0, 255)
    if cfg.hsv_v > 0.0:
        val_gain = 1.0 + random.uniform(-cfg.hsv_v, cfg.hsv_v)
        hsv[..., 2] = np.clip(hsv[..., 2].astype(np.float32) * val_gain, 0, 255)
    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")


def flip_boxes_horizontal(boxes: torch.Tensor, width: int) -> torch.Tensor:
    flipped = boxes.clone()
    flipped[:, 0] = width - boxes[:, 2]
    flipped[:, 2] = width - boxes[:, 0]
    return flipped


def flip_boxes_vertical(boxes: torch.Tensor, height: int) -> torch.Tensor:
    flipped = boxes.clone()
    flipped[:, 1] = height - boxes[:, 3]
    flipped[:, 3] = height - boxes[:, 1]
    return flipped


def apply_random_flips(
    image: Image.Image,
    boxes: torch.Tensor,
    cfg: AugmentConfig,
) -> tuple[Image.Image, torch.Tensor]:
    width, height = image.size
    if cfg.fliplr > 0.0 and random.random() < cfg.fliplr:
        image = ImageOps.mirror(image)
        if boxes.numel() > 0:
            boxes = flip_boxes_horizontal(boxes, width)
    if cfg.flipud > 0.0 and random.random() < cfg.flipud:
        image = ImageOps.flip(image)
        if boxes.numel() > 0:
            boxes = flip_boxes_vertical(boxes, height)
    return image, boxes


def apply_random_erasing(
    image_tensor: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    if probability <= 0.0 or random.random() >= probability:
        return image_tensor

    _, height, width = image_tensor.shape
    erase_area = random.uniform(0.02, 0.12) * height * width
    aspect = random.uniform(0.3, 3.3)
    erase_h = int(round(math.sqrt(erase_area / aspect)))
    erase_w = int(round(math.sqrt(erase_area * aspect)))
    erase_h = max(1, min(erase_h, height))
    erase_w = max(1, min(erase_w, width))
    top = random.randint(0, height - erase_h)
    left = random.randint(0, width - erase_w)
    image_tensor[:, top : top + erase_h, left : left + erase_w] = torch.rand(
        (3, erase_h, erase_w),
        dtype=image_tensor.dtype,
    )
    return image_tensor


def apply_copy_paste(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    cfg: AugmentConfig,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    if cfg.copy_paste <= 0.0 or cfg.copy_paste_mode != "flip":
        return image, boxes, labels
    if boxes.numel() == 0 or random.random() >= cfg.copy_paste:
        return image, boxes, labels

    pasted = image.copy()
    new_boxes: list[list[float]] = []
    new_labels: list[int] = []
    max_samples = max(1, min(len(boxes), len(boxes) // 2 or 1))
    for box_index in torch.randperm(len(boxes))[:max_samples].tolist():
        x1, y1, x2, y2 = boxes[box_index].round().to(dtype=torch.long).tolist()
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        dest_x1 = image.width - x2
        dest_x2 = image.width - x1
        if dest_x2 - dest_x1 < 4:
            continue
        patch = ImageOps.mirror(image.crop((x1, y1, x2, y2)))
        pasted.paste(patch, (dest_x1, y1))
        new_boxes.append([float(dest_x1), float(y1), float(dest_x2), float(y2)])
        new_labels.append(int(labels[box_index].item()))

    if not new_boxes:
        return pasted, boxes, labels

    box_tensor = torch.tensor(new_boxes, dtype=boxes.dtype)
    label_tensor = torch.tensor(new_labels, dtype=labels.dtype)
    boxes = torch.cat([boxes, box_tensor], dim=0)
    labels = torch.cat([labels, label_tensor], dim=0)
    boxes, labels = clip_and_filter_boxes(boxes, labels, image.width)
    return pasted, boxes, labels


def apply_random_affine(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    size: int,
    cfg: AugmentConfig,
    fill: tuple[int, int, int] = DEFAULT_FILL,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    if cfg.degrees <= 0.0 and cfg.scale <= 0.0 and cfg.translate <= 0.0:
        return image, boxes, labels

    angle = random.uniform(-cfg.degrees, cfg.degrees)
    scale = random.uniform(max(0.25, 1.0 - cfg.scale), 1.0 + cfg.scale)
    tx = random.uniform(-cfg.translate, cfg.translate) * size
    ty = random.uniform(-cfg.translate, cfg.translate) * size
    center = size * 0.5
    radians = math.radians(angle)
    cos_a = math.cos(radians) * scale
    sin_a = math.sin(radians) * scale

    transform = np.array(
        [
            [cos_a, -sin_a, center + tx - cos_a * center + sin_a * center],
            [sin_a, cos_a, center + ty - sin_a * center - cos_a * center],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    inverse = np.linalg.inv(transform)
    image = image.transform(
        (size, size),
        Image.AFFINE,
        data=(
            float(inverse[0, 0]),
            float(inverse[0, 1]),
            float(inverse[0, 2]),
            float(inverse[1, 0]),
            float(inverse[1, 1]),
            float(inverse[1, 2]),
        ),
        resample=Image.BILINEAR,
        fillcolor=fill,
    )

    if boxes.numel() == 0:
        return image, boxes, labels

    corners = torch.stack(
        [
            boxes[:, [0, 1]],
            boxes[:, [2, 1]],
            boxes[:, [2, 3]],
            boxes[:, [0, 3]],
        ],
        dim=1,
    )
    ones = torch.ones((corners.shape[0], 4, 1), dtype=boxes.dtype)
    homogenous = torch.cat([corners, ones], dim=2).numpy()
    warped = homogenous @ transform.T
    warped = torch.from_numpy(warped[:, :, :2]).to(dtype=boxes.dtype)

    min_xy = warped.min(dim=1).values
    max_xy = warped.max(dim=1).values
    boxes = torch.cat([min_xy, max_xy], dim=1)
    boxes, labels = clip_and_filter_boxes(boxes, labels, size)
    return image, boxes, labels


class YoloDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(
        self,
        yaml_path: str | Path,
        split: str,
        image_size: int,
        augment: bool = False,
        augment_config: AugmentConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = load_dataset_config(yaml_path)
        entry = getattr(self.config, split, None)
        if entry is None:
            raise ValueError(f"Split '{split}' is not configured in dataset YAML: {yaml_path}")
        self.image_paths = resolve_split_paths(self.config.root, entry)
        self.label_paths = [image_to_label_path(path) for path in self.image_paths]
        self.image_size = image_size
        self.augment = augment
        self.augment_config = augment_config or AugmentConfig()
        self._label_class_counts: list[Counter[int]] | None = None

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_label_class_counts(self) -> list[Counter[int]]:
        if self._label_class_counts is None:
            self._label_class_counts = [
                Counter(load_label_class_ids(label_path))
                for label_path in self.label_paths
            ]
        return self._label_class_counts

    def build_sampling_weights(
        self,
        target_classes: list[int],
        boost_factor: float = 4.0,
    ) -> torch.Tensor:
        if not target_classes:
            return torch.ones(len(self.image_paths), dtype=torch.double)

        target_class_set = set(target_classes)
        weights: list[float] = []
        for class_counts in self.get_label_class_counts():
            target_instances = sum(
                count for class_id, count in class_counts.items() if class_id in target_class_set
            )
            weight = 1.0 + boost_factor * float(target_instances)
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.double)

    def _load_image_target(
        self,
        index: int,
    ) -> tuple[Image.Image, torch.Tensor, torch.Tensor, tuple[int, int]]:
        image_path = self.image_paths[index]
        label_path = image_to_label_path(image_path)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes, labels = load_yolo_labels(label_path, width, height)
        return image, boxes, labels, (height, width)

    def _load_mosaic(
        self,
        index: int,
    ) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
        tile = self.image_size // 2
        mosaic = Image.new("RGB", (self.image_size, self.image_size), DEFAULT_FILL)
        all_boxes: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        indices = [index] + random.choices(range(len(self.image_paths)), k=3)
        placements = [(0, 0), (tile, 0), (0, tile), (tile, tile)]

        for sample_index, (offset_x, offset_y) in zip(indices, placements):
            image, boxes, labels, _ = self._load_image_target(sample_index)
            tile_image, tile_boxes = resize_and_pad(image, boxes, tile, tile)
            mosaic.paste(tile_image, (offset_x, offset_y))
            if tile_boxes.numel() > 0:
                tile_boxes = tile_boxes.clone()
                tile_boxes[:, [0, 2]] += offset_x
                tile_boxes[:, [1, 3]] += offset_y
                all_boxes.append(tile_boxes)
                all_labels.append(labels)

        if not all_boxes:
            return mosaic, torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        return mosaic, torch.cat(all_boxes, dim=0), torch.cat(all_labels, dim=0)

    def _build_training_sample(
        self,
        index: int,
        allow_mixup: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.augment_config

        if cfg.mosaic > 0.0 and random.random() < cfg.mosaic:
            image, boxes, labels = self._load_mosaic(index)
        else:
            raw_image, raw_boxes, raw_labels, _ = self._load_image_target(index)
            image, boxes = resize_and_pad(raw_image, raw_boxes, self.image_size, self.image_size)
            labels = raw_labels

        image, boxes, labels = apply_copy_paste(image, boxes, labels, cfg)
        image, boxes, labels = apply_random_affine(image, boxes, labels, self.image_size, cfg)
        image, boxes = apply_random_flips(image, boxes, cfg)
        image = apply_hsv_augment(image, cfg)

        image_tensor = TF.pil_to_tensor(image).float() / 255.0

        if allow_mixup and cfg.mixup > 0.0 and random.random() < cfg.mixup:
            mix_index = random.randrange(len(self.image_paths))
            mix_image, mix_boxes, mix_labels = self._build_training_sample(
                mix_index,
                allow_mixup=False,
            )
            mix_ratio = float(np.random.beta(32.0, 32.0))
            image_tensor = image_tensor * mix_ratio + mix_image * (1.0 - mix_ratio)
            if mix_boxes.numel() > 0:
                boxes = torch.cat([boxes, mix_boxes], dim=0)
                labels = torch.cat([labels, mix_labels], dim=0)

        image_tensor = apply_random_erasing(image_tensor, cfg.erasing)
        boxes, labels = clip_and_filter_boxes(boxes, labels, self.image_size)
        return image_tensor, boxes, labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.augment:
            image_tensor, boxes, labels = self._build_training_sample(index)
            target_orig_size = torch.tensor([self.image_size, self.image_size], dtype=torch.long)
        else:
            image, boxes, labels, orig_size = self._load_image_target(index)
            height, width = orig_size
            image_tensor, boxes = letterbox(image, boxes, self.image_size)
            target_orig_size = torch.tensor([height, width], dtype=torch.long)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.long),
            "orig_size": target_orig_size,
            "image_size": torch.tensor([self.image_size, self.image_size], dtype=torch.long),
        }
        return image_tensor, target


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)
