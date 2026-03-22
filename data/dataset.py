from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    train: object
    val: object
    names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.names)


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

    return DatasetConfig(root=root, train=data["train"], val=data["val"], names=ordered)


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


def letterbox(
    image: Image.Image,
    boxes: torch.Tensor,
    size: int,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[torch.Tensor, torch.Tensor]:
    width, height = image.size
    scale = min(size / width, size / height)
    new_width = max(int(round(width * scale)), 1)
    new_height = max(int(round(height * scale)), 1)

    resized = image.resize((new_width, new_height), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    pad_left = (size - new_width) // 2
    pad_top = (size - new_height) // 2
    canvas.paste(resized, (pad_left, pad_top))

    boxes = boxes.clone()
    if boxes.numel() > 0:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_top

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


class YoloDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(
        self,
        yaml_path: str | Path,
        split: str,
        image_size: int,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.config = load_dataset_config(yaml_path)
        entry = getattr(self.config, split)
        self.image_paths = resolve_split_paths(self.config.root, entry)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.image_paths[index]
        label_path = image_to_label_path(image_path)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes, labels = load_yolo_labels(label_path, width, height)

        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
            if boxes.numel() > 0:
                x1 = width - boxes[:, 2]
                x2 = width - boxes[:, 0]
                boxes[:, 0] = x1
                boxes[:, 2] = x2

        if self.augment and random.random() < 0.2:
            image = TF.adjust_brightness(image, 0.8 + random.random() * 0.4)
            image = TF.adjust_contrast(image, 0.8 + random.random() * 0.4)
            image = TF.adjust_saturation(image, 0.8 + random.random() * 0.4)

        if self.augment:
            image_tensor, boxes, labels = random_affine_letterbox(
                image,
                boxes,
                labels,
                self.image_size,
            )
        else:
            image_tensor, boxes = letterbox(image, boxes, self.image_size)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.long),
            "orig_size": torch.tensor([height, width], dtype=torch.long),
            "image_size": torch.tensor([self.image_size, self.image_size], dtype=torch.long),
        }
        return image_tensor, target


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)
