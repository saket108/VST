from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - graceful fallback for stale environments
    matplotlib = None
    plt = None


def _load_history(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _series(rows: list[dict[str, str]], key: str) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        epoch_text = row.get("epoch", "").strip()
        value_text = row.get(key, "").strip()
        if not epoch_text or not value_text:
            continue
        try:
            x_values.append(float(epoch_text))
            y_values.append(float(value_text))
        except ValueError:
            continue
    return x_values, y_values


def _series_with_fallback(
    rows: list[dict[str, str]],
    key: str,
    fallback: str | None = None,
) -> tuple[list[float], list[float]]:
    xs, ys = _series(rows, key)
    if xs or fallback is None:
        return xs, ys
    return _series(rows, fallback)


def plot_history(history_path: str | Path, output_path: str | Path) -> Path | None:
    if plt is None:
        return None
    history_path = Path(history_path)
    rows = _load_history(history_path)
    if not rows:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    loss_axis = axes[0]
    for key, label, color in (
        ("total", "train total", "#1f77b4"),
        ("cls", "train cls", "#ff7f0e"),
        ("box", "train box", "#2ca02c"),
        ("quality", "train quality", "#d62728"),
    ):
        xs, ys = _series_with_fallback(rows, key, "center" if key == "quality" else None)
        if xs:
            loss_axis.plot(xs, ys, label=label, color=color, linewidth=2)
    loss_axis.set_title("Train Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.grid(alpha=0.3)
    if loss_axis.lines:
        loss_axis.legend()

    val_loss_axis = axes[1]
    for key, label, color in (
        ("val_total", "val total", "#1f77b4"),
        ("val_cls", "val cls", "#ff7f0e"),
        ("val_box", "val box", "#2ca02c"),
        ("val_quality", "val quality", "#d62728"),
    ):
        xs, ys = _series_with_fallback(
            rows,
            key,
            "val_center" if key == "val_quality" else None,
        )
        if xs:
            val_loss_axis.plot(xs, ys, label=label, color=color, linewidth=2)
    val_loss_axis.set_title("Validation Loss")
    val_loss_axis.set_xlabel("Epoch")
    val_loss_axis.grid(alpha=0.3)
    if val_loss_axis.lines:
        val_loss_axis.legend()

    metric_axis = axes[2]
    for key, label, color in (
        ("precision", "precision", "#9467bd"),
        ("recall", "recall", "#8c564b"),
        ("map50", "mAP50", "#17becf"),
        ("map50_95", "mAP50-95", "#e377c2"),
    ):
        xs, ys = _series(rows, key)
        if xs:
            metric_axis.plot(xs, ys, label=label, color=color, linewidth=2)
    metric_axis.set_title("Validation Metrics")
    metric_axis.set_xlabel("Epoch")
    metric_axis.set_ylim(bottom=0.0)
    metric_axis.grid(alpha=0.3)
    if metric_axis.lines:
        metric_axis.legend()

    lr_axis = axes[3]
    xs, ys = _series(rows, "lr")
    if xs:
        lr_axis.plot(xs, ys, label="lr", color="#7f7f7f", linewidth=2)
    lr_axis.set_title("Learning Rate")
    lr_axis.set_xlabel("Epoch")
    lr_axis.grid(alpha=0.3)
    if lr_axis.lines:
        lr_axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_per_class_metrics(report: dict[str, Any], output_path: str | Path) -> Path | None:
    if plt is None:
        return None
    per_class = report.get("per_class")
    if not isinstance(per_class, list) or not per_class:
        return None

    rows = [row for row in per_class if int(row.get("instances", 0)) > 0]
    if not rows:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_names = [str(row["class_name"]) for row in rows]
    map50 = [float(row["map50"]) for row in rows]
    map50_95 = [float(row["map50_95"]) for row in rows]
    precision = [float(row["precision"]) for row in rows]
    recall = [float(row["recall"]) for row in rows]
    positions = list(range(len(rows)))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    axes[0].bar(
        [position - width / 2 for position in positions],
        map50,
        width=width,
        label="mAP50",
        color="#17becf",
    )
    axes[0].bar(
        [position + width / 2 for position in positions],
        map50_95,
        width=width,
        label="mAP50-95",
        color="#1f77b4",
    )
    axes[0].set_title("Per-Class Average Precision")
    axes[0].set_ylim(0.0, max(0.1, max(map50 + map50_95) * 1.15))
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(
        [position - width / 2 for position in positions],
        precision,
        width=width,
        label="precision",
        color="#9467bd",
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        recall,
        width=width,
        label="recall",
        color="#8c564b",
    )
    axes[1].set_title("Per-Class Precision / Recall")
    axes[1].set_ylim(0.0, max(0.1, max(precision + recall) * 1.15))
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(class_names, rotation=25, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path
