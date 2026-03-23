from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


RESULTS_SUMMARY_FIELDS = [
    "timestamp",
    "kind",
    "run_name",
    "output_dir",
    "config",
    "data",
    "source",
    "checkpoint",
    "best_checkpoint",
    "last_checkpoint",
    "split",
    "device",
    "variant",
    "backbone",
    "neck",
    "head_depth",
    "detail_branch",
    "assigner",
    "imgsz",
    "batch_size",
    "epochs",
    "best_epoch",
    "best_metric",
    "total_hours",
    "class_aware_sampling",
    "sample_classes",
    "sample_boost_factor",
    "precision",
    "recall",
    "map50",
    "map50_95",
]


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def append_results_summary(path: str | Path, row: dict[str, object]) -> None:
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **row,
    }
    file_exists = summary_path.exists()
    with summary_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULTS_SUMMARY_FIELDS,
            extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                field: _stringify(payload.get(field, ""))
                for field in RESULTS_SUMMARY_FIELDS
            }
        )
