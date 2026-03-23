from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from model.detector import VSTDet
from training.engine import append_history, save_checkpoint
from utils.evaluator import format_metrics_table
from utils.plots import plot_history, plot_per_class_metrics
from utils.reporting import append_results_summary


@dataclass
class EpochCallbackState:
    epoch: int
    epochs: int
    output_dir: Path
    row: dict[str, float | int]
    val_report: dict[str, object] | None
    best_metric: float
    best_epoch: int
    names: list[str]
    model: VSTDet
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


@dataclass
class TrainEndState:
    epochs: int
    output_dir: Path
    total_hours: float
    best_epoch: int
    best_metric: float
    last_val_report: dict[str, object] | None
    run_metadata: dict[str, object]


class Callback:
    def on_epoch_end(self, state: EpochCallbackState) -> None:
        return None

    def on_train_end(self, state: TrainEndState) -> None:
        return None


class CallbackManager:
    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self.callbacks = callbacks or []

    def on_epoch_end(self, state: EpochCallbackState) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(state)

    def on_train_end(self, state: TrainEndState) -> None:
        for callback in self.callbacks:
            callback.on_train_end(state)


class CheckpointCallback(Callback):
    def __init__(self, save_every: int) -> None:
        self.save_every = save_every

    def on_epoch_end(self, state: EpochCallbackState) -> None:
        metric = float(state.row.get("map50_95", float("-inf")))
        if state.val_report is not None and metric >= state.best_metric:
            state.best_metric = metric
            state.best_epoch = state.epoch
            save_checkpoint(
                state.output_dir / "best.pt",
                state.model,
                state.optimizer,
                state.scheduler,
                state.epoch,
                state.best_metric,
                state.best_epoch,
                state.names,
            )

        if state.epoch % self.save_every == 0 or state.epoch == state.epochs:
            save_checkpoint(
                state.output_dir / f"epoch_{state.epoch:03d}.pt",
                state.model,
                state.optimizer,
                state.scheduler,
                state.epoch,
                state.best_metric,
                state.best_epoch,
                state.names,
            )
        save_checkpoint(
            state.output_dir / "last.pt",
            state.model,
            state.optimizer,
            state.scheduler,
            state.epoch,
            state.best_metric,
            state.best_epoch,
            state.names,
        )


class ArtifactCallback(Callback):
    def on_epoch_end(self, state: EpochCallbackState) -> None:
        history_path = state.output_dir / "history.csv"
        append_history(history_path, state.row)
        plot_history(history_path, state.output_dir / "training_curves.png")
        if state.val_report is not None:
            plot_per_class_metrics(state.val_report, state.output_dir / "per_class_metrics.png")


class FinalReportCallback(Callback):
    def __init__(self, results_summary_path: Path | None = None) -> None:
        self.results_summary_path = results_summary_path

    def on_train_end(self, state: TrainEndState) -> None:
        print(f"\n{state.epochs} epochs completed in {state.total_hours:.3f} hours.")
        print(f"Results saved to {state.output_dir}")
        if state.last_val_report is not None:
            table = format_metrics_table(state.last_val_report)
            print("\nFinal checkpoint val:")
            print(table)
            (state.output_dir / "final_metrics.txt").write_text(
                "Final checkpoint val:\n" + table + "\n",
                encoding="utf-8",
            )
            if self.results_summary_path is not None:
                summary = state.last_val_report["summary"]
                append_results_summary(
                    self.results_summary_path,
                    {
                        **state.run_metadata,
                        "kind": "train",
                        "run_name": state.output_dir.name,
                        "output_dir": state.output_dir,
                        "best_checkpoint": state.output_dir / "best.pt",
                        "last_checkpoint": state.output_dir / "last.pt",
                        "epochs": state.epochs,
                        "best_epoch": state.best_epoch,
                        "best_metric": state.best_metric,
                        "total_hours": state.total_hours,
                        "precision": summary["precision"],
                        "recall": summary["recall"],
                        "map50": summary["map50"],
                        "map50_95": summary["map50_95"],
                    },
                )


def build_default_callbacks(
    save_every: int,
    results_summary_path: Path | None = None,
) -> CallbackManager:
    return CallbackManager(
        callbacks=[
            CheckpointCallback(save_every=save_every),
            ArtifactCallback(),
            FinalReportCallback(results_summary_path=results_summary_path),
        ]
    )
