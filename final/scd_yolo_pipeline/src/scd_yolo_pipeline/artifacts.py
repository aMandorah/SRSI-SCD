from __future__ import annotations

import json
import math
import re
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluation import EvaluationResult
from .pipeline import DEFAULT_CLASSES, Instance, PipelineResult

CLASS_COLORS = {
    0: np.array([0.30, 0.45, 0.69]),
    1: np.array([0.77, 0.31, 0.32]),
}
DETECTION_COLOR = np.array([0.20, 0.72, 0.48])


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "image"


@dataclass
class Experiment:
    path: Path
    index: int

    @classmethod
    def create(cls, root: Path) -> Experiment:
        root.mkdir(parents=True, exist_ok=True)
        for index in range(1_000_000):
            path = root / f"exp_{index:03d}"
            try:
                path.mkdir()
            except FileExistsError:
                continue
            (path / "images").mkdir()
            (path / "tables").mkdir()
            experiment = cls(path, index)
            experiment.write_json(
                "run.json",
                {
                    "experiment": path.name,
                    "status": "running",
                    "started_at": datetime.now(UTC).isoformat(),
                },
            )
            return experiment
        raise RuntimeError(f"No available experiment number under {root}")

    def write_json(self, relative_path: str | Path, payload: Any) -> Path:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def complete(self, payload: dict[str, Any]) -> None:
        current = json.loads((self.path / "run.json").read_text())
        current.update(payload)
        current.update(
            {
                "status": "complete",
                "finished_at": datetime.now(UTC).isoformat(),
            }
        )
        self.write_json("run.json", current)

    def fail(self, error: BaseException) -> None:
        current = json.loads((self.path / "run.json").read_text())
        current.update(
            {
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        self.write_json("run.json", current)
        (self.path / "error.txt").write_text(traceback.format_exc())


def _write_bgr(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Failed to write image: {path}")


def _overlay(
    canvas: np.ndarray,
    instances: list[Instance],
    *,
    classified: bool,
) -> np.ndarray:
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    overlay = rgb.copy()
    for instance in instances:
        region = (slice(instance.y0, instance.y1), slice(instance.x0, instance.x1))
        color = CLASS_COLORS[instance.cls] if classified else DETECTION_COLOR
        overlay[region][instance.mask] = 0.35 * rgb[region][instance.mask] + 0.65 * color
    return overlay


def _save_overlay(
    path: Path,
    canvas: np.ndarray,
    instances: list[Instance],
    title: str,
    *,
    classified: bool,
    score_source: str = "inception",
) -> None:
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(_overlay(canvas, instances, classified=classified))
    axis.set_title(title)
    axis.set_axis_off()
    if classified:
        for instance in instances:
            axis.text(
                instance.x0,
                max(0, instance.y0 - 3),
                (
                    f"{instance.det_confidence:.2f}"
                    if score_source == "yolo"
                    else f"{instance.score:.2f}"
                ),
                fontsize=5,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
            )
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _save_crop_grid(path: Path, result: PipelineResult, max_crops: int = 64) -> None:
    count = min(len(result.crops), max_crops)
    columns = 8
    rows = max(1, math.ceil(max(count, 1) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 1.65 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.set_axis_off()
    for index, crop in enumerate(result.crops[:count]):
        axis = axes.ravel()[index]
        axis.imshow(crop)
        instance = result.instances[index]
        axis.set_title(
            f"#{index} I={instance.score:.2f} Y={instance.det_class}",
            fontsize=7,
            color="#b33" if instance.cls == 1 else "#245a9c",
        )
    if count == 0:
        axes.ravel()[0].text(0.5, 0.5, "No valid crops", ha="center", va="center")
    figure.suptitle(f"Stage 03 · source-resolution cell crops ({count}/{len(result.crops)})")
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def save_stage_images(
    experiment: Experiment,
    result: PipelineResult,
    image_key: str,
) -> dict[str, str]:
    directory = experiment.path / "images" / safe_name(image_key)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "stage_00_source": directory / "stage_00_source.jpg",
        "stage_01_canvas": directory / "stage_01_canvas.jpg",
        "stage_02_segmentation": directory / "stage_02_segmentation.png",
        "stage_03_crops": directory / "stage_03_crops.png",
        "stage_04_classification": directory / "stage_04_classification.png",
    }
    _write_bgr(paths["stage_00_source"], result.source)
    _write_bgr(paths["stage_01_canvas"], result.canvas)
    _save_overlay(
        paths["stage_02_segmentation"],
        result.canvas,
        result.segmentation_instances,
        f"Stage 02 · YOLO segmentation · {len(result.segmentation_instances)} cells",
        classified=False,
    )
    _save_crop_grid(paths["stage_03_crops"], result)
    _save_overlay(
        paths["stage_04_classification"],
        result.canvas,
        result.instances,
        (
            f"Stage 04 · {result.classification_source} classification · "
            f"{len(result.instances)} cells · "
            f"{result.sickle_count} sickle ({result.sickle_percentage:.1f}%)"
        ),
        classified=True,
        score_source=result.classification_source,
    )
    return {key: str(path.relative_to(experiment.path)) for key, path in paths.items()}


def save_evaluation_figure(
    experiment: Experiment,
    result: EvaluationResult,
) -> Path:
    per_smear = result.per_smear
    confusion = result.confusion_matrix
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    detection_labels = ("F1", "precision", "recall", "mIoU")
    axes[0].bar(
        detection_labels,
        [result.detection[label] for label in detection_labels],
        color="#2a6",
    )
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_title("Detection @ mask IoU 0.50")

    axes[1].imshow(confusion, cmap="Blues")
    midpoint = confusion.max() / 2 if confusion.size else 0
    for row in range(2):
        for column in range(2):
            axes[1].text(
                column,
                row,
                confusion[row, column],
                ha="center",
                va="center",
                color="white" if confusion[row, column] > midpoint else "black",
            )
    axes[1].set_xticks([0, 1], DEFAULT_CLASSES)
    axes[1].set_yticks([0, 1], DEFAULT_CLASSES)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Ground truth")
    axes[1].set_title(
        f"Matched-cell classification\naccuracy {result.classification['accuracy']:.3f} · "
        f"macro-F1 {result.classification['macro_f1']:.3f}"
    )

    limit = max(float(per_smear["gt_pct"].max()), float(per_smear["pred_pct"].max())) * 1.15 + 1
    axes[2].plot([0, limit], [0, limit], "--", color="grey", linewidth=1)
    for set_name, group in per_smear.groupby("set"):
        axes[2].scatter(group["gt_pct"], group["pred_pct"], label=set_name, s=45)
    axes[2].set_xlim(0, limit)
    axes[2].set_ylim(0, limit)
    axes[2].grid(alpha=0.3)
    axes[2].legend()
    axes[2].set_xlabel("Ground-truth sickle %")
    axes[2].set_ylabel("Pipeline sickle %")
    axes[2].set_title(f"Per-smear burden\nMAE {result.end_to_end['mae_sickle_pct']:.1f} points")
    figure.suptitle(f"YOLO two-stage pipeline · {len(per_smear)} held-out smears")
    figure.tight_layout()
    path = experiment.path / "images" / "evaluation_scores.png"
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return path
