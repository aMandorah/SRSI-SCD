from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .nigeria import NigeriaSample
from .pipeline import SCDPipeline


@dataclass
class ExternalEvaluation:
    fields: pd.DataFrame
    samples: pd.DataFrame
    cells: pd.DataFrame
    excluded: pd.DataFrame
    metrics: dict[str, object]


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score, kind="stable")
    y = y[order]
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    y = y[order]
    positives = max(int(y.sum()), 1)
    cumulative = np.cumsum(y)
    return float((cumulative[y == 1] / np.arange(1, len(y) + 1)[y == 1]).sum() / positives)


def _bootstrap(
    y: np.ndarray, score: np.ndarray, groups: np.ndarray, seed: int = 42
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    values: list[tuple[float, float]] = []
    for _ in range(2000):
        selected = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in selected])
        values.append(
            (_auc(y[indices], score[indices]), _average_precision(y[indices], score[indices]))
        )
    array = np.asarray(values, dtype=float)
    return {
        "auroc": _auc(y, score),
        "average_precision": _average_precision(y, score),
        "bootstrap_95ci": {
            "auroc": np.nanpercentile(array[:, 0], [2.5, 97.5]).tolist(),
            "average_precision": np.nanpercentile(array[:, 1], [2.5, 97.5]).tolist(),
        },
        "bootstrap_replicates": len(values),
        "bootstrap_group": "repeat_family",
    }


def _metrics(frame: pd.DataFrame, score_column: str) -> dict[str, object]:
    valid = frame.loc[frame[score_column].notna()]
    y = valid["label"].to_numpy(dtype=int)
    score = valid[score_column].to_numpy(dtype=float)
    groups = valid["family_id"].to_numpy(dtype=str)
    result = _bootstrap(y, score, groups)
    result["n_samples"] = len(valid)
    result["positive"] = int(y.sum())
    result["negative"] = int(len(y) - y.sum())
    return result


def evaluate_nigeria(
    pipeline: SCDPipeline,
    samples: list[NigeriaSample],
    experiment_path: Path,
    *,
    stage_sample_ids: set[str] | None = None,
    on_field: Callable[[int, NigeriaSample, Path, object], None] | None = None,
) -> ExternalEvaluation:
    field_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    cell_frames: list[pd.DataFrame] = []
    excluded_rows: list[dict[str, object]] = []
    for sample_index, sample in enumerate(samples):
        totals = {"pred": 0, "yolo_elongated": 0, "inception_elongated": 0, "fields": 0}
        for field_index, image in enumerate(sample.images):
            try:
                result = pipeline.run(image)
            except (
                OSError,
                ValueError,
                RuntimeError,
            ) as error:  # retain the cohort audit and continue
                excluded_rows.append(
                    {"sample_id": sample.sample_id, "image": str(image), "reason": str(error)}
                )
                continue
            count = len(result.instances)
            yolo = sum(instance.det_class == 1 for instance in result.instances)
            inception = sum(instance.inception_class == 1 for instance in result.instances)
            if count == 0:
                excluded_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "image": str(image),
                        "reason": "zero_detections",
                    }
                )
                continue
            totals["pred"] += count
            totals["yolo_elongated"] += yolo
            totals["inception_elongated"] += inception
            totals["fields"] += 1
            row = {
                "sample_id": sample.sample_id,
                "family_id": sample.family_id,
                "label": sample.label,
                "field_index": field_index,
                "image": str(image),
                "detected_cells": count,
                "yolo_elongated": yolo,
                "inception_elongated": inception,
                "yolo_burden_pct": 100.0 * yolo / count,
                "inception_burden_pct": 100.0 * inception / count,
            }
            field_rows.append(row)
            cells = result.cells.copy()
            cells.insert(0, "field_index", field_index)
            cells.insert(0, "sample_id", sample.sample_id)
            cells.insert(1, "family_id", sample.family_id)
            cells.insert(2, "label", sample.label)
            cells.insert(3, "image", str(image))
            cell_frames.append(cells)
            if on_field is not None:
                on_field(field_index, sample, image, result)
        if totals["fields"] >= 3 and totals["pred"] >= 100:
            sample_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "family_id": sample.family_id,
                    "label": sample.label,
                    "fields": totals["fields"],
                    "detected_cells": totals["pred"],
                    "yolo_elongated": totals["yolo_elongated"],
                    "inception_elongated": totals["inception_elongated"],
                    "yolo_burden_pct": 100.0 * totals["yolo_elongated"] / totals["pred"],
                    "inception_burden_pct": 100.0 * totals["inception_elongated"] / totals["pred"],
                }
            )
        else:
            excluded_rows.append(
                {"sample_id": sample.sample_id, "image": "", "reason": "sample_qc_failed"}
            )
    fields = pd.DataFrame(field_rows)
    samples_frame = pd.DataFrame(sample_rows)
    cells_frame = pd.concat(cell_frames, ignore_index=True) if cell_frames else pd.DataFrame()
    excluded = pd.DataFrame(excluded_rows)
    metrics = {
        "n_samples_processed": len(samples_frame),
        "n_fields_processed": len(fields),
        "n_excluded_records": len(excluded),
        "yolo": _metrics(samples_frame, "yolo_burden_pct") if not samples_frame.empty else {},
        "inception": _metrics(samples_frame, "inception_burden_pct")
        if not samples_frame.empty
        else {},
    }
    return ExternalEvaluation(fields, samples_frame, cells_frame, excluded, metrics)
