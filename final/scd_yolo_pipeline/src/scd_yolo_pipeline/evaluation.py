from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset import HeldOutDataset
from .pipeline import Instance, PipelineResult, SCDPipeline, mask_iou

PAIR_COLUMNS = (
    "smear",
    "set",
    "gt_index",
    "pred_index",
    "iou",
    "y",
    "pred",
    "inception_pred",
    "segmenter_pred",
    "detection_confidence",
    "p_sickle",
)


@dataclass
class EvaluationResult:
    per_smear: pd.DataFrame
    matched_cells: pd.DataFrame
    detection: dict[str, float | int]
    classification: dict[str, float | int]
    inception_classification: dict[str, float | int]
    segmenter_classification: dict[str, float | int]
    end_to_end: dict[str, float | int]
    quality_controlled: dict[str, object]
    confusion_matrix: np.ndarray
    inception_confusion_matrix: np.ndarray
    segmenter_confusion_matrix: np.ndarray

    def summary(self) -> dict[str, object]:
        return {
            "n_test_smears": len(self.per_smear),
            "detection": self.detection,
            "classification_on_matched": self.classification,
            "inception_classification_on_matched": self.inception_classification,
            "segmenter_classification_on_matched": self.segmenter_classification,
            "end_to_end": self.end_to_end,
            "quality_controlled": self.quality_controlled,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "inception_confusion_matrix": self.inception_confusion_matrix.tolist(),
            "segmenter_confusion_matrix": self.segmenter_confusion_matrix.tolist(),
        }


def greedy_match(
    ground_truth: list[Instance],
    predictions: list[Instance],
    threshold: float,
) -> list[tuple[int, int, float]]:
    if not ground_truth or not predictions:
        return []
    matrix = np.array(
        [[mask_iou(gt, pred) for pred in predictions] for gt in ground_truth],
        dtype=float,
    )
    matches: list[tuple[int, int, float]] = []
    while matrix.size:
        flat_index = int(matrix.argmax())
        gt_index, pred_index = divmod(flat_index, matrix.shape[1])
        score = float(matrix[gt_index, pred_index])
        if score < threshold:
            break
        matches.append((gt_index, pred_index, score))
        matrix[gt_index, :] = -1.0
        matrix[:, pred_index] = -1.0
    return matches


def _classification_metrics(
    matched_cells: pd.DataFrame,
    prediction_column: str,
) -> tuple[dict[str, float | int], np.ndarray]:
    confusion = np.zeros((2, 2), dtype=int)
    for row in matched_cells.itertuples():
        confusion[int(row.y), int(getattr(row, prediction_column))] += 1
    class_f1: list[float] = []
    for class_id in (0, 1):
        precision = confusion[class_id, class_id] / max(int(confusion[:, class_id].sum()), 1)
        recall = confusion[class_id, class_id] / max(int(confusion[class_id].sum()), 1)
        class_f1.append(2.0 * precision * recall / max(precision + recall, 1e-9))
    count = len(matched_cells)
    accuracy = (
        float((matched_cells["y"] == matched_cells[prediction_column]).mean()) if count else 0.0
    )
    metrics: dict[str, float | int] = {
        "n_matched": count,
        "accuracy": accuracy,
        "macro_f1": float(np.mean(class_f1)),
        "f1_circular": class_f1[0],
        "f1_elongated": class_f1[1],
        "sickle_recall": float(confusion[1, 1]) / max(int(confusion[1].sum()), 1),
    }
    return metrics, confusion


def evaluate_pipeline(
    pipeline: SCDPipeline,
    dataset: HeldOutDataset,
    *,
    limit: int | None = None,
    on_result: Callable[[int, str, PipelineResult], None] | None = None,
) -> EvaluationResult:
    test_smears = dataset.test_smears[:limit] if limit else dataset.test_smears
    pair_rows: list[dict[str, object]] = []
    smear_rows: list[dict[str, object]] = []

    for index, smear in enumerate(test_smears):
        result = pipeline.run(dataset.source_paths[smear])
        ground_truth = dataset.ground_truth[smear]
        matches = greedy_match(
            ground_truth,
            result.instances,
            pipeline.config.mask_iou_threshold,
        )

        result.cells["matched"] = False
        result.cells["gt_class"] = pd.NA
        result.cells["match_iou"] = np.nan
        for gt_index, pred_index, iou in matches:
            gt = ground_truth[gt_index]
            pred = result.instances[pred_index]
            result.cells.loc[pred_index, ["matched", "gt_class", "match_iou"]] = (
                True,
                pipeline.classes[gt.cls],
                iou,
            )
            pair_rows.append(
                {
                    "smear": smear,
                    "set": dataset.set_by_smear[smear],
                    "gt_index": gt_index,
                    "pred_index": pred_index,
                    "iou": iou,
                    "y": gt.cls,
                    "pred": pred.cls,
                    "inception_pred": pred.inception_class,
                    "segmenter_pred": pred.det_class,
                    "detection_confidence": pred.det_confidence,
                    "p_sickle": pred.score,
                }
            )

        if on_result is not None:
            on_result(index, smear, result)

        gt_sickle = sum(instance.cls == 1 for instance in ground_truth)
        pred_sickle = result.sickle_count
        inception_sickle = sum(instance.inception_class == 1 for instance in result.instances)
        segmenter_sickle = sum(instance.det_class == 1 for instance in result.instances)
        matched_gt_sickle = sum(ground_truth[gt].cls == 1 for gt, _, _ in matches)
        matched_pred_sickle = sum(result.instances[pred].cls == 1 for _, pred, _ in matches)
        matched_inception_sickle = sum(
            result.instances[pred].inception_class == 1 for _, pred, _ in matches
        )
        matched_segmenter_sickle = sum(
            result.instances[pred].det_class == 1 for _, pred, _ in matches
        )
        matched_count = len(matches)
        unmatched_count = len(result.instances) - matched_count
        smear_rows.append(
            {
                "smear": smear,
                "set": dataset.set_by_smear[smear],
                "annotation_complete": smear not in dataset.missing_masks,
                "gt": len(ground_truth),
                "pred": len(result.instances),
                "tp": matched_count,
                "sum_iou": sum(iou for _, _, iou in matches),
                "gt_sickle": gt_sickle,
                "pred_sickle": pred_sickle,
                "inception_sickle": inception_sickle,
                "segmenter_sickle": segmenter_sickle,
                "matched_gt_sickle": matched_gt_sickle,
                "matched_pred_sickle": matched_pred_sickle,
                "matched_inception_sickle": matched_inception_sickle,
                "matched_segmenter_sickle": matched_segmenter_sickle,
                "unmatched_pred": unmatched_count,
                "unmatched_pred_sickle": pred_sickle - matched_pred_sickle,
                "unmatched_inception_sickle": inception_sickle - matched_inception_sickle,
                "unmatched_segmenter_sickle": segmenter_sickle - matched_segmenter_sickle,
                "gt_pct": 100.0 * gt_sickle / max(len(ground_truth), 1),
                "pred_pct": result.sickle_percentage,
                "inception_pct": 100.0 * inception_sickle / max(len(result.instances), 1),
                "segmenter_pct": 100.0 * segmenter_sickle / max(len(result.instances), 1),
                "matched_gt_pct": 100.0 * matched_gt_sickle / max(matched_count, 1),
                "matched_pred_pct": 100.0 * matched_pred_sickle / max(matched_count, 1),
                "matched_inception_pct": 100.0 * matched_inception_sickle / max(matched_count, 1),
                "matched_segmenter_pct": 100.0 * matched_segmenter_sickle / max(matched_count, 1),
            }
        )
        print(
            f"[{index + 1:02d}/{len(test_smears):02d}] {smear:<24} "
            f"{len(result.instances):>4} detected / {len(ground_truth):>4} gt · "
            f"{matched_count:>4} matched · sickle {pred_sickle:>3} vs {gt_sickle:>3}"
        )

    per_smear = pd.DataFrame(smear_rows)
    matched_cells = pd.DataFrame(pair_rows, columns=PAIR_COLUMNS)
    true_positives = int(per_smear["tp"].sum())
    ground_truth_count = int(per_smear["gt"].sum())
    prediction_count = int(per_smear["pred"].sum())
    precision = true_positives / max(prediction_count, 1)
    recall = true_positives / max(ground_truth_count, 1)
    detection = {
        "F1": 2.0 * precision * recall / max(precision + recall, 1e-9),
        "precision": precision,
        "recall": recall,
        "mIoU": float(per_smear["sum_iou"].sum()) / max(true_positives, 1),
        "pred": prediction_count,
        "gt": ground_truth_count,
    }

    classification, confusion = _classification_metrics(matched_cells, "pred")
    inception_classification, inception_confusion = _classification_metrics(
        matched_cells, "inception_pred"
    )
    segmenter_classification, segmenter_confusion = _classification_metrics(
        matched_cells, "segmenter_pred"
    )
    differences = per_smear["pred_pct"] - per_smear["gt_pct"]
    inception_differences = per_smear["inception_pct"] - per_smear["gt_pct"]
    segmenter_differences = per_smear["segmenter_pct"] - per_smear["gt_pct"]
    matched_differences = per_smear["matched_pred_pct"] - per_smear["matched_gt_pct"]
    matched_inception_differences = per_smear["matched_inception_pct"] - per_smear["matched_gt_pct"]
    matched_segmenter_differences = per_smear["matched_segmenter_pct"] - per_smear["matched_gt_pct"]
    end_to_end: dict[str, float | int] = {
        "mae_sickle_pct": float(differences.abs().mean()),
        "bias_sickle_pct": float(differences.mean()),
        "matched_only_mae_sickle_pct": float(matched_differences.abs().mean()),
        "matched_only_bias_sickle_pct": float(matched_differences.mean()),
        "inception_mae_sickle_pct": float(inception_differences.abs().mean()),
        "inception_bias_sickle_pct": float(inception_differences.mean()),
        "matched_inception_mae_sickle_pct": float(matched_inception_differences.abs().mean()),
        "matched_inception_bias_sickle_pct": float(matched_inception_differences.mean()),
        "segmenter_mae_sickle_pct": float(segmenter_differences.abs().mean()),
        "segmenter_bias_sickle_pct": float(segmenter_differences.mean()),
        "matched_segmenter_mae_sickle_pct": float(matched_segmenter_differences.abs().mean()),
        "matched_segmenter_bias_sickle_pct": float(matched_segmenter_differences.mean()),
        "unmatched_predictions": int(per_smear["unmatched_pred"].sum()),
        "unmatched_classifier_sickle": int(per_smear["unmatched_pred_sickle"].sum()),
        "unmatched_inception_sickle": int(per_smear["unmatched_inception_sickle"].sum()),
        "unmatched_segmenter_sickle": int(per_smear["unmatched_segmenter_sickle"].sum()),
    }
    excluded_smears = sorted(dataset.missing_masks)
    complete_smears = per_smear.loc[per_smear["annotation_complete"]]
    complete_matches = matched_cells.loc[~matched_cells["smear"].isin(excluded_smears)]
    qc_classification, _ = _classification_metrics(complete_matches, "pred")
    qc_inception, _ = _classification_metrics(complete_matches, "inception_pred")
    qc_segmenter, _ = _classification_metrics(complete_matches, "segmenter_pred")
    qc_tp = int(complete_smears["tp"].sum())
    qc_pred = int(complete_smears["pred"].sum())
    qc_gt = int(complete_smears["gt"].sum())
    qc_precision = qc_tp / max(qc_pred, 1)
    qc_recall = qc_tp / max(qc_gt, 1)
    qc_selected_difference = complete_smears["pred_pct"] - complete_smears["gt_pct"]
    qc_inception_difference = complete_smears["inception_pct"] - complete_smears["gt_pct"]
    qc_segmenter_difference = complete_smears["segmenter_pct"] - complete_smears["gt_pct"]
    quality_controlled: dict[str, object] = {
        "n_test_smears": len(complete_smears),
        "excluded_smears": {smear: dataset.missing_masks[smear] for smear in excluded_smears},
        "detection": {
            "F1": 2.0 * qc_precision * qc_recall / max(qc_precision + qc_recall, 1e-9),
            "precision": qc_precision,
            "recall": qc_recall,
            "mIoU": float(complete_smears["sum_iou"].sum()) / max(qc_tp, 1),
            "pred": qc_pred,
            "gt": qc_gt,
        },
        "classification_on_matched": qc_classification,
        "inception_classification_on_matched": qc_inception,
        "segmenter_classification_on_matched": qc_segmenter,
        "end_to_end": {
            "mae_sickle_pct": float(qc_selected_difference.abs().mean()),
            "bias_sickle_pct": float(qc_selected_difference.mean()),
            "inception_mae_sickle_pct": float(qc_inception_difference.abs().mean()),
            "inception_bias_sickle_pct": float(qc_inception_difference.mean()),
            "segmenter_mae_sickle_pct": float(qc_segmenter_difference.abs().mean()),
            "segmenter_bias_sickle_pct": float(qc_segmenter_difference.mean()),
        },
    }
    return EvaluationResult(
        per_smear,
        matched_cells,
        detection,
        classification,
        inception_classification,
        segmenter_classification,
        end_to_end,
        quality_controlled,
        confusion,
        inception_confusion,
        segmenter_confusion,
    )
