# A100 evaluation and investigation

## Outcome

exp/exp_006 is the final audit-aware run. It completed all 12 held-out smears on an
NVIDIA A100-SXM4-80GB in 53 seconds, producing 60 stage images, one evaluation figure,
12 per-smear cell tables, checkpoint hashes, runtime versions, and a complete run manifest.

The YOLO-primary mode is the recommended experimental pipeline:

~~~bash
uv run scd-yolo evaluate --classification-source yolo
~~~

Inception is still executed and recorded for direct comparison. The notebook-compatible default
remains --classification-source inception.

## Results

| Metric | Inception decision | YOLO decision |
|---|---:|---:|
| Matched-cell accuracy, all 12 | 84.17% | 96.14% |
| Matched-cell macro-F1, all 12 | 83.21% | 95.65% |
| Sickle recall, all 12 | 93.98% | 97.59% |
| Raw burden MAE, all 12 | 28.43 points | 11.99 points |
| Quality-controlled burden MAE, 11 | 25.53 points | 5.57 points |
| Quality-controlled burden bias, 11 | +25.53 points | +3.79 points |

Detection is identical between modes because only the classification source changes:

| Detection metric | Raw 12-smear result | Quality-controlled 11-smear result |
|---|---:|---:|
| Precision | 57.30% | 64.78% |
| Recall | 97.00% | 97.67% |
| F1 | 72.04% | 77.90% |
| Mean matched-mask IoU | 93.29% | 93.34% |

The detector numbers reproduce the final notebook result, approximately F1 0.719 and mIoU 0.932,
which confirms that the Python conversion is faithful.

## Why the original end-to-end result was weak

The localization itself is strong: recall is 97% and matched-mask IoU is 93%. Precision is lower
because 452 cells are predicted against 267 annotated instances, leaving 193 unmatched
predictions. Some are false positives, while others appear to be real cells omitted by partial
annotations.

The Inception checkpoint also has a crop-domain mismatch. It marks 120 of the 193 unmatched
detections as sickle, compared with 46 from the YOLO checkpoint. On matched cells it reaches
84.17% accuracy, while YOLO reaches 96.14%. Visual review of the previous worst smear,
49erythrocytesIDB2, showed centered source-resolution crops but many visually circular cells
receiving high Inception sickle probabilities. This rules out a simple coordinate-mapping bug and
points to classifier domain shift.

## Annotation defect

37erythrocytesIDB2 contains:

- mask-elongated.jpg
- mask-other.jpg
- mask.jpg

but no mask-circular.jpg. The old notebook silently interpreted this as zero circular cells.
That smear reports 9 ground-truth instances versus 63 detections and dominates the raw burden
error. The new package does not alter or discard it: summary.json retains the raw 12-smear
metrics, lists the exact missing filename under dataset.annotation_warnings, and separately
reports evaluation.quality_controlled over the 11 complete smears.

## Recommendation

Use YOLO-primary mode for reviewing this pipeline because it is materially stronger on the
available held-out data and uses the same supplied checkpoint. Keep Inception outputs in the
tables for comparison and future retraining work. Before clinical or production use, repair the
missing annotation and evaluate on a clean external smear-level split; the supplied data cannot
establish deployment-grade generalization.


## External Nigerian validation

The Nigerian release is an independent thin-blood-film cohort from University College Hospital,
Ibadan, with sample-level SCD labels obtained by hemoglobin electrophoresis. It is evaluated as an
external discrimination study. Because the public release does not include cell-level masks or
circular/elongated cell annotations, its results must not be merged with the Cuban mask-based
detection metrics. The external report records sample and field counts, elongated burden, classifier
agreement, AUROC, average precision, clustered bootstrap intervals, and all exclusions.
