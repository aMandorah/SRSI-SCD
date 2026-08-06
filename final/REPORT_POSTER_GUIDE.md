# Report, Poster, and Presentation Guide

This document is a content bank for writing the project poster, paper/report, and presentation.
It distinguishes the original notebook-compatible two-stage pipeline from the stronger audited
YOLO-primary configuration. Use the exact numbers below and preserve the limitations; do not turn
the results into a clinical claim.

## 1. Recommended project framing

### Primary title

**An Audited Deep-Learning Pipeline for Red Blood Cell Segmentation, Morphological Classification,
and Sickle-Cell Burden Estimation**

### Alternative titles

- **From Blood Smear to Sickle-Cell Burden: An Instance-Segmentation and Classification Pipeline**
- **Automated Red Blood Cell Analysis Using YOLO11 Segmentation and Morphological Classification**
- **Benchmarking Segmentation and Classification Models for Sickle-Cell Morphology Analysis**

### One-sentence summary

We developed and audited an image-analysis pipeline that segments individual red blood cells from
whole-smear images, labels them as circular or elongated, and estimates the percentage of
elongated/sickle-like cells per smear.

### Thirty-second pitch

Manual review of blood smears requires locating many individual cells and judging their morphology.
Our pipeline automates this workflow: a fine-tuned YOLO11 segmentation model finds each red blood
cell, source-resolution crops are generated for classification, and the final cell labels are
aggregated into a smear-level sickle-cell burden. We compared multiple segmenters and classifiers,
converted the final notebook into a reproducible Python/`uv` package, and audited the complete
pipeline on held-out smear masks. The recommended YOLO-primary configuration achieved 96.1%
matched-cell accuracy and a quality-controlled burden MAE of 5.57 percentage points, although the
available evaluation remains too small and incomplete for clinical deployment claims.

## 2. Research problem and motivation

### Problem statement

- Whole blood-smear images contain many touching, overlapping, and visually variable red blood
  cells.
- A useful automated system must solve three connected tasks:
  1. locate and segment every cell;
  2. classify each detected cell by morphology;
  3. aggregate cell predictions into a clinically interpretable smear-level percentage.
- Strong isolated-cell classification does not guarantee strong whole-smear performance because
  detection errors and crop-domain shift propagate through the pipeline.

### Suggested motivation paragraph

Sickle cell disease alters red blood cell morphology, producing elongated or sickle-like cells that
can be observed in microscopy images. Automated image analysis could support consistent,
quantitative review by reducing the burden of manually locating and counting cells. The challenge
is not only classifying isolated cells: a complete system must first segment cells in a whole smear,
preserve enough source-image detail for morphology analysis, and combine the predictions into a
reliable per-smear burden estimate. This project therefore evaluates the complete path from a raw
smear image to per-cell predictions and a final sickle-cell percentage.

### Research questions

1. Which segmentation approach best detects individual red blood cells on held-out annotated
   smears?
2. Which classifier best separates circular and elongated cell crops on the internal crop dataset?
3. How much do detector errors and classifier domain shift affect end-to-end sickle-cell burden?
4. Does the YOLO model's own morphology prediction outperform the separately trained InceptionV3
   classifier when both are evaluated on matched whole-smear cells?

### Main contributions

- A full smear-to-burden pipeline, not just an isolated classifier.
- Smear-level splitting for the segmentation benchmark to reduce image leakage.
- Comparison of YOLO11-seg, fine-tuned SAM, zero-shot SAM, and Moondream-based detection.
- Comparison of deep, classical-kernel, and simulated quantum-kernel classifiers.
- Source-resolution crop mapping that prevents loss of morphology detail.
- Side-by-side auditing of InceptionV3 and YOLO morphology predictions.
- Explicit data-quality checks that report missing annotations rather than silently treating them
  as empty classes.
- A reproducible Python package with a locked `uv` environment and immutable numbered experiments.

## 3. Dataset section

### Dataset composition

The project uses the `SCD_Final` dataset containing Cuban and Ugandan microscopy data.

| Component | Recorded count | Role |
|---|---:|---|
| Cuba full-smear images | 80 | Whole-smear development/inference data |
| Cuba isolated circular cells | 202 | Morphology classification |
| Cuba isolated elongated cells | 211 | Morphology classification |
| Cuba mask directories | 79 | Pixel-level segmentation ground truth |
| Usable/labeled Cuba GT smears | 78 | Segmentation labels and benchmarking |
| GT cell instances | 2,118 | 1,635 circular + 483 elongated |
| Uganda positive-sickle-clean smears | 80 | Population/domain data |
| Uganda negative-normal smears | 147 | Population/domain data |
| Uganda isolated elongated cells | 1,273 | Morphology classification |

The ground-truth smears contained a mean of 27.2 labeled cells per smear, with a recorded range of
6 to 54.

### Segmentation data preparation

- Connected components from `mask-circular.jpg` and `mask-elongated.jpg` supplied the trusted seed
  instances.
- For segmentation training, circular and elongated instances were collapsed into an RBC detection
  class during the earlier development pipeline; the final fine-tuned checkpoint also retained
  circular/elongated class outputs.
- SAM was used as a pseudo-labeling tool for additional unannotated smears.
- The SAM filter was calibrated against annotated smears using cell area, aspect ratio, solidity,
  matched IoU, and instance recall—not selected by visual guesswork alone.
- SAM pseudo-label quality against Cuba ground truth was mean union-IoU 0.512 and instance
  recall@0.5 of 0.965.
- SAM generated 20,227 instances across 260 smears.
- The combined segmentation dataset contained 22,345 instances: 2,118 ground-truth-derived and
  20,227 SAM-derived.
- The recorded segmentation split contained 271 training and 67 validation smears.
- Ground-truth benchmark smears were reserved and never scored against SAM pseudo-labels.

### Classification crop dataset

| Split | Circular | Elongated | Total |
|---|---:|---:|---:|
| Train | 142 | 1,036 | 1,178 |
| Validation | 30 | 220 | 250 |
| Test | 30 | 228 | 258 |
| **Total** | **202** | **1,484** | **1,686** |

- The split was approximately 70/15/15 and grouped by source image where grouping information was
  available.
- Recorded source groups: 311 train, 67 validation, and 67 test, with no group shared across splits.
- The training set was imbalanced; recorded class weights were 4.148 for circular and 0.569 for
  elongated.
- Important caveat: the Cuba isolated-cell files used by the final step-1 classifier do not carry
  reliable smear IDs. Therefore overlap with the 12 final held-out mask smears cannot be ruled out.

### Final pipeline evaluation set

- The final end-to-end evaluation regenerated the notebook split with seed 42.
- Twelve Cuban mask smears were evaluated.
- Raw annotations contained 267 instances, of which 259 were matched to predictions at mask
  IoU ≥ 0.50.
- One test smear, `37erythrocytesIDB2`, has no `mask-circular.jpg` file. Raw 12-smear metrics are
  retained, while a separate quality-controlled summary reports the 11 fully annotated smears.
- Only Cuba supplies the pixel masks needed for this evaluation; do not claim cross-population
  segmentation validation.

### Dataset wording to avoid

- Do not say that the final evaluation proves performance across countries.
- Do not say that all unmatched predictions are false positives; some appear to be real but
  unlabeled cells.
- Do not call elongated morphology a definitive disease diagnosis. It is a morphology proxy used
  by this dataset.
- Do not invent patient demographics, acquisition protocols, ethics approvals, dataset licenses,
  or microscope specifications. Add those only after verifying the original dataset documentation.

## 4. Models and training

### Segmentation candidates

All development candidates were compared on the same 17 held-out ground-truth smears containing
466 real cell instances.

| Method | F1@0.50 | Precision | Recall | Mean matched IoU | Time/smear |
|---|---:|---:|---:|---:|---:|
| YOLO11-seg | **0.734** | 0.586 | **0.983** | **0.923** | **0.064 s** |
| Fine-tuned SAM | 0.662 | 0.505 | 0.961 | 0.874 | 1.553 s |
| Zero-shot SAM | 0.566 | 0.399 | 0.970 | 0.891 | 2.532 s |
| Moondream boxes/ellipses | 0.123 | 0.582 | 0.069 | ≈0.12 | ≈0.32 s |

Key development configuration for YOLO:

- YOLO11s-seg architecture.
- Input size: 1024×1024.
- Training: 100 epochs in the development benchmark.
- Recorded model size: approximately 10.1 million parameters and 32.9 GFLOPs.
- The final checkpoint manifest identifies `yolo_seg_finetuned.pt` as fine-tuned on 55 Cuban
  smears.
- Development comparison used confidence 0.15; the final packaged inference pipeline uses 0.25.
- YOLO won because it provided the best F1, recall, mask overlap, and latency balance.

### Classification candidates

All classifier candidates used the same internal crop split and scoring code.

| Model | Family | Macro-F1 | Accuracy | AUC | Time/crop |
|---|---|---:|---:|---:|---:|
| InceptionV3 | Deep fine-tuning | **1.000** | **1.000** | 1.000 | 9.65 ms |
| ResNet50 | Deep fine-tuning | **1.000** | **1.000** | 1.000 | 2.16 ms |
| MaxViT-Tiny | Deep fine-tuning | 0.982 | 0.992 | 1.000 | 11.25 ms |
| MobileNetV3 + RBF SVM | Frozen features + classical kernel | 0.982 | 0.992 | 1.000 | 0.15 ms |
| QSVM-ZZ | Simulated quantum kernel | 0.964 | 0.984 | 0.995 | 0.002 ms* |
| PCA-8 + RBF SVM | Classical control for QSVM | 0.945 | 0.977 | 0.995 | Not recorded |

\*The QSVM result used exact noiseless state-vector simulation, not quantum hardware. Its timing is
not comparable with hardware execution.

Deep-model training recipe:

- Two-phase fine-tuning: 8 + 12 epochs.
- Adam learning rates: 0.001 then 0.0001.
- Class weighting addressed the circular/elongated imbalance.
- Model selection used validation macro-F1, with AUC as a tiebreaker.
- InceptionV3 used 299×299 classifier inputs and approximately 24.3 million parameters.
- The selected threshold was tuned on validation and then frozen.

Final supplied InceptionV3 checkpoint metadata:

- Validation macro-F1: 1.0000.
- Internal test macro-F1: 0.9986.
- Internal test accuracy: 0.9992.
- Internal test AUC: 0.999996.
- Internal sickle/elongated recall: 1.0000.
- Frozen sickle probability threshold: 0.5044.

The development notebook table rounds InceptionV3 to 1.000 on the earlier 258-crop benchmark; use
the more precise `best_model.json` numbers when describing the final supplied checkpoint.

### Why both classifiers appear in the final system

The original step-4 notebook used YOLO only to detect cells and used InceptionV3 to assign the
circular/elongated label. The audit showed that the fine-tuned YOLO checkpoint's own morphology
labels generalize much better to whole-smear detections. The packaged system therefore supports:

- `--classification-source inception`: exact notebook-compatible two-stage behavior;
- `--classification-source yolo`: recommended final behavior;
- both outputs retained in every per-cell table for transparent comparison.

This is an important ablation result, not something to hide. It shows why isolated-cell benchmark
accuracy must be validated inside the complete inference pipeline.

## 5. Final inference pipeline

### Pipeline diagram text

Use this exact flow in a poster figure:

**Whole-smear image → bottom/right square padding → 1024×1024 canvas → YOLO11 instance
segmentation → source-coordinate mapping → 15%-padded cell crops → InceptionV3 probability + YOLO
morphology label → selected per-cell class → sickle-cell count and percentage**

### Detailed inference steps

1. Read the source smear with OpenCV.
2. Pad the bottom/right edges to a square without shifting the image origin.
3. Resize the square to 1024×1024 for YOLO inference.
4. Run YOLO with confidence threshold 0.25.
5. Threshold predicted masks at 0.5 and discard masks below 30 canvas pixels.
6. Map every detection box back to source coordinates using
   `scale = max(source_height, source_width) / 1024`.
7. Expand each source box by 15% of its maximum side.
8. Crop from the original-resolution image—not from the 1024 canvas.
9. Resize the cell crop to 80×80, matching the isolated-cell data representation.
10. For InceptionV3, convert to a tensor, resize to 299×299, and apply ImageNet normalization.
11. Infer in batches of up to 64 cells.
12. Record the YOLO class, YOLO confidence, Inception sickle probability, Inception class, selected
    class, source-coordinate box, and evaluation matching fields.
13. Compute smear burden as:

    `sickle percentage = 100 × number of elongated predictions / number of detected cells`

### Why source-resolution cropping matters

The source image and 1024 canvas can differ substantially in scale. On a 3136×2352 set-2 smear, a
cell may be approximately 26 pixels wide on the canvas but approximately 80 pixels in the source.
Cropping from the canvas would blur morphology before classification. Bottom/right padding keeps
the coordinate transform free of an offset term.

### Outputs per experiment

Each run creates the next immutable `exp_XXX` directory and saves:

- `config.json`: all paths and inference parameters;
- `run.json`: status, timestamps, elapsed time, and image count;
- `summary.json`: checkpoint hashes, software versions, raw metrics, QC metrics, and warnings;
- `tables/per_smear.csv`: counts and burden estimates per smear;
- `tables/matched_cells.csv`: ground-truth/prediction matches and both classifier outputs;
- `tables/cells/<smear>.csv`: every detected cell;
- five images per smear:
  1. source image;
  2. padded 1024 canvas;
  3. segmentation overlay;
  4. crop grid;
  5. classification overlay;
- an aggregate evaluation figure for full evaluation runs.

## 6. Evaluation protocol

### Detection

- Class-agnostic instance matching.
- Greedy highest-IoU one-to-one matching.
- A prediction is a true positive when mask IoU ≥ 0.50.
- Metrics: precision, recall, F1, and mean IoU over matched masks.

### Classification

- Calculated only on predictions matched to a ground-truth cell.
- Metrics: accuracy, macro-F1, class-wise F1, and sickle/elongated recall.
- Detection misses are represented in detection recall, not counted again as classification errors.

### End-to-end burden

- Predicted elongated percentage is calculated independently for every smear.
- Mean absolute error is measured in percentage points against annotated smear burden.
- Bias is predicted percentage minus ground-truth percentage.
- This is the key operational metric because it combines detection and classification behavior.

### Quality control

The pipeline reports two result sets:

1. **Raw 12-smear metrics**, preserving the dataset exactly as supplied.
2. **Quality-controlled 11-smear metrics**, excluding only `37erythrocytesIDB2` because its
   circular mask file is absent.

The sample is not silently deleted. Its filename and missing annotation are stored in
`dataset.annotation_warnings` and `evaluation.quality_controlled.excluded_smears`.

## 7. Final A100 results

The final audited experiment is `final/scd_yolo_pipeline/exp/exp_006`.

### Execution environment

- GPU: NVIDIA A100-SXM4-80GB.
- Python 3.11.15.
- PyTorch 2.1.2 + CUDA 12.1.
- Torchvision 0.16.2 + CUDA 12.1.
- Ultralytics 8.4.115.
- OpenCV 4.11.0.
- NumPy 1.26.4 and pandas 2.3.3.
- Twelve smears, both model outputs, CSV tables, and all visual artifacts completed in 53.1 seconds.
- The 53.1-second number includes loading, both models, evaluation, CSV writing, and figure
  generation; do not present it as pure neural-network inference latency.

### Raw 12-smear detection

| Metric | Value |
|---|---:|
| Ground-truth instances | 267 |
| Predicted instances | 452 |
| Matched instances | 259 |
| Precision | 57.30% |
| Recall | 97.00% |
| F1 | 72.04% |
| Mean matched-mask IoU | 93.29% |

The detector reproduces the notebook's documented approximately 0.719 F1 and 0.932 mIoU,
supporting fidelity of the Python conversion.

### Matched-cell classification ablation, all 12 smears

| Metric | InceptionV3 output | YOLO output |
|---|---:|---:|
| Matched cells | 259 | 259 |
| Accuracy | 84.17% | **96.14%** |
| Macro-F1 | 83.21% | **95.65%** |
| Circular F1 | 87.23% | **97.11%** |
| Elongated F1 | 79.19% | **94.19%** |
| Sickle/elongated recall | 93.98% | **97.59%** |

Confusion matrices, with rows as ground truth and columns as prediction:

- YOLO: `[[168, 8], [2, 81]]`
- InceptionV3: `[[140, 36], [5, 78]]`

### End-to-end burden, all 12 smears

| Metric | InceptionV3 output | YOLO output |
|---|---:|---:|
| Burden MAE | 28.43 points | **11.99 points** |
| Burden bias | +18.37 points | **−3.40 points** |
| Matched-only burden MAE | 13.05 points | **2.23 points** |
| Unmatched predictions called sickle | 120 | **46** |

### Quality-controlled result, 11 complete smears

| Metric | Value |
|---|---:|
| Ground-truth instances | 258 |
| Predicted instances | 389 |
| Matched instances | 252 |
| Detection precision | 64.78% |
| Detection recall | 97.67% |
| Detection F1 | 77.90% |
| Mean matched-mask IoU | 93.34% |
| YOLO matched-cell accuracy | 96.03% |
| YOLO matched-cell macro-F1 | 95.39% |
| YOLO sickle recall | 97.37% |
| YOLO burden MAE | **5.57 percentage points** |
| YOLO burden bias | **+3.79 percentage points** |
| Inception burden MAE | 25.53 percentage points |
| Inception burden bias | +25.53 percentage points |

### Result interpretation

- Localization is strong: recall and mask overlap are high.
- Precision is moderate because the model predicts more cells than the available annotations.
- The 193 unmatched raw predictions include false detections and likely real but unlabeled cells;
  the current data cannot cleanly separate the two.
- The isolated-crop InceptionV3 result does not transfer cleanly to detected whole-smear crops.
- InceptionV3 labels 120 of 193 unmatched detections as sickle, while YOLO labels 46 as sickle.
- Visual review of `49erythrocytesIDB2` showed centered source-resolution crops but many visually
  circular cells receiving high Inception sickle probabilities. This supports classifier domain
  shift rather than a coordinate-mapping bug.
- The YOLO output is therefore recommended for this dataset, while Inception output remains an
  auditable secondary result.

## 8. Ready-to-use abstract

### Structured abstract draft

**Background:** Automated blood-smear analysis requires both reliable cell localization and
morphological classification. High accuracy on isolated cells may not transfer to detected cells
inside a complete smear-level pipeline.

**Methods:** We developed a pipeline that square-pads whole-smear images, performs 1024×1024
YOLO11 instance segmentation, maps detections back to source resolution, creates padded cell crops,
and predicts circular versus elongated morphology. YOLO11-seg, fine-tuned SAM, zero-shot SAM, and
Moondream-based localization were compared on held-out pixel masks. InceptionV3, ResNet50,
MaxViT-Tiny, MobileNetV3-SVM, and simulated quantum/classical kernel models were compared on a
grouped crop split. The final pipeline retained both InceptionV3 and YOLO morphology outputs and
was evaluated using mask IoU matching, matched-cell classification, and per-smear sickle-burden
error.

**Results:** In development benchmarking, YOLO11-seg achieved F1 0.734, recall 0.983, and mean
matched IoU 0.923, outperforming the alternative segmenters. The final 12-smear audit reproduced
strong localization with 0.970 recall and 0.933 mean IoU. On 259 matched cells, YOLO morphology
labels achieved 96.14% accuracy and 95.65% macro-F1, compared with 84.17% and 83.21% for
InceptionV3. One smear lacked a circular-cell annotation mask. Across the remaining 11 complete
smears, the YOLO-primary pipeline achieved a burden MAE of 5.57 percentage points and bias of +3.79
points.

**Conclusion:** Whole-pipeline auditing revealed that the segmentation model's class output
generalized better to detected smear cells than the isolated-crop classifier, despite the latter's
near-perfect internal benchmark. The results support YOLO-primary morphology estimation for this
experimental dataset and demonstrate the importance of source-resolution cropping, annotation
auditing, and end-to-end evaluation. External, independently annotated smear-level validation is
required before clinical use.

## 9. Poster content

### Suggested poster layout

Use a three-column poster.

#### Column 1: Why and what data

**Background**

Manual blood-smear review requires locating many cells and assessing their morphology. We aim to
automatically segment individual RBCs, classify circular versus elongated morphology, and estimate
the fraction of sickle-like cells per smear.

**Dataset**

- Cuban and Ugandan microscopy data.
- 78 usable Cuban ground-truth mask smears with 2,118 annotated instances.
- 260 additional smears pseudo-labeled by an audited SAM procedure.
- 1,686 classification crops: 202 circular and 1,484 elongated.
- Smear/source-grouped data splitting where identifiers were available.

**Model comparison**

Show the segmentation and classifier comparison tables or condensed bar charts.

#### Column 2: Pipeline

Place one large horizontal flow diagram:

1. Original smear.
2. 1024×1024 padded canvas.
3. YOLO mask overlay.
4. Source-resolution crop grid.
5. Final morphology overlay and burden percentage.

Add this caption:

> Cell boxes are mapped from the 1024-pixel inference canvas back to the original image before
> cropping. This preserves morphology detail and avoids classifying low-resolution canvas crops.

Add a small methods box:

- YOLO confidence 0.25.
- Mask IoU match threshold 0.50.
- Crop padding 15%.
- Inception threshold 0.5044.
- Final recommended class source: YOLO; Inception retained as an ablation.

#### Column 3: Results and conclusion

Use four large headline values:

- **97.7%** detection recall, QC set.
- **93.3%** mean matched-mask IoU, QC set.
- **96.0%** matched-cell accuracy, YOLO/QC set.
- **5.57 points** per-smear burden MAE, YOLO/QC set.

Show the matched-cell ablation:

- YOLO macro-F1: 95.65% raw.
- Inception macro-F1: 83.21% raw.
- YOLO burden MAE: 5.57 points QC.
- Inception burden MAE: 25.53 points QC.

**Conclusion box**

> YOLO provided high-recall, high-overlap cell segmentation and its morphology output was more
> robust on detected whole-smear cells than the isolated-crop InceptionV3 classifier. End-to-end
> auditing and annotation quality control were essential: one missing class mask substantially
> distorted the raw aggregate. The system is promising for research use, but independent external
> smear-level validation is required before clinical deployment.

### Poster figures to include

1. One representative five-stage pipeline sequence from `exp_006/images/<smear>/`.
2. Detection metric bar chart from `exp_006/images/evaluation_scores.png`.
3. YOLO-versus-Inception matched-cell confusion matrices.
4. Ground-truth versus predicted burden scatter plot.
5. Optional small data composition diagram.

Do not overcrowd the poster with code, every experiment number, or all per-smear rows.

## 10. Paper/report structure

### 1. Introduction

- Explain sickle-like RBC morphology and the need for quantitative smear analysis.
- Distinguish isolated-cell classification from whole-smear analysis.
- State the three tasks: segmentation, classification, and burden estimation.
- End with the research questions and contributions listed above.

### 2. Related work

Organize by method, not by notebook:

- CNN-based red blood cell morphology classification.
- Instance segmentation with YOLO and SAM.
- Vision-language detection with Moondream.
- Deep features with classical SVMs.
- Quantum-kernel classification as an exploratory comparison.
- Domain shift between isolated object crops and detector-generated crops.

Add verified citations from the original papers and dataset documentation. Do not use this guide as
a bibliographic source.

### 3. Data and preprocessing

- Describe Cuba and Uganda components and counts.
- Explain pixel masks, connected-component conversion, and SAM pseudo-label calibration.
- Describe grouped splits and class imbalance.
- State the missing-mask anomaly and smear-ID limitation.
- Include a table of all splits and class counts.

### 4. Methods

Recommended subsections:

1. Segmentation candidate models.
2. Pseudo-label generation and quality audit.
3. Classification candidate models.
4. Final source-resolution crop pipeline.
5. Selected inference configurations: Inception-compatible and YOLO-primary.
6. Evaluation metrics and mask matching.
7. Reproducibility and experiment tracking.

### 5. Results

Present in this order:

1. Development segmentation comparison.
2. Internal crop-classifier comparison.
3. Final 12-smear detection reproduction.
4. Matched-cell YOLO versus Inception ablation.
5. Per-smear burden results.
6. Quality-controlled 11-smear result.
7. Runtime and artifact generation.

Always label a table as **development**, **raw final**, or **quality-controlled final**.

### 6. Discussion

Main discussion points:

- YOLO recall and mask overlap are strong, but annotation incompleteness complicates precision.
- Near-perfect internal crop scores overstated Inception performance in the whole-smear pipeline.
- Source-resolution mapping was correct; the remaining Inception errors indicate domain shift.
- YOLO morphology labels benefit from joint spatial/contextual learning on smear images.
- Burden MAE is more operationally meaningful than matched-cell accuracy alone.
- Explicit annotation auditing prevented one malformed sample from dominating interpretation.
- The clean 11-smear set is too small for narrow confidence intervals or deployment claims.

### 7. Limitations

Include all of the following:

- Only 12 final test smears, with 11 fully annotated.
- Only Cuban smears have the masks required for final segmentation evaluation.
- One test smear lacks `mask-circular.jpg`.
- Unmatched detections cannot be reliably separated into false positives and unlabeled real cells.
- Classifier crop files lack trustworthy smear IDs, so overlap with final mask smears cannot be
  excluded.
- The elongated label is a morphological proxy and not a patient-level diagnosis.
- No calibration analysis, confidence intervals, prospective study, pathologist comparison, or
  clinical decision threshold was performed.
- The QSVM was simulated noiselessly and is not a quantum-hardware result.
- The A100 runtime includes artifact generation and is not a pure inference benchmark.

### 8. Conclusion

Suggested conclusion paragraph:

> This work demonstrates a reproducible end-to-end pipeline for segmenting red blood cells,
> assigning circular or elongated morphology, and estimating smear-level sickle-cell burden.
> YOLO11 provided the strongest segmentation performance and, during final pipeline auditing, its
> morphology labels substantially outperformed the separately trained InceptionV3 classifier on
> detector-generated cells. On 11 fully annotated held-out smears, the recommended configuration
> achieved 97.67% detection recall, 93.34% mean matched-mask IoU, 96.03% matched-cell accuracy, and
> 5.57-percentage-point burden MAE. These findings demonstrate the value of whole-pipeline and
> data-quality auditing, while the small, single-population mask set and uncertain split overlap
> require independent validation before clinical use.

## 11. Presentation plan

### Slide 1 — Title and one-line objective

Say: “Our goal is to turn a raw blood-smear image into segmented cells, morphology labels, and a
smear-level sickle-cell percentage.”

### Slide 2 — Why this is difficult

- Many cells per image.
- Touching cells and variable staining.
- Severe crop-class imbalance.
- Detection errors propagate into burden estimation.
- Isolated-cell accuracy may not transfer to full-smear crops.

### Slide 3 — Data

Show Cuba/Uganda composition, 2,118 ground-truth instances, 22,345 segmentation instances after
pseudo-labeling, and 1,686 classifier crops. Mention grouped splitting and the annotation caveat.

### Slide 4 — Segmentation model comparison

Show a bar chart of F1 and recall. State that YOLO11-seg won with development F1 0.734, recall
0.983, IoU 0.923, and much lower latency than SAM.

### Slide 5 — Classification comparison

Show the candidate table. State that InceptionV3 and ResNet50 were nearly perfect internally, and
that InceptionV3 was selected using validation macro-F1/AUC and frozen threshold 0.5044.

### Slide 6 — Full inference pipeline

Animate or reveal the five saved stage images. Emphasize source-resolution cropping and the burden
formula.

### Slide 7 — Final detection results

Show raw and QC detection results. Explain why high recall/IoU and moderate precision can coexist
when annotations are incomplete.

### Slide 8 — The important audit result

Use a large two-column comparison:

- Inception: 84.17% matched accuracy, 28.43-point raw burden MAE.
- YOLO: 96.14% matched accuracy, 11.99-point raw burden MAE.

Say: “The model that was nearly perfect on isolated crops did not transfer perfectly to cells
generated by the detector. This was the most important end-to-end finding.”

### Slide 9 — Data-quality correction

Show the missing `mask-circular.jpg` issue and the 11-smear QC result: detection F1 77.90%, matched
accuracy 96.03%, and burden MAE 5.57 points. Clarify that both raw and QC results are reported.

### Slide 10 — Demo/reproducibility

Show the CLI command, experiment folder, five stages, JSON/CSV outputs, checkpoint hashes, and
locked environment.

### Slide 11 — Limitations and next steps

- Repair/re-annotate missing and incomplete masks.
- Obtain an independent external smear-level test set.
- Preserve patient/smear IDs and use group-aware splitting.
- Retrain or adapt InceptionV3 on detector-generated crops.
- Calibrate confidence and burden estimates.
- Compare against blinded expert/pathologist counts.

### Slide 12 — Take-home message

> The pipeline localizes cells reliably and estimates burden well on the quality-controlled subset,
> but the key lesson is methodological: strong component scores must be verified end-to-end, and
> data-quality defects must be made visible.

## 12. Figure captions

### Pipeline figure

**Figure 1. End-to-end smear analysis pipeline.** The source smear is square-padded and resized to
1024×1024 for YOLO11 instance segmentation. Detected boxes are mapped back to source coordinates,
expanded by 15%, and cropped from the original-resolution image. Both YOLO and InceptionV3
morphology predictions are recorded, and the selected elongated count is divided by all detected
cells to estimate smear-level sickle burden.

### Segmentation comparison

**Figure 2. Segmentation model comparison on 17 held-out annotated smears.** YOLO11-seg achieved
the highest F1@0.50 and recall while requiring substantially less time per smear than the SAM
alternatives. Moondream produced box-derived elliptical masks and had very low recall.

### Classification confusion matrices

**Figure 3. Matched-cell classification in the final 12-smear audit.** YOLO morphology predictions
produced 168 true circular, 81 true elongated, 8 circular-to-elongated errors, and 2
elongated-to-circular errors. InceptionV3 showed substantially more circular-to-elongated errors,
consistent with overestimation of sickle burden.

### Burden scatter plot

**Figure 4. Predicted versus annotated elongated-cell percentage per smear.** The identity line
indicates perfect agreement. The malformed `37erythrocytesIDB2` annotation is retained in the raw
analysis and separately excluded from the quality-controlled summary because its circular mask is
missing.

### Qualitative stages

**Figure 5. Qualitative inference artifacts.** From left to right: source smear, padded inference
canvas, YOLO mask overlay, source-resolution cell crops, and final morphology overlay. These
artifacts are saved for every experiment to support error analysis and reproducibility.

## 13. Reproducibility and inference commands

From `final/scd_yolo_pipeline`:

~~~bash
uv sync --frozen --group dev

# Recommended final configuration
uv run scd-yolo evaluate --classification-source yolo

# Exact original notebook-compatible configuration
uv run scd-yolo evaluate --classification-source inception

# One unseen smear
uv run scd-yolo image /path/to/smear.jpg --classification-source yolo

# Quick smoke test
uv run scd-yolo evaluate --limit 1 --device cpu
~~~

The package automatically discovers:

- `final/yolo_seg_finetuned.pt`;
- `final/inceptionv3_best.pt`;
- `final/best_model.json`;
- `data/SCD_Final.zip`.

Checkpoint SHA-256 hashes used by `exp_006`:

- YOLO: `c57091e25dc2b90e8bb8310bbbbc2aa829d2cc65c0d10bce4c7acd649be55965`
- InceptionV3: `e5b81fec82351d34a05e8f0b4b3afa5da8ac66e11dadef60e1a0bf901b0d1d15`

The checkpoints and data are intentionally excluded from Git because of size and data-management
concerns. Store them in approved artifact/object storage and document retrieval instructions for
reviewers.

## 14. Claims you can and cannot make

### Defensible claims

- YOLO11-seg was the strongest tested segmenter on the development benchmark.
- The final Python pipeline reproduces the notebook's detection behavior.
- YOLO morphology labels outperformed InceptionV3 on matched cells in this final held-out set.
- Source-resolution cropping avoids a known resolution-loss failure mode.
- The QC subset achieved a 5.57-percentage-point burden MAE.
- The package is reproducible at the software/configuration level through `uv.lock`, hashes, and
  immutable experiment directories.

### Claims to avoid

- “The model diagnoses sickle cell disease.”
- “The model is clinically validated.”
- “The model generalizes to Uganda or all populations.”
- “Detection precision is definitely low because YOLO produces false positives.”
- “InceptionV3 achieved 99.9% independent smear-level accuracy.”
- “The QSVM was run on a quantum computer.”
- “The system processes a smear in 53.1/12 seconds of pure inference.”
- “The quality-controlled result is based on a large test set.”

## 15. Likely questions and answers

### Why use YOLO labels if the project was designed as a two-stage pipeline?

The original two-stage configuration remains reproducible, but end-to-end auditing showed a clear
domain-shift failure: InceptionV3 was trained on isolated crops and over-predicted elongated cells
on detector-generated crops. The YOLO checkpoint had been fine-tuned with morphology classes and
performed substantially better on matched whole-smear cells. Reporting this ablation is more
scientifically honest than keeping the weaker configuration solely because it matched the original
design.

### Why is precision lower than recall?

The detector produces 452 instances against 267 supplied raw annotations. Some predictions are
false positives, but the masks are also incomplete, so not every unmatched cell can be treated as
wrong. High matched-mask IoU shows that the cells that do match are localized accurately.

### Why report both raw and QC results?

Raw metrics preserve the supplied test set. QC metrics answer what performance looks like on files
with both required class masks. Reporting both prevents hidden post-hoc exclusion while avoiding a
known missing file dominating interpretation.

### Why is internal Inception accuracy so high but final performance lower?

The internal test uses isolated-cell images that resemble its training data. Whole-smear inference
uses detector-generated crops with different scale, context, blur, and staining. In addition, the
Cuba crop files lack reliable smear IDs, so internal independence cannot be fully verified.

### Is the 5.57-point burden MAE clinically acceptable?

The study does not establish a clinical acceptance threshold. The value is promising as an
experimental result, but it must be compared with expert variability and validated on a larger,
independent, prospectively annotated set.

### Why use macro-F1?

The crop dataset is highly imbalanced toward elongated cells. Macro-F1 gives circular and elongated
classes equal influence and is less flattering to majority-class behavior than accuracy alone.

### Why was SAM not selected?

Fine-tuned SAM and zero-shot SAM achieved strong recall but lower precision/F1 and much higher
latency than YOLO on the same held-out masks.

### What should be improved next?

1. Repair and independently audit the mask annotations.
2. Build a larger external smear-level test split with patient/source IDs.
3. Train the classifier on detector-generated crops or use joint multi-task training.
4. Calibrate probabilities and burden estimates.
5. Quantify uncertainty with confidence intervals and repeated grouped splits.
6. Compare with blinded expert counts and report inter-rater variability.

## 16. Final writing checklist

- [ ] Add verified dataset citation, license, and acquisition details.
- [ ] Add authors, affiliations, acknowledgments, and funding information.
- [ ] State whether figures use raw or quality-controlled results.
- [ ] Keep development and final-evaluation tables clearly separated.
- [ ] Explain the missing circular mask explicitly.
- [ ] Mention possible crop-level leakage and the absence of cross-population mask validation.
- [ ] Present YOLO-versus-Inception as an end-to-end ablation.
- [ ] Label elongated cells as morphology/sickle-like rather than a complete diagnosis.
- [ ] Include software versions and checkpoint hashes in the paper supplement.
- [ ] Archive the exact checkpoints outside Git and provide approved retrieval instructions.
- [ ] Use the five stage images to show qualitative success and failure cases.
- [ ] End with external validation—not deployment—as the next step.

## 17. Source-of-truth files

- Final metrics: `final/scd_yolo_pipeline/exp/exp_006/summary.json`
- Per-smear results: `final/scd_yolo_pipeline/exp/exp_006/tables/per_smear.csv`
- Matched-cell results: `final/scd_yolo_pipeline/exp/exp_006/tables/matched_cells.csv`
- Final pipeline analysis: `final/scd_yolo_pipeline/ANALYSIS.md`
- Pipeline usage: `final/scd_yolo_pipeline/README.md`
- Final model metadata: `final/best_model.json`
- Original final notebook: `final/scd_final_step4_yolo.ipynb`
- Development data preparation: `00_setup_and_data.ipynb`
- Development segmentation comparison: `04_segmentation_comparison.ipynb`
- Development classifier comparison: `11_classifier_comparison.ipynb`

When numbers disagree between an older development notebook and `exp_006`, label them by stage and
use `exp_006` as the source of truth for the final packaged pipeline.
