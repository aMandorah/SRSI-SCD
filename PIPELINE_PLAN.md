# SRSI-SCD: Evaluation Note and Full-Smear Pipeline Plan

## Project objective

The current SRSI-SCD model classifies a pre-cropped red blood cell as one of two
classes:

- **Circular**
- **Elongated**

InceptionV3 is the selected classifier. The next engineering objective is to
extend this cell-level classifier into a pipeline that can process a full
microscope field from a prepared peripheral blood smear.

The intended end-to-end flow is:

```text
Full microscope field
        |
        v
Image quality control
        |
        v
RBC detection / instance segmentation
        |
        v
Individual cell crops
        |
        v
InceptionV3: Circular vs Elongated
        |
        +--------> Grad-CAM explanation per cell
        |
        v
Field- and smear-level counts, percentages, and report
```

## Current scope and constraints

For the current engineering phase:

- Do **not** add, remove, relabel, move, or otherwise modify the datasets.
- Keep the medical classification task binary: **Circular vs Elongated**.
- Do not introduce an `Other` medical class.
- Preserve the existing notebooks as records of the experiments already run.
- Engineer the pipeline as modular components so that additional datasets can
  be connected later without rewriting the models.
- Treat unusable images as a technical quality-control issue, not as a third
  cell class.

The Uganda circular images were intentionally excluded because their image
quality was unsuitable. Consequently, Uganda can currently measure elongated
cell sensitivity/recall, but it cannot independently measure binary accuracy,
specificity, or false-positive rate.

---

## Evaluation mistake in the merged notebook

### Location

The issue is in:

`merged_cuba_uganda_train_test_kagglehub.ipynb`

It appears in the fifth code cell, identified in the notebook as `cell-5`. The
cell begins with:

```python
from tensorflow.keras import applications as A
```

The relevant statements are:

```python
hist = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=CFG["epochs"],
    verbose=2,
)

te_pred = model.predict(test_ds, verbose=0)
```

### What happened

The same `test_ds` was used for two different purposes:

1. It was supplied to `model.fit()` as the validation dataset and observed
   after every training epoch.
2. It was used again after training to calculate the reported test results.

The same dataset was also reused while comparing several architectures. If
validation curves or reported results influenced the selection of InceptionV3,
the test set indirectly participated in model selection.

### Why this matters

A test set is intended to estimate performance on unseen data after training,
model selection, threshold selection, and hyperparameter decisions are
complete. Once it is used as validation data, it is no longer an untouched
independent test set.

This does **not** mean that the trained model is unusable or that the experiment
has no value. It means that the reported `test_acc` should be treated as a
validation/model-comparison result rather than as a final unbiased estimate of
generalization.

### Correct evaluation structure for a future training run

```text
Training set
    Used to update model weights

Validation set
    Used to select the architecture, epoch, threshold, and hyperparameters

Test set
    Used once after all modeling decisions are frozen
```

The correction belongs in a future experiment. No dataset files need to be
changed for the present pipeline-engineering phase.

---

## Interpretation of the current cross-dataset experiment

The current class/source coverage is:

| Dataset | Circular | Elongated |
|---|---:|---:|
| Cuba | Available | Available |
| Uganda | Excluded for quality | Available |

This setup exposes the elongated class to more than one acquisition domain,
which is useful. However, the Uganda evaluation is positive-only, so its
reported score is:

```text
Uganda elongated recall =
correctly predicted Uganda elongated cells / all Uganda elongated cells
```

It is not Uganda binary accuracy. The current model should therefore be
described as a **multi-source binary cell classifier with positive-class
external evaluation**, rather than as a fully validated domain-general model.

The engineering design should nevertheless be dataset-adaptive: source-specific
loading and metadata should be isolated behind adapters, while the segmentation,
classification, explanation, and reporting stages use one standard internal
format.

---

## Proposed pipeline architecture

### 1. Full-field input

The pipeline accepts one microscope field at a time, together with optional
metadata:

```text
image
dataset/source identifier
slide identifier
field identifier
patient identifier, when available
magnification and acquisition metadata, when available
```

Raw images remain unchanged. Dataset adapters translate existing folder and
filename conventions into this standard record.

### 2. Image quality control

Before analyzing cells, the pipeline checks whether the microscope field is
usable. Initial engineering checks can include:

- File readability and supported format
- Minimum image dimensions
- Excessive blur or loss of focus
- Severe overexposure or underexposure
- Very low contrast
- Absence of a plausible blood-smear region

A failed field returns `ungradable` with reasons. It is not classified as
Circular, Elongated, or a medical `Other` class.

### 3. RBC detection or instance segmentation

The first vision model should locate **all analyzable RBCs**, not only suspected
elongated cells. This prevents the first stage from pre-deciding the output of
InceptionV3.

Two backends should share one interface:

- **YOLO segmentation** for a trainable, repeatable, and deployable RBC instance
  segmenter
- **SAM3** for prototyping, annotation assistance, and comparison

Expected output for each detected cell:

```json
{
  "cell_id": "field_001_cell_0001",
  "bounding_box": [120, 85, 196, 164],
  "mask": "optional encoded instance mask",
  "detection_confidence": 0.97
}
```

The pipeline should support bounding boxes even when masks are unavailable.
Instance masks are preferable when cells touch or overlap and when explicit
shape measurements are required.

### 4. Cell crop extraction

Each detected RBC is converted to an InceptionV3 input:

- Add a configurable margin around the detected cell.
- Preserve aspect ratio.
- Pad rather than distort when making a square crop.
- Optionally suppress the background using the instance mask.
- Apply the exact preprocessing expected by InceptionV3.
- Keep the transformation information needed to map Grad-CAM back to the
  original microscope field.

The crop stage should also flag cells cut off by the image border or too heavily
overlapped for reliable classification.

### 5. InceptionV3 binary classification

Every accepted crop is classified as:

```text
Circular
Elongated
```

The classifier output should preserve the probability, chosen threshold, model
version, and preprocessing version:

```json
{
  "predicted_class": "Elongated",
  "elongated_probability": 0.91,
  "classification_threshold": 0.50,
  "model_version": "inceptionv3-v1"
}
```

Low-confidence cases can be marked for review without adding a third learned
class. For example:

```text
probability >= upper threshold  -> Elongated
probability <= lower threshold  -> Circular
between thresholds              -> Review / uncertain
```

The review state is an operational decision, not a new medical label.

### 6. Grad-CAM explanation

Grad-CAM is calculated **after** InceptionV3 produces a prediction. It is not
another input to the classifier.

For each accepted cell, save:

- The original cell crop
- The Grad-CAM heatmap
- A heatmap overlay
- The predicted class and probability
- The location of the cell in the full microscope field

The explanation module should be optional so that routine batch inference can
run without Grad-CAM when speed is more important.

Grad-CAM should be inspected across:

- Correct Circular predictions
- Correct Elongated predictions
- False positives
- False negatives
- Low-confidence cases
- Images from each source dataset

### 7. Field- and smear-level aggregation

The pipeline should not reduce a full field to a single cell prediction. It
should aggregate all accepted cells:

```text
total detected RBCs
total analyzable RBCs
Circular count
Elongated count
review/uncertain count
rejected cell count
elongated percentage among analyzable cells
```

The central quantitative output is:

```text
elongated percentage =
Elongated cells / (Circular cells + Elongated cells) * 100
```

If several fields belong to the same smear, aggregation should occur across the
fields while retaining individual field results for auditing.

The research output should initially be described as automated cell morphology
classification and sickled-cell quantification, not as a standalone diagnosis
of sickle cell disease.

### 8. Report generation

Each run should produce both:

- A machine-readable JSON or CSV result
- A visual report with the full field, detected cells, class colors,
  probabilities, and selected Grad-CAM examples

Recommended overlay convention:

```text
Green  = Circular
Red    = Elongated
Yellow = Review / uncertain
Gray   = Rejected or ungradable
```

---

## Engineering modules

A proposed implementation layout is:

```text
srsi_scd/
    config.py
    schemas.py

    adapters/
        base.py
        cuba.py
        uganda.py

    quality/
        field_quality.py
        cell_quality.py

    segmentation/
        base.py
        yolo_backend.py
        sam3_backend.py

    classification/
        inceptionv3.py
        preprocessing.py

    explainability/
        gradcam.py

    pipeline/
        full_field.py
        aggregation.py

    reporting/
        overlays.py
        export.py

tests/
    test_adapters.py
    test_crop_mapping.py
    test_aggregation.py
    test_pipeline_contracts.py
```

The segmentation backends should return the same schema. This allows YOLO and
SAM3 to be exchanged without changing InceptionV3, Grad-CAM, or reporting code.
Similarly, dataset-specific path logic should remain inside the adapters.

---

## Implementation sequence

### Phase 1: Define interfaces and preserve the existing classifier

- Define standard schemas for fields, detected cells, predictions, and reports.
- Wrap the existing InceptionV3 model behind one inference interface.
- Implement crop preparation and coordinate mapping.
- Add unit tests for preprocessing and aggregation.
- Do not retrain models or modify datasets.

**Deliverable:** a pipeline that can accept already-defined cell boxes and run
classification, Grad-CAM, and aggregation.

### Phase 2: Add full-field segmentation backends

- Implement a common segmentation interface.
- Add a YOLO backend.
- Add a SAM3 backend.
- Visualize detections and masks on full microscope fields.
- Keep model-specific settings in configuration files.

**Deliverable:** a full field can be converted automatically into InceptionV3
cell crops.

### Phase 3: Complete end-to-end inference

- Connect quality control, segmentation, cropping, classification, Grad-CAM,
  aggregation, and reporting.
- Add batch processing for multiple fields.
- Record model and configuration versions in every result.
- Add deterministic run settings and structured error handling.

**Deliverable:** one command or notebook entry point produces a complete
field-level report.

### Phase 4: Evaluate the pipeline without changing the raw data

- Measure runtime and failure modes.
- Confirm that coordinates and Grad-CAM overlays map correctly.
- Review detections, crops, and classifications qualitatively.
- Report Uganda performance explicitly as elongated recall.
- Treat existing merged-notebook results as validation/model-comparison
  results.

**Deliverable:** an engineering validation report, not yet a final clinical
performance claim.

### Phase 5: Future model evaluation

This phase occurs only when a new training/evaluation run is approved:

- Create separate training, validation, and untouched test roles.
- Group splits by patient, slide, or original field whenever identifiers exist.
- Select thresholds on validation data only.
- Evaluate the frozen pipeline once on the test set.
- Add a third dataset containing usable Circular and Elongated cells when
  available.
- Perform leave-one-dataset-out evaluation when enough datasets are available.

**Deliverable:** an unbiased estimate of cross-dataset generalization.

---

## Evaluation metrics for the completed pipeline

### Segmentation

- Cell detection recall
- Precision
- Box and mask mAP
- Dice or IoU for masks
- Performance on overlapping and border cells

### Cell classification

- Circular and Elongated sensitivity
- Specificity
- Balanced accuracy
- Macro F1
- Precision-recall AUC
- Confusion matrix
- Calibration and low-confidence review rate

### End-to-end field or smear analysis

- Error in Circular and Elongated cell counts
- Error in elongated-cell percentage
- Fraction of rejected fields and cells
- Agreement with expert cell counts
- Processing time per field

Metrics should be reported both overall and separately for every source dataset.

---

## Immediate next milestone

Without changing any data, the first implementation milestone should be:

> Given a full microscope image and a supplied list of cell bounding boxes,
> extract consistent cell crops, classify each crop with the existing
> InceptionV3 model, generate optional Grad-CAM overlays, and return an
> aggregated field-level report.

This milestone establishes the complete downstream pipeline before choosing or
training the final YOLO/SAM3 segmentation backend.
