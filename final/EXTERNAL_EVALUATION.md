# Nigerian External Evaluation Guide

This document is the external-validation companion to `REPORT_POSTER_GUIDE.md`. It records the
completed Nigerian evaluation, explains how it extends the Cuban/Ugandan development study, and
provides text and tables that can be used in a poster, paper, report, or presentation. The source of
truth is the completed immutable experiment `final/scd_yolo_pipeline/exp/exp_009`.

The central result is strong, but the scope of the claim matters: this is an external **sample-level
discrimination** study using electrophoresis-derived SCD labels. It is not an external cell-level
segmentation or morphology-annotation benchmark.

## 1. Headline finding

The frozen YOLO-primary pipeline transferred successfully to an external Nigerian thin-blood-film
cohort. Across 141 labeled samples, YOLO-derived elongated-cell burden discriminated label-positive
from label-negative samples with:

- AUROC **0.9253** (clustered-bootstrap 95% CI **0.8817–0.9616**);
- average precision **0.9332** (95% CI **0.8814–0.9683**);
- 72 label-positive and 69 label-negative samples;
- 1,969 successfully processed fields and 187,838 segmented/classified cells.

The corresponding InceptionV3 burden was substantially weaker: AUROC **0.6471** (95% CI
**0.5551–0.7326**) and average precision **0.6542** (95% CI **0.5314–0.7643**). This external result
reinforces the internal end-to-end audit: the YOLO morphology output is the recommended primary
output, while InceptionV3 is an informative domain-shift ablation.

### One-sentence paper result

> In an external Nigerian cohort of 141 thin-blood-film samples, the frozen YOLO-derived
> elongated-cell burden achieved an AUROC of 0.925 (95% CI 0.882–0.962) and average precision of
> 0.933 (95% CI 0.881–0.968) for discrimination of electrophoresis-derived SCD status.

### One-sentence poster result

> External Nigerian validation: **AUROC 0.925** and **average precision 0.933** across 141 samples
> using the frozen YOLO elongated-cell burden.

## 2. How this extends the original study

`REPORT_POSTER_GUIDE.md` documents model development and the final mask-based audit using Cuban
smears, with Ugandan images contributing population/domain data during development. At that stage,
independent external smear-level validation was still a required next step.

Experiment `exp_009` fills that gap at the sample level:

| Study stage | Population/data | Available reference | Supported evaluation |
|---|---|---|---|
| Development and internal audit | Cuba and Uganda | Cuban cell masks and morphology classes; development crops | Segmentation, matched-cell classification, and burden error on Cuban masks |
| External evaluation | Nigeria | Sample-level SCD status from hemoglobin electrophoresis | Sample-level discrimination from aggregated elongated-cell burden |

The Nigerian release does not contain public per-cell masks or circular/elongated annotations.
Consequently, it cannot establish external segmentation precision, recall, F1, mask IoU, cell-level
morphology accuracy, or burden error against an expert cell count. Those remain future validation
requirements.

The updated overall project message is therefore:

> The pipeline has strong internal cell-level performance on the quality-controlled Cuban mask set
> and strong external sample-level discrimination in a Nigerian cohort. External independently
> annotated cell-mask validation and prospective clinical evaluation are still required.

## 3. External dataset and provenance

The external cohort is the public **Digitized Thin Blood Films for Sickle Cell Disease Detection**
dataset from University College Hospital, Ibadan, Nigeria. The official release states that:

- sample SCD status was obtained using hemoglobin electrophoresis;
- images were acquired using a custom brightfield microscope with a 100×/1.4 NA objective, a
  motorized x-y stage, and a color camera;
- z-stacks were projected into a single plane using an Extended Depth of Field algorithm;
- the dataset is released under CC BY-NC-SA 4.0.

Dataset record and required citation:

- P. Manescu, C. Bendkowski, R. Claveau, M. Elmi, B. J. Brown, V. Pawar, M. Shaw, and D.
  Fernandez-Reyes, *A weakly supervised deep learning approach for detecting malaria and sickle
  cells in blood films*, MICCAI 2020.
- UCL Research Data Repository, [Digitized Thin Blood Films for Sickle Cell Disease
  Detection](https://rdr.ucl.ac.uk/articles/dataset/Digitized_Thin_Blood_Films_for_Sickle_Cell_Disease_Detection/12407567),
  DOI: `10.5522/04/12407567.v1`.

Use the original publication and repository record in the final bibliography; do not cite this
guide as the dataset source.

### Cohort audit

| Item | Count |
|---|---:|
| Rows in supplied label file | 162 |
| Unique labeled sample IDs | 156 |
| Duplicate label rows | 6 |
| Labeled samples matched to image directories | 141 |
| Labeled IDs without matched released images | 15 |
| Label-positive samples evaluated | 72 |
| Label-negative samples evaluated | 69 |
| Repeat-family groups used for bootstrap clustering | 135 |
| Image fields discovered | 1,971 |
| Image fields successfully processed | 1,969 |
| Fields excluded for zero detections | 2 |
| Samples passing QC | 141 |

The 15 unmatched label IDs are retained in the dataset audit inside `summary.json`. No evaluated
sample failed the prespecified sample-level QC rule. Two individual fields, one from `080119-07`
and one from `280120-29`, were excluded because the pipeline returned zero detections. Technical
field coverage was therefore 1,969/1,971, or **99.90%**.

Avoid calling the 141 records “patients” unless patient/sample identity is verified from the
original study. This analysis conservatively calls them samples.

## 4. Frozen external-evaluation protocol

No Nigerian image or label was used to retrain, fine-tune, or select the supplied checkpoints in
this external run. The packaged pipeline used the same frozen configuration documented in the
internal audit:

1. Bottom/right square-pad each field and resize it to 1024×1024.
2. Run the fine-tuned YOLO11 segmentation checkpoint at confidence 0.25.
3. Map detections back to the source image and generate 15%-padded source-resolution crops.
4. Record both the YOLO circular/elongated class and the frozen InceptionV3 prediction for every
   detected cell.
5. Aggregate each output independently across all valid fields belonging to a sample.
6. Define sample elongated burden as:

   `100 × elongated predictions / all detected cells`

7. Use the continuous sample burden to discriminate the supplied binary electrophoresis-derived
   label.

The primary endpoint was YOLO-burden AUROC. Average precision was included because it summarizes
precision-recall ranking; the positive prevalence in the evaluated cohort was 72/141, or 51.1%.
Inception-burden AUROC and average precision were prespecified comparative outputs.

### Quality control

A sample was eligible for the discrimination analysis when it had:

- at least three successfully processed fields; and
- at least 100 detected cells in aggregate.

All 141 matched samples passed these criteria. Processing failures and zero-detection fields were
recorded rather than silently discarded.

### Confidence intervals

Uncertainty was estimated with 2,000 bootstrap replicates. Resampling was clustered by
`repeat_family`, which removes a trailing repeat suffix such as `r1` or `r2` from related sample
IDs. This prevents repeated records from being treated as fully independent during interval
estimation. The analysis contained 135 such groups.

### What was not estimated

- No diagnostic threshold was selected on the Nigerian cohort.
- No sensitivity, specificity, positive predictive value, negative predictive value, or accuracy
  at a clinical operating point is reported.
- No probability calibration analysis was performed.
- No cell-level metrics were calculated because cell-level Nigerian ground truth is unavailable.
- No segmentation claims are inferred from the 99.90% technical processing rate.

## 5. Complete external results

### Primary discrimination results

| Aggregated morphology output | AUROC (95% CI) | Average precision (95% CI) | Samples |
|---|---:|---:|---:|
| **YOLO elongated burden** | **0.9253 (0.8817–0.9616)** | **0.9332 (0.8814–0.9683)** | 141 |
| InceptionV3 elongated burden | 0.6471 (0.5551–0.7326) | 0.6542 (0.5314–0.7643) | 141 |

Both model outputs were calculated from the same detected-cell population. The difference therefore
reflects morphology-label behavior rather than a different segmentation run.

### Descriptive sample-level burden

| Supplied sample label | Samples | Fields | Detected cells | YOLO burden, mean ± SD | YOLO burden, median | Inception burden, mean ± SD | Inception burden, median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Negative (0) | 69 | 1,144 | 120,099 | 3.18% ± 2.72% | 2.22% | 66.61% ± 21.84% | 73.08% |
| Positive (1) | 72 | 825 | 67,739 | 16.87% ± 12.70% | 12.80% | 76.27% ± 18.51% | 83.54% |

These summaries are descriptive and unadjusted. They show substantially greater separation in the
YOLO burden distributions. InceptionV3 assigned high elongated burdens in both groups, consistent
with systematic over-calling under external image-domain shift.

### Processing summary

| Metric | Value |
|---|---:|
| Successfully processed samples | 141/141 |
| Successfully processed fields | 1,969/1,971 (99.90%) |
| Detected and classified cells | 187,838 |
| Median fields per sample | 10 |
| Median detected cells per sample | 879 |
| Sample detected-cell range | 217–9,767 |
| Excluded fields | 2, both zero detections |
| End-to-end run time | 20 min 11.6 s |
| Hardware | NVIDIA A100-SXM4-80GB |

The elapsed time includes model loading, processing both classifier outputs, stage-image creation,
bootstrap analysis, and artifact writing. It must not be presented as pure inference latency.

## 6. Interpretation

### Main scientific interpretation

The YOLO-primary result is strong evidence that the learned elongated-cell morphology signal
transfers across the Nigerian acquisition domain at the sample-ranking level. Its AUROC confidence
interval remains well above chance, and average precision is high relative to the 51.1% positive
prevalence. The result is especially meaningful because the checkpoints and inference settings were
frozen before this external run.

This evaluation also reproduces the central ablation from the Cuban end-to-end audit. InceptionV3
was nearly perfect on its internal isolated-cell test set, yet it over-predicted elongated cells in
detector-generated whole-smear crops. In Nigeria, its median predicted burden was 73.08% even in
label-negative samples. YOLO, by contrast, produced medians of 2.22% in negative and 12.80% in
positive samples. The likely explanation is crop/acquisition domain shift involving staining,
illumination, scale, context, or image formation—not a failure of the shared detection stage.

The external findings therefore support these conclusions:

- YOLO-derived elongated burden is the primary morphology measure for the present pipeline.
- Joint detection/morphology learning transferred better than the isolated-crop InceptionV3
  classifier.
- End-to-end auditing was necessary; internal isolated-crop accuracy alone would have selected a
  weaker external output.
- Aggregating many cell predictions per sample produced a robust sample-level ranking signal.

### What AUROC 0.925 does and does not mean

An AUROC of 0.925 means that a randomly selected label-positive sample will generally receive a
higher YOLO elongated burden than a randomly selected label-negative sample. It does not mean
92.5% diagnostic accuracy, 92.5% correctly classified cells, or 92.5% segmentation accuracy.

Because no external threshold was frozen or independently tested, this result should be described
as **strong discrimination**, not as a validated diagnostic test. A future threshold analysis must
separate threshold selection from threshold evaluation and should include sensitivity, specificity,
predictive values, calibration, and decision-relevant uncertainty.

### Relationship to the Cuban mask audit

The two evaluations answer complementary questions:

- `exp_006` establishes internal localization quality, matched-cell classification, and burden
  error against Cuban pixel masks.
- `exp_009` tests whether aggregated morphology burden transfers to an independent Nigerian
  sample-label domain.

Do not pool their metrics. The Nigerian AUROC cannot replace Cuban cell-level IoU or accuracy, and
the Cuban burden MAE cannot be assumed for Nigeria without expert Nigerian cell counts.

## 7. Ready-to-use paper text

### Methods paragraph

> We externally evaluated the frozen pipeline using the public Digitized Thin Blood Films for
> Sickle Cell Disease Detection cohort from University College Hospital, Ibadan, Nigeria. The
> release provides thin-film fields grouped by sample and binary SCD status derived from hemoglobin
> electrophoresis, but no public cell-level masks or morphology annotations. Each field was
> processed using the unchanged 1024×1024 YOLO11 instance-segmentation pipeline at confidence 0.25.
> Detected cells were mapped to source resolution, and both YOLO and frozen InceptionV3 circular
> versus elongated predictions were retained. For each sample, elongated burden was calculated as
> the number of elongated predictions divided by all detected cells across valid fields. Samples
> with at least three valid fields and at least 100 detected cells were included. Discrimination of
> electrophoresis-derived status was measured using AUROC and average precision; 95% confidence
> intervals were estimated from 2,000 bootstrap replicates clustered by repeat-family identifier.

### Results paragraph

> All 141 matched samples passed quality control, comprising 72 label-positive and 69
> label-negative samples. The pipeline successfully processed 1,969 of 1,971 fields (99.90%) and
> segmented/classified 187,838 cells; two fields were excluded because no cells were detected.
> YOLO-derived elongated burden achieved an AUROC of 0.925 (95% CI 0.882–0.962) and average
> precision of 0.933 (95% CI 0.881–0.968). Median YOLO burden was 12.80% in label-positive samples
> and 2.22% in label-negative samples. In comparison, InceptionV3 burden achieved an AUROC of 0.647
> (95% CI 0.555–0.733) and average precision of 0.654 (95% CI 0.531–0.764), while assigning high
> median elongated burdens to both positive (83.54%) and negative (73.08%) samples.

### Discussion paragraph

> The strong external YOLO AUROC supports transfer of the pipeline's aggregated morphology signal
> to a Nigerian thin-film acquisition domain. The markedly weaker InceptionV3 result is consistent
> with the whole-pipeline Cuban audit, in which the isolated-crop classifier was sensitive to the
> detector-generated crop domain. This agreement across analyses strengthens the choice of YOLO
> morphology labels as the primary pipeline output. However, the Nigerian dataset supplies only
> sample-level electrophoresis labels; consequently, the study does not establish external
> segmentation accuracy, cell-level morphology accuracy, burden calibration, or performance at a
> clinical decision threshold.

### Revised abstract result and conclusion

Add this to the results section of the abstract in `REPORT_POSTER_GUIDE.md`:

> In external evaluation on 141 Nigerian thin-film samples, YOLO-derived elongated burden achieved
> an AUROC of 0.925 (95% CI 0.882–0.962) and average precision of 0.933 (95% CI 0.881–0.968) for
> electrophoresis-derived SCD status, compared with AUROC 0.647 for InceptionV3 burden.

Replace “external validation is required” in the abstract conclusion with:

> External Nigerian sample-level evaluation supported transfer of the YOLO-derived morphology
> signal; independently annotated external cell masks, prospective evaluation, and threshold
> validation remain required before clinical use.

## 8. Poster integration

Add a clearly labeled **External Nigerian Evaluation** box to the results column. Recommended
headline values:

- **141** externally evaluated samples;
- **187,838** segmented/classified cells;
- **0.925** YOLO burden AUROC;
- **0.933** YOLO burden average precision;
- **99.90%** field processing coverage.

Recommended compact table:

| External output | AUROC | Average precision |
|---|---:|---:|
| **YOLO burden** | **0.925** | **0.933** |
| Inception burden | 0.647 | 0.654 |

Recommended caption below the table:

> External discrimination of electrophoresis-derived SCD status in 141 Nigerian samples. Values
> are sample-level ranking metrics from aggregated elongated-cell burden; the release does not
> provide cell-level ground truth.

### Suggested poster conclusion

> YOLO achieved strong internal cell-level morphology performance and strong external sample-level
> discrimination in a Nigerian cohort. The weaker InceptionV3 transfer result demonstrates that
> near-perfect isolated-crop performance does not guarantee robustness inside a whole-smear
> pipeline. External cell-mask and prospective threshold validation remain necessary before
> clinical use.

### Suggested presentation slide

Use one slide immediately after the internal `exp_006` audit:

**Title:** Does the morphology signal transfer to a third country?

**Left:** a representative Nigerian five-stage sequence from `exp_009/images/`.

**Right:**

- Nigeria: 141 samples, 1,969 fields, 187,838 cells.
- YOLO: AUROC 0.925 (95% CI 0.882–0.962).
- Inception: AUROC 0.647 (95% CI 0.555–0.733).
- Take-home: YOLO morphology transferred; isolated-crop Inception remained domain-sensitive.

Say explicitly: “This is sample-level external discrimination, because the Nigerian release has
electrophoresis labels but no cell masks.”

## 9. Recommended figures and captions

These figures are not yet generated as aggregate plots in `exp_009`; they should be made directly
from `tables/nigeria_samples.csv` without rerunning inference.

### External ROC and precision-recall curves

> **External sample-level discrimination.** Receiver-operating-characteristic and
> precision-recall curves compare continuous YOLO- and Inception-derived elongated-cell burden for
> electrophoresis-derived SCD status in 141 Nigerian samples. Confidence intervals in the text were
> estimated using 2,000 repeat-family-clustered bootstrap replicates.

Do not add a threshold marker unless that threshold is defined as exploratory and is not reported
as independently validated.

### Burden distribution plot

Use violin/box plots with individual sample points, split by supplied label and faceted by model.

> **External elongated-burden distributions.** YOLO-derived burden showed substantially greater
> separation between label-negative and label-positive Nigerian samples than InceptionV3-derived
> burden. Each point represents one sample aggregated over all valid fields; labels were obtained
> from the supplied electrophoresis-derived file.

### Qualitative five-stage sequence

> **Qualitative Nigerian external inference.** From left to right: source thin-film field, padded
> inference canvas, YOLO segmentation overlay, source-resolution cell crops, and morphology
> classification overlay. Qualitative outputs demonstrate technical transfer but are not a
> substitute for unavailable Nigerian cell-level annotations.

Choose representative examples using a documented rule, such as samples near the group medians,
rather than selecting only the most visually convincing fields.

## 10. Limitations

Include all of the following in the paper or supplement:

- The public Nigerian release provides sample labels but no cell masks or expert circular/elongated
  annotations.
- External cell detection, mask overlap, and morphology accuracy therefore cannot be quantified.
- The binary label is used as supplied from hemoglobin-electrophoresis-derived status; elongated
  burden is a morphology proxy and not equivalent to a complete clinical diagnosis.
- A continuous ranking endpoint was evaluated. No clinical operating threshold, calibration
  analysis, decision-curve analysis, or prospective workflow was assessed.
- Sample counts are not asserted to equal unique patient counts.
- Related repeat IDs exist. Confidence intervals were clustered by 135 repeat-family groups, but
  richer patient/acquisition identifiers were unavailable to this pipeline.
- Fifteen labeled IDs did not match released image directories and were not evaluated; six rows in
  the label file were duplicates.
- Two of 1,971 fields returned zero detections. All failures were retained in the audit.
- Field and cell counts differed between label groups. Primary evaluation was sample-level so that
  samples, rather than individual cells, were the units of analysis.
- The study evaluates a single external Nigerian dataset and acquisition system; it does not prove
  universal geographic, laboratory, device, or clinical generalization.
- The A100 elapsed time includes analysis and artifact generation and is not a pure inference
  benchmark.

## 11. Claims you can and cannot make

### Defensible claims

- The frozen YOLO burden achieved strong sample-level external discrimination in the evaluated
  Nigerian cohort.
- YOLO burden substantially outperformed InceptionV3 burden on the same external detections.
- All 141 matched samples met prespecified QC, and 99.90% of image fields were processed
  successfully.
- The external result supports cross-domain transfer of an aggregated morphology signal.
- The Nigerian and Cuban findings consistently favor YOLO over InceptionV3 as the primary
  morphology output.

### Claims to avoid

- “The pipeline is 92.5% accurate in Nigeria.”
- “External segmentation accuracy was validated.”
- “The model correctly classified 187,838 Nigerian cells.”
- “The pipeline diagnoses SCD from a smear.”
- “The method is clinically validated or ready for deployment.”
- “The result proves generalization to all countries, laboratories, or microscope systems.”
- “Every sample corresponds to a unique patient.”
- “The Nigerian burden percentage is an expert-verified sickle-cell percentage.”
- “Inception failed because of one confirmed mechanism.” Domain shift is the supported
  interpretation, but its individual acquisition causes were not experimentally isolated.

## 12. Reproducibility and source-of-truth files

The completed external experiment is `exp_009`:

- Completion status and timestamps: `final/scd_yolo_pipeline/exp/exp_009/run.json`
- Metrics, audit, versions, and checkpoint hashes:
  `final/scd_yolo_pipeline/exp/exp_009/summary.json`
- Exact configuration: `final/scd_yolo_pipeline/exp/exp_009/config.json`
- Sample-level burdens: `final/scd_yolo_pipeline/exp/exp_009/tables/nigeria_samples.csv`
- Field-level burdens: `final/scd_yolo_pipeline/exp/exp_009/tables/nigeria_fields.csv`
- All detected cells: `final/scd_yolo_pipeline/exp/exp_009/tables/nigeria_cells.csv`
- Dataset/image manifest:
  `final/scd_yolo_pipeline/exp/exp_009/tables/nigeria_sample_manifest.csv`
- Explicit exclusions: `final/scd_yolo_pipeline/exp/exp_009/tables/nigeria_excluded.csv`
- Qualitative stage images: `final/scd_yolo_pipeline/exp/exp_009/images/`

### Execution environment

- NVIDIA A100-SXM4-80GB
- Python 3.11.15
- PyTorch 2.1.2 + CUDA 12.1
- Torchvision 0.16.2 + CUDA 12.1
- Ultralytics 8.4.115
- OpenCV 4.11.0
- NumPy 1.26.4
- pandas 2.3.3

Checkpoint SHA-256 hashes:

- YOLO: `c57091e25dc2b90e8bb8310bbbbc2aa829d2cc65c0d10bce4c7acd649be55965`
- InceptionV3: `e5b81fec82351d34a05e8f0b4b3afa5da8ac66e11dadef60e1a0bf901b0d1d15`

### Reproduction command

From the repository root, after the verified data archives are present:

~~~bash
bash final/run_nigeria_external.sh
~~~

The command creates a new immutable experiment number; it does not overwrite `exp_009`.

## 13. External-evaluation writing checklist

- [ ] Call the result external sample-level discrimination, not external cell-level validation.
- [ ] Report both AUROC and average precision with their clustered-bootstrap confidence intervals.
- [ ] State the evaluated class balance: 72 positive and 69 negative samples.
- [ ] Describe hemoglobin electrophoresis as the source of the supplied sample labels.
- [ ] Report the 141-sample QC denominator and the two excluded fields.
- [ ] Keep YOLO primary and InceptionV3 as an external domain-shift ablation.
- [ ] Do not convert AUROC into accuracy.
- [ ] Do not report threshold metrics until a threshold protocol is implemented.
- [ ] Keep Nigerian sample-level results separate from Cuban cell-mask metrics.
- [ ] Cite the original MICCAI paper and UCL dataset record.
- [ ] Include the CC BY-NC-SA 4.0 license when documenting data availability.
- [ ] State that external cell-mask, calibration, and prospective validation remain future work.
