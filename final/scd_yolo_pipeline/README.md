# SCD YOLO pipeline

A reproducible Python package built from ../scd_final_step4_yolo.ipynb.

The pipeline:

1. square-pads and resizes a smear to 1024×1024;
2. segments cells with ../yolo_seg_finetuned.pt;
3. maps detections back to source resolution and makes 80×80 crops with 15% padding;
4. runs ../inceptionv3_best.pt using the metadata and threshold in ../best_model.json;
5. retains both YOLO and Inception class predictions;
6. saves tables, metrics, and an image for every stage.

No training occurs here. The notebook preprocessing, split seed, mask cleanup, matching rule,
normalization, and decision threshold are preserved.

## Environment

From this directory:

~~~bash
uv sync --frozen --group dev
~~~

This creates .venv/ from uv.lock. The project pins Python 3.11, Torch 2.1.2,
Torchvision 0.16.2, and the CUDA 12.1 PyTorch index used in the successful A100 run.
The CLI selects CUDA when available and otherwise runs on CPU.

## Recommended evaluation

The best validated end-to-end result uses the class already predicted by the fine-tuned YOLO
checkpoint while still running and recording the Inception output:

~~~bash
uv run scd-yolo evaluate --classification-source yolo
~~~

The final A100 validation is in exp/exp_006/. See [ANALYSIS.md](ANALYSIS.md) for the comparison,
failure analysis, and data-quality caveat.

## Exact notebook-compatible evaluation

Inception remains the default primary classifier:

~~~bash
uv run scd-yolo evaluate
# Equivalent:
uv run scd-yolo evaluate --classification-source inception
~~~

For a quick smoke test:

~~~bash
uv run scd-yolo evaluate --limit 1 --device cpu
~~~

## Run one image

~~~bash
uv run scd-yolo image /path/to/smear.jpg --classification-source yolo
~~~

Use uv run scd-yolo image --help or uv run scd-yolo evaluate --help for every path,
device, confidence, model-source, and artifact option.

## Experiment outputs

Each command atomically reserves the next directory under exp/, beginning with exp_000.
Existing experiments are never overwritten. Failed runs retain run.json plus error.txt.

~~~text
exp/exp_XXX/
├── config.json
├── run.json
├── summary.json
├── images/
│   ├── evaluation_scores.png
│   └── <smear>/
│       ├── stage_00_source.jpg
│       ├── stage_01_canvas.jpg
│       ├── stage_02_segmentation.png
│       ├── stage_03_crops.png
│       └── stage_04_classification.png
└── tables/
    ├── per_smear.csv
    ├── matched_cells.csv
    └── cells/<smear>.csv
~~~

Per-cell tables contain the selected pred, segmenter_class, inception_pred,
Inception probability, YOLO detection confidence, box coordinates, and match diagnostics.
Evaluation summaries contain raw metrics and quality-controlled metrics whenever annotation
files are missing.

## Default inputs

Paths are discovered from the repository checkout:

- data archive: ../../data/SCD_Final.zip
- checkpoints: ../yolo_seg_finetuned.pt, ../inceptionv3_best.pt
- model metadata: ../best_model.json
- extracted data: ../../data/scd_final_data/
- experiments: exp/exp_XXX/

CLI flags override all defaults. The notebook-compatible SCD_PROJECT_ROOT,
SCD_CHECKPOINT_DIR, SCD_DATA_ZIP, SCD_DATA_DIR, SCD_YOLO_FT_CKPT,
SCD_BEST_MODEL_JSON, SCD_CLF_CKPT, and SCD_CLF_MODEL variables also work.

## Validation limitations

The supplied evaluation is useful for pipeline regression, not clinical validation.
One held-out smear is missing a circular-cell mask; the pipeline reports it instead of silently
treating the class as empty. The Cuba classifier crops also lack smear IDs, so training/evaluation
overlap cannot be ruled out. A clean external test set is still required before deployment.


## Nigerian external cohort

The `external-evaluate` command evaluates the public Nigerian thin-film cohort from University
College Hospital, Ibadan, using the frozen checkpoints:

~~~bash
uv run scd-yolo external-evaluate --download --classification-source yolo
~~~

The release has electrophoresis labels at the sample level, but no public cell masks. The external
run therefore reports per-field/per-sample elongated-cell burden, YOLO-versus-Inception agreement,
and AUROC/average precision for SCD status. It does not report cell-level segmentation precision,
recall, IoU, or morphology accuracy. Raw downloads and extracted data are stored under
`data/external/nigeria_ucl_scd/`; the immutable experiment contains CSV tables, hashes, exclusions,
and a deterministic set of stage images.
