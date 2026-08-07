from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .artifacts import Experiment, safe_name, save_evaluation_figure, save_stage_images
from .config import PipelineConfig
from .dataset import load_held_out_dataset
from .evaluation import evaluate_pipeline
from .external_evaluation import evaluate_nigeria
from .nigeria import download_nigeria, enumerate_nigeria, write_sample_manifest
from .pipeline import PipelineResult, SCDPipeline


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--project-root", type=Path, help="Repository root containing final/ and data/"
    )
    parser.add_argument("--checkpoint-dir", type=Path, help="Directory containing all checkpoints")
    parser.add_argument("--segmenter-checkpoint", type=Path, help="YOLO segmentation checkpoint")
    parser.add_argument("--classifier-checkpoint", type=Path, help="Deep classifier checkpoint")
    parser.add_argument("--best-model-json", type=Path, help="Step-1 model metadata JSON")
    parser.add_argument("--classifier-name", help="Classifier architecture override")
    parser.add_argument("--data-zip", type=Path, help="SCD_Final ZIP archive")
    parser.add_argument("--data-dir", type=Path, help="Dataset extraction directory")
    parser.add_argument("--experiment-root", type=Path, help="Parent directory for exp_XXX runs")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO segmentation confidence threshold",
    )
    parser.add_argument(
        "--stage-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save an image after every pipeline stage",
    )
    parser.add_argument(
        "--classification-source",
        choices=("inception", "yolo"),
        default="inception",
        help="Model whose class decision becomes the primary pipeline prediction",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scd-yolo",
        description="Run the frozen YOLO11-seg → crop → classifier SCD pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()

    image = subparsers.add_parser(
        "image",
        parents=[common],
        help="Run the pipeline on one smear image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    image.add_argument("image", type=Path, help="Smear image to process")
    image.set_defaults(handler=run_image)

    evaluate = subparsers.add_parser(
        "evaluate",
        parents=[common],
        help="Evaluate the exact held-out split used by the notebook",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    evaluate.add_argument(
        "--limit",
        type=_positive_int,
        help="Process only the first N held-out smears (useful for smoke tests)",
    )
    evaluate.set_defaults(handler=run_evaluate)

    nigeria = subparsers.add_parser(
        "external-evaluate",
        parents=[common],
        help="Evaluate the Nigerian external cohort",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nigeria.add_argument("--external-root", type=Path)
    nigeria.add_argument("--download", action="store_true")
    nigeria.add_argument("--overwrite-download", action="store_true")
    nigeria.add_argument("--stage-sample-count", type=_positive_int, default=16)
    nigeria.add_argument(
        "--limit-samples", type=_positive_int, help="Process only the first N samples"
    )
    nigeria.set_defaults(handler=run_external_evaluate)
    return parser


def _config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig.create(
        project_root=args.project_root,
        checkpoint_dir=args.checkpoint_dir,
        data_zip=args.data_zip,
        data_dir=args.data_dir,
        experiment_root=args.experiment_root,
        segmenter_checkpoint=args.segmenter_checkpoint,
        best_model_json=args.best_model_json,
        classifier_checkpoint=args.classifier_checkpoint,
        classifier_name=args.classifier_name,
        device=args.device,
        segmentation_confidence=args.confidence,
        classification_source=args.classification_source,
    )


def run_image(args: argparse.Namespace) -> int:
    config = _config(args)
    config.validate_inputs(require_data=False)
    if not args.image.expanduser().is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    experiment = Experiment.create(config.experiment_root)
    started = time.perf_counter()
    try:
        experiment.write_json("config.json", config.as_json())
        pipeline = SCDPipeline(config)
        result = pipeline.run(args.image)
        table_path = experiment.path / "tables" / "cells.csv"
        result.cells.to_csv(table_path, index=False)
        stage_artifacts = (
            save_stage_images(experiment, result, args.image.stem) if args.stage_images else {}
        )
        summary = {
            "mode": "image",
            "pipeline": pipeline.describe(),
            "result": result.summary(),
            "artifacts": {
                "cells": str(table_path.relative_to(experiment.path)),
                **stage_artifacts,
            },
        }
        experiment.write_json("summary.json", summary)
        elapsed = time.perf_counter() - started
        experiment.complete({"mode": "image", "elapsed_seconds": elapsed})
    except BaseException as error:
        experiment.fail(error)
        raise

    print(
        f"{args.image.name}: {len(result.cells)} cells · {result.sickle_count} sickle · "
        f"{result.sickle_percentage:.1f}%"
    )
    print(f"Experiment: {experiment.path}")
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    config = _config(args)
    config.validate_inputs(require_data=True)
    experiment = Experiment.create(config.experiment_root)
    started = time.perf_counter()
    try:
        experiment.write_json("config.json", config.as_json())
        dataset = load_held_out_dataset(config)
        pipeline = SCDPipeline(config)
        stage_artifacts: dict[str, object] = {}
        cells_dir = experiment.path / "tables" / "cells"
        cells_dir.mkdir(parents=True, exist_ok=True)

        def on_result(index: int, smear: str, result: PipelineResult) -> None:
            result.cells.to_csv(cells_dir / f"{safe_name(smear)}.csv", index=False)
            if args.stage_images:
                key = f"{index:02d}_{smear}"
                stage_artifacts[smear] = save_stage_images(experiment, result, key)

        evaluation = evaluate_pipeline(
            pipeline,
            dataset,
            limit=args.limit,
            on_result=on_result,
        )
        per_smear_path = experiment.path / "tables" / "per_smear.csv"
        matched_path = experiment.path / "tables" / "matched_cells.csv"
        evaluation.per_smear.to_csv(per_smear_path, index=False)
        evaluation.matched_cells.to_csv(matched_path, index=False)
        score_figure = save_evaluation_figure(experiment, evaluation)
        summary = {
            "mode": "evaluate",
            "pipeline": pipeline.describe(),
            "dataset": {
                "archive": str(config.data_zip),
                "test_smears": evaluation.per_smear["smear"].tolist(),
                "limited": args.limit is not None,
                "annotation_warnings": dataset.missing_masks,
            },
            "evaluation": evaluation.summary(),
            "artifacts": {
                "per_smear": str(per_smear_path.relative_to(experiment.path)),
                "matched_cells": str(matched_path.relative_to(experiment.path)),
                "score_figure": str(score_figure.relative_to(experiment.path)),
                "stage_images": stage_artifacts,
            },
        }
        experiment.write_json("summary.json", summary)
        elapsed = time.perf_counter() - started
        experiment.complete(
            {
                "mode": "evaluate",
                "elapsed_seconds": elapsed,
                "n_images": len(evaluation.per_smear),
            }
        )
    except BaseException as error:
        experiment.fail(error)
        raise

    print("\nDetection:", evaluation.detection)
    print("Classification:", evaluation.classification)
    print("End-to-end:", evaluation.end_to_end)
    print(f"Experiment: {experiment.path}")
    return 0


def run_external_evaluate(args: argparse.Namespace) -> int:
    config = _config(args)
    config.validate_inputs(require_data=False)
    external_root = (
        args.external_root or (config.project_root / "data" / "external" / "nigeria_ucl_scd")
    ).resolve()
    if args.download:
        download_nigeria(external_root, overwrite=args.overwrite_download)
    samples, audit = enumerate_nigeria(external_root)
    if not samples:
        raise RuntimeError("No label-matched Nigerian images found")
    experiment = Experiment.create(config.experiment_root)
    experiment.write_json(
        "config.json",
        {
            **config.as_json(),
            "external_dataset": "nigeria_ucl_scd",
            "external_root": str(external_root),
        },
    )
    write_sample_manifest(experiment.path / "tables" / "nigeria_sample_manifest.csv", samples)
    pipeline = SCDPipeline(config)
    if args.limit_samples:
        samples = samples[: args.limit_samples]
    stage_ids = {sample.sample_id for sample in samples[: args.stage_sample_count]}
    stage_artifacts: dict[str, object] = {}

    def on_field(field_index: int, sample: object, image: Path, result: object) -> None:
        if sample.sample_id not in stage_ids or field_index != 0:
            return
        key = f"nigeria_{sample.sample_id}"
        stage_artifacts[sample.sample_id] = save_stage_images(experiment, result, key)

    evaluation = evaluate_nigeria(pipeline, samples, experiment.path, on_field=on_field)
    evaluation.fields.to_csv(experiment.path / "tables" / "nigeria_fields.csv", index=False)
    evaluation.samples.to_csv(experiment.path / "tables" / "nigeria_samples.csv", index=False)
    evaluation.cells.to_csv(experiment.path / "tables" / "nigeria_cells.csv", index=False)
    evaluation.excluded.to_csv(experiment.path / "tables" / "nigeria_excluded.csv", index=False)
    experiment.write_json(
        "summary.json",
        {
            "mode": "external-evaluate",
            "dataset": {
                "name": "nigeria_ucl_scd",
                "audit": audit,
                "external_root": str(external_root),
            },
            "pipeline": pipeline.describe(),
            "evaluation": evaluation.metrics,
            "artifacts": {"stage_images": stage_artifacts},
        },
    )
    experiment.complete(
        {
            "mode": "external-evaluate",
            "n_samples": len(evaluation.samples),
            "n_fields": len(evaluation.fields),
        }
    )
    print(json.dumps(evaluation.metrics, indent=2))
    print(f"Experiment: {experiment.path}")
    return 0


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raise SystemExit(args.handler(args))
