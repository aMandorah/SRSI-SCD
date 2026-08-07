import json
from pathlib import Path

from scd_yolo_pipeline.artifacts import Experiment
from scd_yolo_pipeline.config import PipelineConfig, discover_project_root
from scd_yolo_pipeline.external_evaluation import _auc, _average_precision
from scd_yolo_pipeline.nigeria import _family, parse_nigeria_labels


def test_discovers_repository_root() -> None:
    root = discover_project_root(Path(__file__).resolve())
    assert (root / "final" / "yolo_seg_finetuned.pt").is_file()
    assert (root / "data" / "SCD_Final.zip").is_file()


def test_default_inputs_resolve() -> None:
    config = PipelineConfig.create(project_root=discover_project_root())
    config.validate_inputs(require_data=True)
    assert config.segmenter_checkpoint.name == "yolo_seg_finetuned.pt"
    metadata = json.loads(config.best_model_json.read_text())
    assert (config.checkpoint_dir / Path(metadata["checkpoint"]).name).is_file()
    assert config.classification_source == "inception"


def test_classification_source_validation() -> None:
    root = discover_project_root(Path(__file__).resolve())
    yolo_config = PipelineConfig.create(project_root=root, classification_source="yolo")
    assert yolo_config.classification_source == "yolo"
    try:
        PipelineConfig.create(project_root=root, classification_source="invalid")
    except ValueError as error:
        assert "classification_source" in str(error)
    else:
        raise AssertionError("invalid classification source was accepted")


def test_experiment_numbers_are_monotonic(tmp_path: Path) -> None:
    first = Experiment.create(tmp_path)
    second = Experiment.create(tmp_path)
    assert first.path.name == "exp_000"
    assert second.path.name == "exp_001"
    assert json.loads(first.path.joinpath("run.json").read_text())["status"] == "running"


def test_nigeria_label_parser_and_repeat_family(tmp_path: Path) -> None:
    labels = tmp_path / "labels.txt"
    labels.write_text("a, 0\na,0\nb, 1\n")
    parsed, audit = parse_nigeria_labels(labels)
    assert parsed == {"a": 0, "b": 1}
    assert audit["duplicate_rows"] == 1
    assert _family("101017-07r2") == "101017-07"


def test_external_metrics_perfect_scores() -> None:
    import numpy as np

    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert _auc(y, score) == 1.0
    assert _average_precision(y, score) == 1.0
