import json
from pathlib import Path

from scd_yolo_pipeline.artifacts import Experiment
from scd_yolo_pipeline.config import PipelineConfig, discover_project_root


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
