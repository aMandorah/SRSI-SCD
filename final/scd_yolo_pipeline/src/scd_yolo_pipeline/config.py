from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def discover_project_root(explicit: str | Path | None = None) -> Path:
    """Locate the repository containing final/ and data/."""
    if explicit is not None:
        start = Path(explicit).expanduser()
        start = start if start.is_dir() else start.parent
        candidates = [start, *start.parents]
    elif os.environ.get("SCD_PROJECT_ROOT"):
        start = Path(os.environ["SCD_PROJECT_ROOT"]).expanduser()
        candidates = [start, *start.parents]
    else:
        starts = [Path.cwd(), Path(__file__).resolve()]
        candidates = []
        for start in starts:
            start = start if start.is_dir() else start.parent
            candidates.extend([start, *start.parents])

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "final").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the repository root. Run from the checkout or pass --project-root."
    )


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    checkpoint_dir: Path
    data_zip: Path
    data_dir: Path
    experiment_root: Path
    segmenter_checkpoint: Path
    best_model_json: Path
    classifier_checkpoint: Path | None = None
    classifier_name: str | None = None
    device: str = "auto"
    seed: int = 42
    work_size: int = 1024
    crop_size: int = 80
    crop_padding: float = 0.15
    segmentation_confidence: float = 0.25
    mask_iou_threshold: float = 0.50
    min_mask_pixels: int = 30
    batch_size: int = 64
    classification_source: str = "inception"

    @classmethod
    def create(
        cls,
        *,
        project_root: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        data_zip: str | Path | None = None,
        data_dir: str | Path | None = None,
        experiment_root: str | Path | None = None,
        segmenter_checkpoint: str | Path | None = None,
        best_model_json: str | Path | None = None,
        classifier_checkpoint: str | Path | None = None,
        classifier_name: str | None = None,
        device: str = "auto",
        segmentation_confidence: float = 0.25,
        classification_source: str = "inception",
    ) -> PipelineConfig:
        if classification_source not in {"inception", "yolo"}:
            raise ValueError("classification_source must be 'inception' or 'yolo'")
        root = discover_project_root(project_root)
        checkpoints = (
            Path(checkpoint_dir).expanduser().resolve()
            if checkpoint_dir
            else _env_path("SCD_CHECKPOINT_DIR", root / "final")
        )
        project_dir = root / "final" / "scd_yolo_pipeline"
        return cls(
            project_root=root,
            checkpoint_dir=checkpoints,
            data_zip=(
                Path(data_zip).expanduser().resolve()
                if data_zip
                else _env_path("SCD_DATA_ZIP", root / "data" / "SCD_Final.zip")
            ),
            data_dir=(
                Path(data_dir).expanduser().resolve()
                if data_dir
                else _env_path("SCD_DATA_DIR", root / "data" / "scd_final_data")
            ),
            experiment_root=(
                Path(experiment_root).expanduser().resolve()
                if experiment_root
                else _env_path("SCD_EXPERIMENT_ROOT", project_dir / "exp")
            ),
            segmenter_checkpoint=(
                Path(segmenter_checkpoint).expanduser().resolve()
                if segmenter_checkpoint
                else _env_path("SCD_YOLO_FT_CKPT", checkpoints / "yolo_seg_finetuned.pt")
            ),
            best_model_json=(
                Path(best_model_json).expanduser().resolve()
                if best_model_json
                else _env_path("SCD_BEST_MODEL_JSON", checkpoints / "best_model.json")
            ),
            classifier_checkpoint=(
                Path(classifier_checkpoint).expanduser().resolve()
                if classifier_checkpoint
                else (
                    Path(os.environ["SCD_CLF_CKPT"]).expanduser().resolve()
                    if os.environ.get("SCD_CLF_CKPT")
                    else None
                )
            ),
            classifier_name=classifier_name or os.environ.get("SCD_CLF_MODEL"),
            device=device,
            segmentation_confidence=segmentation_confidence,
            classification_source=classification_source,
        )

    def validate_inputs(self, *, require_data: bool) -> None:
        required = {
            "checkpoint directory": self.checkpoint_dir,
            "segmenter checkpoint": self.segmenter_checkpoint,
            "classifier metadata": self.best_model_json,
        }
        if self.classifier_checkpoint is not None:
            required["classifier checkpoint"] = self.classifier_checkpoint
        if require_data:
            required["data archive"] = self.data_zip
        missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required input(s):\n  " + "\n  ".join(missing))

    def as_json(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }
