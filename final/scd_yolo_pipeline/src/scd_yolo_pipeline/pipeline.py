from __future__ import annotations

import hashlib
import json
import platform
import random
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.models as tvm
from torch import nn
from torchvision import transforms
from ultralytics import YOLO

from .config import PipelineConfig

DEFAULT_CLASSES = ("circular", "elongated")
CELL_COLUMNS = (
    "cell",
    "x0",
    "y0",
    "x1",
    "y1",
    "detection_confidence",
    "segmenter_class",
    "inception_pred",
    "p_sickle",
    "pred",
)


class Instance:
    """One instance mask, cropped to its bounding box on the work canvas."""

    __slots__ = (
        "cls",
        "det_class",
        "det_confidence",
        "inception_class",
        "mask",
        "score",
        "x0",
        "x1",
        "y0",
        "y1",
    )

    def __init__(
        self,
        full_mask: np.ndarray,
        cls: int = 0,
        score: float = 1.0,
        det_confidence: float = 1.0,
        det_class: int = 0,
    ) -> None:
        ys, xs = np.nonzero(full_mask)
        if len(xs) == 0:
            raise ValueError("Cannot create an instance from an empty mask")
        self.x0, self.x1 = int(xs.min()), int(xs.max()) + 1
        self.y0, self.y1 = int(ys.min()), int(ys.max()) + 1
        self.mask = np.ascontiguousarray(
            full_mask[self.y0 : self.y1, self.x0 : self.x1], dtype=bool
        )
        self.cls = int(cls)
        self.score = float(score)
        self.det_confidence = float(det_confidence)
        self.det_class = int(det_class)
        self.inception_class = int(cls)

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1

    def full(self, size: int) -> np.ndarray:
        result = np.zeros((size, size), dtype=bool)
        result[self.y0 : self.y1, self.x0 : self.x1] = self.mask
        return result


@dataclass
class PipelineResult:
    image_path: Path
    segmentation_instances: list[Instance]
    instances: list[Instance]
    canvas: np.ndarray
    source: np.ndarray
    crops: list[np.ndarray]
    cells: pd.DataFrame
    classification_source: str

    @property
    def sickle_count(self) -> int:
        return int((self.cells["pred"] == "elongated").sum()) if len(self.cells) else 0

    @property
    def sickle_percentage(self) -> float:
        return 100.0 * self.sickle_count / max(len(self.cells), 1)

    def summary(self) -> dict[str, object]:
        return {
            "image": str(self.image_path),
            "segmented_cells": len(self.segmentation_instances),
            "cells": len(self.cells),
            "sickle": self.sickle_count,
            "sickle_pct": self.sickle_percentage,
            "classification_source": self.classification_source,
        }


def build_backbone(name: str) -> nn.Module:
    """Rebuild the exact two-class architecture used during step-1 training."""
    if name == "inceptionv3":
        model = tvm.inception_v3(weights=None, init_weights=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 2)
    elif name == "maxvit_t":
        model = tvm.maxvit_t(weights=None)
    elif name == "mobilenetv3":
        model = tvm.mobilenet_v3_large(weights=None)
    elif name == "resnet18":
        model = tvm.resnet18(weights=None)
    elif name == "resnet50":
        model = tvm.resnet50(weights=None)
    elif name == "vgg16":
        model = tvm.vgg16(weights=None)
    elif name == "vgg19":
        model = tvm.vgg19(weights=None)
    else:
        raise ValueError(
            f"{name!r} is not a supported deep classifier. "
            "Choose inceptionv3, maxvit_t, mobilenetv3, resnet18, resnet50, vgg16, or vgg19."
        )

    if name in ("resnet18", "resnet50"):
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name in ("maxvit_t", "mobilenetv3", "vgg16", "vgg19"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    return model


def to_canvas(image: np.ndarray, work_size: int, interpolation: int) -> np.ndarray:
    """Square-pad on the bottom/right, then resize to the segmenter's frame."""
    height, width = image.shape[:2]
    side = max(height, width)
    padded = np.zeros((side, side) + image.shape[2:], dtype=image.dtype)
    padded[:height, :width] = image
    return cv2.resize(padded, (work_size, work_size), interpolation=interpolation)


def mask_iou(left: Instance, right: Instance) -> float:
    x0, y0 = max(left.x0, right.x0), max(left.y0, right.y0)
    x1, y1 = min(left.x1, right.x1), min(left.y1, right.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = np.logical_and(
        left.mask[y0 - left.y0 : y1 - left.y0, x0 - left.x0 : x1 - left.x0],
        right.mask[y0 - right.y0 : y1 - right.y0, x0 - right.x0 : x1 - right.x0],
    ).sum()
    return float(intersection) / float(left.area + right.area - intersection)


class SCDPipeline:
    """YOLO11-seg detection followed by the frozen step-1 deep classifier."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        config.validate_inputs(require_data=False)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.device = (
            "cuda"
            if config.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if config.device == "auto"
            else config.device
        )
        with config.best_model_json.open() as handle:
            self.metadata = json.load(handle)

        self.classifier_name = config.classifier_name or self.metadata["winner"]
        self.classifier_size = int(self.metadata.get("input_size") or 299)
        self.classifier_threshold = float(self.metadata["threshold"])
        classes = tuple(self.metadata.get("classes", DEFAULT_CLASSES))
        if classes != DEFAULT_CLASSES:
            raise ValueError(f"Unexpected class order {classes}; expected {DEFAULT_CLASSES}")
        self.classes = classes

        if config.classifier_checkpoint is not None:
            self.classifier_checkpoint = config.classifier_checkpoint
        else:
            filename = (
                Path(self.metadata["checkpoint"]).name
                if self.classifier_name == self.metadata["winner"]
                else f"{self.classifier_name}_best.pt"
            )
            self.classifier_checkpoint = config.checkpoint_dir / filename
        if not self.classifier_checkpoint.is_file():
            available = sorted(path.name for path in config.checkpoint_dir.glob("*.pt"))
            raise FileNotFoundError(
                f"Classifier checkpoint not found: {self.classifier_checkpoint}. "
                f"Available checkpoints: {available}"
            )

        self.segmenter = YOLO(str(config.segmenter_checkpoint))
        self.classifier = build_backbone(self.classifier_name)
        try:
            state = torch.load(
                self.classifier_checkpoint,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:  # PyTorch < 2.0 compatibility
            state = torch.load(self.classifier_checkpoint, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if state and all(str(key).startswith("module.") for key in state):
            state = {str(key)[7:]: value for key, value in state.items()}
        self.classifier.load_state_dict(state)
        self.classifier = self.classifier.to(self.device).eval()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.classifier_size, self.classifier_size), antialias=True),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def describe(self) -> dict[str, object]:
        return {
            "device": self.device,
            "gpu": torch.cuda.get_device_name(0) if self.device.startswith("cuda") else None,
            "segmenter": str(self.config.segmenter_checkpoint),
            "segmenter_sha256": self._sha256(self.config.segmenter_checkpoint),
            "segmentation_confidence": self.config.segmentation_confidence,
            "classification_source": self.config.classification_source,
            "classifier": self.classifier_name,
            "classifier_checkpoint": str(self.classifier_checkpoint),
            "classifier_sha256": self._sha256(self.classifier_checkpoint),
            "classifier_input_size": self.classifier_size,
            "classifier_threshold": self.classifier_threshold,
            "classes": list(self.classes),
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": version("torchvision"),
                "ultralytics": version("ultralytics"),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }

    def _segment(self, canvas: np.ndarray) -> list[Instance]:
        prediction = self.segmenter.predict(
            canvas,
            imgsz=self.config.work_size,
            conf=self.config.segmentation_confidence,
            verbose=False,
            device=self.device,
        )[0]
        if prediction.masks is None:
            return []

        masks = prediction.masks.data
        masks = masks.cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
        confidences = (
            prediction.boxes.conf.detach().cpu().numpy()
            if prediction.boxes is not None and prediction.boxes.conf is not None
            else np.ones(len(masks), dtype=float)
        )
        detector_classes = (
            prediction.boxes.cls.detach().cpu().numpy()
            if prediction.boxes is not None and prediction.boxes.cls is not None
            else np.zeros(len(masks), dtype=int)
        )
        instances: list[Instance] = []
        for mask, confidence, detector_class in zip(masks, confidences, detector_classes):
            binary = (mask > 0.5).astype(np.uint8)
            if binary.shape != (self.config.work_size, self.config.work_size):
                binary = cv2.resize(
                    binary,
                    (self.config.work_size, self.config.work_size),
                    interpolation=cv2.INTER_NEAREST,
                )
            if int(binary.sum()) < self.config.min_mask_pixels:
                continue
            instances.append(
                Instance(
                    binary.astype(bool),
                    det_confidence=float(confidence),
                    det_class=int(detector_class),
                )
            )
        return instances

    @torch.inference_mode()
    def _classify(self, crops: list[np.ndarray]) -> np.ndarray:
        probabilities: list[np.ndarray] = []
        for start in range(0, len(crops), self.config.batch_size):
            batch = torch.stack(
                [self.transform(crop) for crop in crops[start : start + self.config.batch_size]]
            ).to(self.device)
            logits = self.classifier(batch)
            probabilities.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(probabilities) if probabilities else np.zeros(0, dtype=float)

    def run(self, image_path: str | Path) -> PipelineResult:
        image_path = Path(image_path).expanduser().resolve()
        source = cv2.imread(str(image_path))
        if source is None:
            raise ValueError(f"Could not read image: {image_path}")
        canvas = to_canvas(source, self.config.work_size, cv2.INTER_AREA)
        raw_instances = self._segment(canvas)

        scale = max(source.shape[:2]) / self.config.work_size
        crops: list[np.ndarray] = []
        boxes: list[tuple[int, int, int, int]] = []
        kept: list[Instance] = []
        for instance in raw_instances:
            x0, y0, x1, y1 = [value * scale for value in instance.box]
            margin = self.config.crop_padding * max(x1 - x0, y1 - y0)
            x0 = int(max(0, x0 - margin))
            y0 = int(max(0, y0 - margin))
            x1 = int(min(source.shape[1], x1 + margin))
            y1 = int(min(source.shape[0], y1 + margin))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            crop = cv2.cvtColor(source[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            crop = cv2.resize(
                crop,
                (self.config.crop_size, self.config.crop_size),
                interpolation=cv2.INTER_AREA,
            )
            crops.append(crop)
            boxes.append((x0, y0, x1, y1))
            kept.append(instance)

        probabilities = self._classify(crops)
        rows: list[dict[str, object]] = []
        for cell_id, (instance, box, probability) in enumerate(zip(kept, boxes, probabilities)):
            instance.score = float(probability)
            instance.inception_class = int(probability >= self.classifier_threshold)
            instance.cls = (
                instance.det_class
                if self.config.classification_source == "yolo"
                else instance.inception_class
            )
            rows.append(
                {
                    "cell": cell_id,
                    "x0": box[0],
                    "y0": box[1],
                    "x1": box[2],
                    "y1": box[3],
                    "detection_confidence": instance.det_confidence,
                    "segmenter_class": self.classes[instance.det_class],
                    "inception_pred": self.classes[instance.inception_class],
                    "p_sickle": float(probability),
                    "pred": self.classes[instance.cls],
                }
            )
        cells = pd.DataFrame(rows, columns=CELL_COLUMNS)
        return PipelineResult(
            image_path=image_path,
            segmentation_instances=raw_instances,
            instances=kept,
            canvas=canvas,
            source=source,
            crops=crops,
            cells=cells,
            classification_source=self.config.classification_source,
        )
