from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .pipeline import DEFAULT_CLASSES, Instance, to_canvas


@dataclass
class HeldOutDataset:
    metadata: pd.DataFrame
    test_smears: list[str]
    ground_truth: dict[str, list[Instance]]
    source_paths: dict[str, Path]
    set_by_smear: dict[str, str]
    missing_masks: dict[str, list[str]]


def extract_dataset(config: PipelineConfig) -> Path:
    """Extract SCD_Final.zip once, rejecting paths outside the destination."""
    config.validate_inputs(require_data=True)
    dataset_root = config.data_dir / "SCD_Final"
    supplementary = dataset_root / "_supplementary"
    if supplementary.is_dir():
        return dataset_root

    config.data_dir.mkdir(parents=True, exist_ok=True)
    destination = config.data_dir.resolve()
    with zipfile.ZipFile(config.data_zip) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(destination)
            except ValueError as error:
                raise ValueError(f"Unsafe archive member: {member.filename}") from error
        archive.extractall(destination)
    if not supplementary.is_dir():
        raise FileNotFoundError(
            f"Archive extracted, but expected directory is absent: {supplementary}"
        )
    return dataset_root


def enumerate_mask_smears(dataset_root: Path) -> pd.DataFrame:
    supplementary = dataset_root / "_supplementary"
    rows: list[dict[str, object]] = []
    for set_dir in sorted(path for path in supplementary.iterdir() if path.is_dir()):
        for smear_dir in sorted(path for path in set_dir.iterdir() if path.is_dir()):
            files = {
                path.name.lower(): path
                for path in smear_dir.iterdir()
                if not path.name.startswith("._")
            }
            if "source.jpg" not in files:
                continue
            rows.append(
                {
                    "smear": smear_dir.name,
                    "dir": smear_dir,
                    "source": files["source.jpg"],
                    "set": "set2" if "set2" in set_dir.name else "set3",
                }
            )
    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise RuntimeError(f"No annotated smears found under {supplementary}")
    return metadata


def reproduce_split(metadata: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Reproduce step 3's exact per-set 70/15/15 split."""
    metadata = metadata.copy()
    rng = np.random.default_rng(seed)
    split: dict[str, str] = {}
    for _, partition in metadata.groupby("set"):
        smear_ids = partition["smear"].tolist()
        rng.shuffle(smear_ids)
        count = len(smear_ids)
        train_count = round(0.70 * count)
        val_count = round(0.15 * count)
        for index, smear in enumerate(smear_ids):
            split[smear] = (
                "train"
                if index < train_count
                else "val"
                if index < train_count + val_count
                else "test"
            )
    metadata["split"] = metadata["smear"].map(split)
    return metadata


def _connected_components(mask: np.ndarray, cls: int) -> list[Instance]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return []
    median_area = float(np.median(areas))
    return [
        Instance(labels == index, cls=cls)
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= 0.25 * median_area
    ]


def load_held_out_dataset(config: PipelineConfig) -> HeldOutDataset:
    dataset_root = extract_dataset(config)
    metadata = reproduce_split(enumerate_mask_smears(dataset_root), config.seed)
    test_rows = metadata.loc[metadata["split"] == "test"]

    ground_truth: dict[str, list[Instance]] = {}
    source_paths: dict[str, Path] = {}
    missing_masks: dict[str, list[str]] = {}
    for row in test_rows.itertuples():
        source_path = Path(row.source)
        source = cv2.imread(str(source_path))
        if source is None:
            raise ValueError(f"Could not read held-out source: {source_path}")
        files = {path.name.lower(): path for path in Path(row.dir).iterdir()}
        instances: list[Instance] = []
        smear_missing_masks: list[str] = []
        for class_id, class_name in enumerate(DEFAULT_CLASSES):
            mask_path = files.get(f"mask-{class_name}.jpg")
            if mask_path is None:
                smear_missing_masks.append(f"mask-{class_name}.jpg")
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not read mask: {mask_path}")
            if mask.shape[:2] != source.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (source.shape[1], source.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            canvas_mask = to_canvas(mask, config.work_size, cv2.INTER_NEAREST)
            instances.extend(_connected_components(canvas_mask, class_id))
        if smear_missing_masks:
            missing_masks[row.smear] = smear_missing_masks
        ground_truth[row.smear] = instances
        source_paths[row.smear] = source_path

    test_smears = sorted(test_rows["smear"].tolist())
    return HeldOutDataset(
        metadata=metadata,
        test_smears=test_smears,
        ground_truth=ground_truth,
        source_paths=source_paths,
        set_by_smear=dict(zip(metadata["smear"], metadata["set"])),
        missing_masks=missing_masks,
    )
