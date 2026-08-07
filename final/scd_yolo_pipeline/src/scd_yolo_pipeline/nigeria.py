from __future__ import annotations

import csv
import hashlib
import json
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

NIGERIA_ROOT = "data/external/nigeria_ucl_scd"
LABEL_URL = "https://ndownloader.figshare.com/files/23561738"
ARCHIVES = {
    "thin_films_part1.tar.gz": {
        "url": "https://ndownloader.figshare.com/files/23559926",
        "size": 8733999424,
        "md5": "7cf50dd6d52bdb7c74b738d377e63e13",
    },
    "thin_films_part2.tar.gz": {
        "url": "https://ndownloader.figshare.com/files/23561570",
        "size": 17312741228,
        "md5": "4fc5cb40a60dc39eaa4bf033859c085d",
    },
    "sickle_slides_new_march.txt": {
        "url": LABEL_URL,
        "size": 2741,
        "md5": "9b04a97fe5f5ca3c2dc6c6ddd438ec1e",
    },
}


@dataclass(frozen=True)
class NigeriaSample:
    sample_id: str
    label: int
    family_id: str
    images: tuple[Path, ...]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_nigeria(root: Path, *, overwrite: bool = False) -> dict[str, object]:
    """Download and verify the public Nigerian release into an ignored data directory."""
    archive_dir = root / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, object] = {"root": str(root), "files": {}}
    for name, metadata in ARCHIVES.items():
        destination = archive_dir / name
        if overwrite or not destination.is_file() or _md5(destination) != metadata["md5"]:
            partial = destination.with_suffix(destination.suffix + ".part")
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "scd-yolo-pipeline"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(metadata["url"], headers=headers)
            with urllib.request.urlopen(request) as response:
                append = bool(offset) and response.status == 206
                if not append:
                    offset = 0
                with partial.open("ab" if append else "wb") as handle:
                    while block := response.read(1024 * 1024):
                        handle.write(block)
            if partial.stat().st_size != metadata["size"] or _md5(partial) != metadata["md5"]:
                partial.unlink(missing_ok=True)
                raise OSError(f"Checksum/size verification failed for {name}")
            partial.replace(destination)
        records["files"][name] = {**metadata, "path": str(destination), "md5": _md5(destination)}
    (root / "download_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        root = destination.resolve()
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(root) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive member: {member.name}")
        handle.extractall(destination)


def extract_nigeria(root: Path) -> Path:
    archive_dir = root / "archives"
    extracted = root / "extracted"
    if not (archive_dir / "thin_films_part1.tar.gz").is_file():
        raise FileNotFoundError("Run the Nigeria download command first")
    marker = extracted / ".complete"
    if not marker.is_file():
        for name in ("thin_films_part1.tar.gz", "thin_films_part2.tar.gz"):
            _safe_extract(archive_dir / name, extracted)
        marker.write_text("complete\n")
    return extracted


def parse_nigeria_labels(path: Path) -> tuple[dict[str, int], dict[str, object]]:
    labels: dict[str, int] = {}
    rows = 0
    duplicates = 0
    conflicts: list[str] = []
    with path.open(newline="") as handle:
        for raw in handle:
            if not raw.strip() or "," not in raw:
                continue
            sample, value = (part.strip() for part in raw.split(",", 1))
            if value not in {"0", "1"}:
                continue
            rows += 1
            label = int(value)
            if sample in labels:
                duplicates += 1
                if labels[sample] != label:
                    conflicts.append(sample)
            else:
                labels[sample] = label
    if conflicts:
        raise ValueError(f"Contradictory Nigeria labels: {sorted(set(conflicts))}")
    return labels, {"rows": rows, "unique_samples": len(labels), "duplicate_rows": duplicates}


def _family(sample_id: str) -> str:
    return re.sub(r"r[0-9]+$", "", sample_id)


def enumerate_nigeria(root: Path) -> tuple[list[NigeriaSample], dict[str, object]]:
    extracted = extract_nigeria(root)
    label_path = root / "archives" / "sickle_slides_new_march.txt"
    labels, audit = parse_nigeria_labels(label_path)
    candidates: dict[str, list[Path]] = {}
    for path in extracted.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff",
        }:
            continue
        components = set(path.relative_to(extracted).parts)
        for sample in labels:
            if sample in components:
                candidates.setdefault(sample, []).append(path)
                break
    samples: list[NigeriaSample] = []
    missing: list[str] = []
    for sample_id, label in sorted(labels.items()):
        images = tuple(sorted(candidates.get(sample_id, [])))
        if images:
            samples.append(NigeriaSample(sample_id, label, _family(sample_id), images))
        else:
            missing.append(sample_id)
    audit.update({"matched_samples": len(samples), "missing_samples": missing})
    return samples, audit


def write_sample_manifest(path: Path, samples: list[NigeriaSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "family_id", "label", "image"])
        for sample in samples:
            for image in sample.images:
                writer.writerow([sample.sample_id, sample.family_id, sample.label, image])
