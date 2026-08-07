#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE="$ROOT/final/scd_yolo_pipeline"
DATA_ROOT="$ROOT/data/external/nigeria_ucl_scd"

cd "$PIPELINE"
if [[ ! -f "$DATA_ROOT/archives/thin_films_part1.tar.gz" || ! -f "$DATA_ROOT/archives/thin_films_part2.tar.gz" || ! -f "$DATA_ROOT/archives/sickle_slides_new_march.txt" ]]; then
  echo "Verified Nigerian archives are missing under $DATA_ROOT" >&2
  exit 2
fi

PART1_MD5=$(md5sum "$DATA_ROOT/archives/thin_films_part1.tar.gz" | awk '{print $1}')
PART2_MD5=$(md5sum "$DATA_ROOT/archives/thin_films_part2.tar.gz" | awk '{print $1}')
LABEL_MD5=$(md5sum "$DATA_ROOT/archives/sickle_slides_new_march.txt" | awk '{print $1}')
[[ "$PART1_MD5" == 7cf50dd6d52bdb7c74b738d377e63e13 ]]
[[ "$PART2_MD5" == 4fc5cb40a60dc39eaa4bf033859c085d ]]
[[ "$LABEL_MD5" == 9b04a97fe5f5ca3c2dc6c6ddd438ec1e ]]

# Run on the allocated GPU. Stage images are limited to a deterministic 16-sample subset.
.venv/bin/python -m scd_yolo_pipeline external-evaluate \
  --external-root "$DATA_ROOT" \
  --classification-source yolo \
  --device cuda \
  --stage-sample-count 16
