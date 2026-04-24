#!/usr/bin/env bash
# Download the six CZ CELLxGENE single-cell atlases used in the benchmark.
#
# Usage:  bash scripts/00_download_datasets.sh [OUT_DIR]
# Default OUT_DIR is ./data/raw

set -euo pipefail

OUT_DIR="${1:-data/raw}"
mkdir -p "${OUT_DIR}"

declare -A FILES=(
    ["TNBC_Breast_Cancer.h5ad"]="https://datasets.cellxgene.cziscience.com/af8c4fce-4c63-4671-b339-91a383cf36f6.h5ad"
    ["Indonesia_PBMC.h5ad"]="https://datasets.cellxgene.cziscience.com/665714af-4be5-49a3-913b-5ab5ac25620d.h5ad"
    ["Brain_Atlas.h5ad"]="https://datasets.cellxgene.cziscience.com/0ab54d91-066c-4223-a9ea-6a3b0d1adef4.h5ad"
    ["Multi_Tissue_TME.h5ad"]="https://datasets.cellxgene.cziscience.com/921d46a3-69b4-44a8-b2d6-9ef5c7803bc3.h5ad"
    ["Human_Pancreas.h5ad"]="https://datasets.cellxgene.cziscience.com/00d88707-e33a-4c2a-821a-cdc32a98d050.h5ad"
    ["Pig_Pancreas.h5ad"]="https://datasets.cellxgene.cziscience.com/55cfae87-6348-44df-a4ed-c132569dea54.h5ad"
)

DOWNLOADER=""
if command -v aria2c >/dev/null 2>&1; then
    DOWNLOADER="aria2c -x 8 -s 8 -c -d ${OUT_DIR} -o"
elif command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl -L --create-dirs -C - -o"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget -c -O"
else
    echo "ERROR: need aria2c, curl or wget on PATH." >&2
    exit 1
fi

echo "Downloading 6 atlases (~5 GB total) into: ${OUT_DIR}"
for name in "${!FILES[@]}"; do
    target="${OUT_DIR}/${name}"
    if [[ -s "${target}" ]]; then
        echo "  [skip] ${name} already exists"
        continue
    fi
    url="${FILES[${name}]}"
    echo "  [get ] ${name}"
    if [[ "${DOWNLOADER}" == aria2c* ]]; then
        aria2c -x 8 -s 8 -c -d "${OUT_DIR}" -o "${name}" "${url}"
    elif [[ "${DOWNLOADER}" == curl* ]]; then
        curl -L --create-dirs -C - -o "${target}" "${url}"
    else
        wget -c -O "${target}" "${url}"
    fi
done

echo "Done. Files saved in ${OUT_DIR}/"
ls -lh "${OUT_DIR}"
