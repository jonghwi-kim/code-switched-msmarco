#!/bin/bash
# MS MARCO passages + neuMARCO translations (zh/fa/ru, used by word alignment).
set -euo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p "${DATA_DIR}/msmarco" "${TRANSLATION_DIR}/neumarco"

[ -f "${COLLECTION}" ] || curl -L https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz | tar -xz -C "${DATA_DIR}/msmarco"

# https://huggingface.co/datasets/neuclir/neumarco (3.7 GB archive)
if [ ! -f "${TRANSLATION_DIR}/neumarco/zh_collection.tsv" ]; then
    curl -L https://huggingface.co/datasets/neuclir/neumarco/resolve/main/data/neumarco.tar.gz \
        | tar -xz -C "${TRANSLATION_DIR}/neumarco"
    for pair in zho:zh fas:fa rus:ru; do
        mv "${TRANSLATION_DIR}/neumarco/eng-${pair%%:*}/msmarco.collection.20210731-scale21-sockeye2-tm1.tsv" \
           "${TRANSLATION_DIR}/neumarco/${pair##*:}_collection.tsv"
    done
fi
