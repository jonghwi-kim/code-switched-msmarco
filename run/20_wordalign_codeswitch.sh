#!/bin/bash
# Word-alignment code-switching (3.1.2). GPU recommended.
# Translations: neuMARCO for zh/fa/ru (00_download_data.sh); for hau/som/swh/yor provide
# ${TRANSLATION_DIR}/nllb/{hau_Latn,som_Latn,swh_Latn,yor_Latn}_collection.tsv (same pids as the collection).
# -> cs_marco/word_align/en_{zh,fa,ru,ha,so,sw,yo}_mixed_collection_with_parallel_0.5.tsv
set -euo pipefail
source "$(dirname "$0")/common.sh"
out="${OUTPUT_DIR}/word_align"; mkdir -p "${out}"

for lang in ${MUSE_LANGS} ${PANLEX_LANGS}; do
    if [[ " ${MUSE_LANGS} " == *" ${lang} "* ]]; then
        translation="${TRANSLATION_DIR}/neumarco/${lang}_collection.tsv"
    else
        translation="${TRANSLATION_DIR}/nllb/${FLORES[$lang]}_collection.tsv"
    fi
    python "${SCRIPTS}/wordalign_codeswitch.py" --source_lang_file "${COLLECTION}" \
        --target_lang "${ISO2[$lang]}" --target_lang_file "${translation}" \
        --prob "${PROB}" --output_path "${out}" --dtype "${DTYPE}" --batch_size "${ALIGN_BATCH_SIZE:-256}" \
        --workers "$(( WORKERS < 16 ? WORKERS : 16 ))" ${SEED_ARG[@]+"${SEED_ARG[@]}"}
done
