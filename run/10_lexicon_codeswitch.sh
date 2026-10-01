#!/bin/bash
# Lexicon code-switching (3.1.1). CPU only, ${WORKERS} processes.
# -> cs_marco/en-{zh,fa,ru}-MUSE/random_prob_0.5_switching_1_en-{lang}_collection.tsv
# -> cs_marco/{hau,som,swh,yor}/random_prob_0.5_switching_1_{lang}_collection.tsv
set -euo pipefail
source "$(dirname "$0")/common.sh"

for lang in ${MUSE_LANGS}; do
    out="${OUTPUT_DIR}/en-${lang}-MUSE"; mkdir -p "${out}"
    python "${SCRIPTS}/lexicon_codeswitch.py" --collection "${COLLECTION}" \
        --lexicon "${LEXICON_DIR}/muse/en-${lang}.json" --lang "en-${lang}" \
        --prob "${PROB}" --output "${out}" --workers "${WORKERS}" ${SEED_ARG[@]+"${SEED_ARG[@]}"}
done
for lang in ${PANLEX_LANGS}; do
    out="${OUTPUT_DIR}/${lang}"; mkdir -p "${out}"
    python "${SCRIPTS}/lexicon_codeswitch.py" --collection "${COLLECTION}" \
        --lexicon "${LEXICON_DIR}/panlex/eng-${lang}.json" --lang "${lang}" \
        --prob "${PROB}" --output "${out}" --workers "${WORKERS}" ${SEED_ARG[@]+"${SEED_ARG[@]}"}
done
