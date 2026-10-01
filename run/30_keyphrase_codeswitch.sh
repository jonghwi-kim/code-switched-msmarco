#!/bin/bash
# Keyphrase-translation code-switching (3.1.3): KeyBERT -> p=0.5 selection -> NLLB-200 1.3B -> replacement.
# -> cs_marco/keyphrase_nllb/en_{zho_Hans,pes_Arab,rus_Cyrl,hau_Latn,som_Latn,swh_Latn,yor_Latn}_mixed_collection.tsv
set -euo pipefail
source "$(dirname "$0")/common.sh"
N_GRAM="${N_GRAM:-3}"; STOP_WORDS="${STOP_WORDS:-english}"
kp_dir="${OUTPUT_DIR}/keyphrase/${N_GRAM}gram_${STOP_WORDS}"; out="${OUTPUT_DIR}/keyphrase_nllb"; mkdir -p "${kp_dir}" "${out}"

# KeyBERT extraction is language independent and expensive: reuse it if present. Selection is re-run every time.
REUSE_ARG=()
if [ -f "${kp_dir}/keyphrase_${N_GRAM}gram.json" ]; then REUSE_ARG=(--input_keyword "${kp_dir}/keyphrase_${N_GRAM}gram.json"); fi
python "${SCRIPTS}/extract_keyphrase.py" --source_lang_file "${COLLECTION}" --output_path "${kp_dir}" \
    --n_gram "${N_GRAM}" --stop_words "${STOP_WORDS}" --prob "${PROB}" \
    ${REUSE_ARG[@]+"${REUSE_ARG[@]}"} ${SEED_ARG[@]+"${SEED_ARG[@]}"}
selected="${kp_dir}/keyphrase_${N_GRAM}gram_selected.json"

for lang in ${MUSE_LANGS} ${PANLEX_LANGS}; do
    flores="${FLORES[$lang]}"
    python "${SCRIPTS}/translate_keyphrase.py" --keyphrase_json "${selected}" --target_language "${flores}" \
        --model_name facebook/nllb-200-distilled-1.3B --output_path "${kp_dir}/${flores}_keyphrases.json" \
        --dtype "${DTYPE}" --batch_size "${NLLB_BATCH_SIZE:-256}"
    python "${SCRIPTS}/apply_keyphrase_switch.py" --source_lang_file "${COLLECTION}" \
        --keyword_file "${kp_dir}/${flores}_keyphrases.json" --output_file "${out}/en_${flores}_mixed_collection.tsv"
done
