# Shared settings; every variable can be overridden from the environment.
RELEASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="${RELEASE_DIR}/scripts"
LEXICON_DIR="${LEXICON_DIR:-${RELEASE_DIR}/lexicons}"

DATA_DIR="${DATA_DIR:-${RELEASE_DIR}/data}"
COLLECTION="${COLLECTION:-${DATA_DIR}/msmarco/collection.tsv}"   # MS MARCO passages, pid 0..8841822 in order
TRANSLATION_DIR="${TRANSLATION_DIR:-${DATA_DIR}/translations}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/cs_marco}"

PROB="${PROB:-0.5}"
WORKERS="${WORKERS:-$(nproc)}"   # CPU processes (lexicon switching; spaCy in word alignment uses at most 16)
DTYPE="${DTYPE:-auto}"           # NLLB / Awesome-Align weights: auto = float16 on GPU, float32 on CPU
SEED_ARG=()
if [ -n "${SEED:-}" ]; then SEED_ARG=(--seed "${SEED}"); fi
# Use as ${SEED_ARG[@]+"${SEED_ARG[@]}"} (empty arrays are "unbound" under set -u in bash < 4.4).

MUSE_LANGS="${MUSE_LANGS-zh fa ru}"             # NeuCLIR22 languages: zh fa ru (set "" to skip)
PANLEX_LANGS="${PANLEX_LANGS-hau som swh yor}"  # CIRAL languages, ISO 639-3 codes: hau som swh yor (set "" to skip)
declare -A ISO2=( [zh]=zh [fa]=fa [ru]=ru [hau]=ha [som]=so [swh]=sw [yor]=yo )
declare -A FLORES=( [zh]=zho_Hans [fa]=pes_Arab [ru]=rus_Cyrl [hau]=hau_Latn [som]=som_Latn [swh]=swh_Latn [yor]=yor_Latn )
