# Code-Switched MS MARCO: generation scripts

Scripts to generate the code-switched MS MARCO passage collections of

> Jonghwi Kim. *Leveraging Code-Switched Data to Improve Multi-Vector Retrievers for Cross-Language
> Information Retrieval.* Master's thesis, POSTECH, 2025. https://oasis.postech.ac.kr/handle/2014.oak/127465

Only the scripts are released, as the collections are large. The generation is not deterministic, so regenerated collections differ from the ones used in the thesis.

## Methods (thesis Sec. 3.1)

| Method | Languages | Resources | Output (`$DATA_DIR/cs_marco/`) |
|---|---|---|---|
| Lexicon | zh, fa, ru | MUSE dictionaries (`lexicons/muse`) | `en-{lang}-MUSE/random_prob_0.5_switching_1_en-{lang}_collection.tsv` |
| | hau, som, swh, yor | PanLex (`lexicons/panlex`) | `{hau,som,swh,yor}/random_prob_0.5_switching_1_{hau,som,swh,yor}_collection.tsv` |
| Word alignment | zh, fa, ru | Awesome-Align + neuMARCO translations | `word_align/en_{lang}_mixed_collection_with_parallel_0.5.tsv` |
| | hau, som, swh, yor | Awesome-Align + your own translations (see below) | same pattern (`en_ha`, `en_so`, `en_sw`, `en_yo`) |
| Keyphrase translation | all 7 | KeyBERT + NLLB-200 1.3B | `keyphrase_nllb/en_{flores}_mixed_collection.tsv` |

- Each switchable unit (lexicon match, aligned content word, keyphrase) is switched independently with probability 0.5. That makes the rate 50% on average; it varies per passage.
- Outputs are `pid<TAB>passage` files with the MS MARCO pids.

## Contents

```
lexicons/muse/en-{zh,fa,ru}.json               MUSE, English -> target
lexicons/panlex/eng-{hau,som,swh,yor}.json     PanLex (20240301 dump), English -> target
scripts/                                       Python scripts
run/                                           run scripts (thesis settings)
```

The lexicons are JSON objects `{english word or phrase: [translations]}`.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; [nltk.download(p) for p in ['stopwords','wordnet','omw-1.4','punkt','punkt_tab']]"
```

## Run

```bash
export DATA_DIR=/path/to/data
bash run/00_download_data.sh          # MS MARCO passages, neuMARCO translations (zh/fa/ru)
bash run/10_lexicon_codeswitch.sh
bash run/20_wordalign_codeswitch.sh
bash run/30_keyphrase_codeswitch.sh
```

Options (environment variables):
- `MUSE_LANGS="zh fa ru"`, `PANLEX_LANGS="hau som swh yor"`: choose languages; use these codes, and `""` skips a group.
- `SEED=1`: reproducible output.
- `PROB`: switching probability.
- `COLLECTION`, `OUTPUT_DIR`: input and output locations.
- `WORKERS` (default: all cores): processes for lexicon switching; word alignment runs spaCy in up to 16 of them.
- `DTYPE=auto`: NLLB and Awesome-Align weights, `auto` = float16 on a GPU and float32 on CPU. `DTYPE=float32` keeps full precision.
- `ALIGN_BATCH_SIZE=256`, `NLLB_BATCH_SIZE=256`: sequences per Awesome-Align forward pass and phrases per NLLB call; lower them if GPU memory runs out.
- `N_GRAM=3`, `STOP_WORDS=english`: KeyBERT n-gram length and stop words (the thesis code used `N_GRAM=5 STOP_WORDS=none`).

A GPU is recommended for steps 20 and 30.

Word alignment for hau/som/swh/yor needs a translated collection with the same pids (the thesis used NLLB-200 600M translations): `$DATA_DIR/translations/nllb/{hau_Latn,som_Latn,swh_Latn,yor_Latn}_collection.tsv`.

## Models and data

| | |
|---|---|
| NLLB-200 1.3B distilled (keyphrase translation) | https://huggingface.co/facebook/nllb-200-distilled-1.3B |
| Awesome-Align (word alignment) | https://huggingface.co/aneuraz/awesome-align-with-co |
| KeyBERT embedding model | https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2 |
| neuMARCO (zh/fa/ru translations) | https://huggingface.co/datasets/neuclir/neumarco |
| MS MARCO passages | https://microsoft.github.io/msmarco/ |
| MUSE dictionaries (CC BY-NC 4.0) | https://github.com/facebookresearch/MUSE |
| PanLex 20240301 (CC0) | https://huggingface.co/datasets/cointegrated/panlex-meanings |

## Using other resources

The lexicons, translations, aligner and translation model can be replaced with your own:
- **Lexicon**: any JSON `{english word or phrase: [translations]}`, passed with `--lexicon` to `scripts/lexicon_codeswitch.py` (or put in `LEXICON_DIR` with the same file names).
- **Translations for word alignment**: any translated collection with the same pids, passed with `--target_lang_file` (or put in `TRANSLATION_DIR`).
- **Aligner**: the model name in `scripts/WordAligner.py` (a BERT-style encoder; choose the layer with `--align_layer`).
- **Keyphrase translation**: `--model_name` of `scripts/translate_keyphrase.py` for another NLLB-200 model (target codes are FLORES-200), or `scripts/Translator.py` for another MT model.

## Notes

- **Lexicons.** Built from the MUSE `en-xx.txt` dictionaries and from PanLex (English and target entries joined on meaning).
  - Removed: translations equal to the English word (e.g. `utc → utc`), duplicate translations, and malformed PanLex entries (e.g. `??`-prefixed forms, `=`-joined variants, etymology notes).
  - MUSE en-zh is in Traditional Chinese script.
- **Keyphrases.**
  - KeyBERT: `stop_words="english"`, 1–3-grams, MMR diversity 0.5, 20 candidates per passage.
  - Each keyphrase is located (case-insensitively) at its first occurrence in the passage, and only that occurrence is replaced when it is selected. Other occurrences of the same words stay in English.
  - Each distinct keyphrase is translated once. NLLB repetition-loop outputs are skipped.
- **Seeds.** Each script prints the seed it uses (a random one when `SEED` is not set). Lexicon switching seeds every passage with (`SEED`, pid), so its output does not depend on `WORKERS`.

## Citation

```bibtex
@mastersthesis{kim2025codeswitched,
  title  = {Leveraging Code-Switched Data to Improve Multi-Vector Retrievers for Cross-Language Information Retrieval},
  author = {Kim, Jonghwi},
  school = {Pohang University of Science and Technology (POSTECH)},
  year   = {2025},
  url    = {https://oasis.postech.ac.kr/handle/2014.oak/127465}
}
```
