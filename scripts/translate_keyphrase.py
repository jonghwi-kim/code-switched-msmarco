"""
Phrase-level translation of the selected keyphrases (thesis Sec. 3.1.3) with
NLLB-200 1.3B distilled: https://huggingface.co/facebook/nllb-200-distilled-1.3B

Input : keyphrase_*_selected.json from extract_keyphrase.py   {pid: [[start, end, phrase], ...]}
Output: {pid: [[start, end, phrase, translation], ...]}        (input of apply_keyphrase_switch.py)
Each distinct phrase is translated once (beam search is deterministic), in batches of similar length.
Generations that reach --max_length never terminated (NLLB repetition loops); they are stored as ""
and therefore not switched.
"""
import os
import json
import argparse

from tqdm.auto import tqdm
from Translator import NLLBTranslation

TO_FLORES = {"zh": "zho_Hans", "fa": "pes_Arab", "ru": "rus_Cyrl",
             "ha": "hau_Latn", "so": "som_Latn", "sw": "swh_Latn", "yo": "yor_Latn"}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyphrase_json", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--target_language", required=True, help="zh fa ru ha so sw yo, or a FLORES-200 code (e.g. pes_Arab)")
    parser.add_argument("--model_name", default="facebook/nllb-200-distilled-1.3B")
    parser.add_argument("--batch_size", type=int, default=256, help="phrases per generate() call")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--n_beams", type=int, default=5)
    parser.add_argument("--dtype", default="auto", help='"auto" (float16 on GPU, float32 on CPU), float32, float16 or bfloat16')
    args = parser.parse_args()

    target_language = TO_FLORES.get(args.target_language, args.target_language)
    translator = NLLBTranslation(args.model_name, target_language, n_beams=args.n_beams, max_length=args.max_length, num_return=1, dtype=args.dtype)

    with open(args.keyphrase_json) as f:
        keyphrases = json.load(f)
    items = [(pid, phrases) for pid, phrases in keyphrases.items() if phrases]

    # Keyphrases repeat across passages: translate each distinct phrase once. Sorting by token length keeps
    # the padding (and the beam search over padded positions) per batch small.
    unique = sorted({p for _, phrases in items for _, _, p in phrases})
    n_tokens = translator.tokenizer(unique, add_special_tokens=False)["input_ids"]
    unique = [p for _, p in sorted(zip(map(len, n_tokens), unique), reverse=True)]   # longest first: OOM shows up early
    print(f"{sum(len(phrases) for _, phrases in items)} phrases in {len(items)} passages, {len(unique)} distinct")

    translation, n_unfinished = {}, 0
    for i in tqdm(range(0, len(unique), args.batch_size), desc=f"Translating to {target_language}"):
        batch = unique[i:i + args.batch_size]
        for phrase, ((target, finished),) in zip(batch, translator.translate(batch)):
            translation[phrase] = target if finished else ""
            n_unfinished += not finished

    translated = {pid: [[s, e, p, translation[p]] for s, e, p in phrases] for pid, phrases in items}
    with open(args.output_path + ".tmp", 'w', encoding='utf-8') as f:
        json.dump(translated, f, ensure_ascii=False)
    os.replace(args.output_path + ".tmp", args.output_path)
    print(f"Translated keyphrases saved to {args.output_path} (unfinished generations skipped: {n_unfinished} distinct phrases)")
