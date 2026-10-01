"""
Replace the selected keyphrases of each passage by their translations (thesis Sec. 3.1.3).
Only the selected occurrence of each keyphrase is replaced (the span recorded by extract_keyphrase.py);
other occurrences of the same words are kept. Sentence-final punctuation added by NLLB is stripped, and
degenerate translations (repetition loops) are skipped (English kept).
Output: pid<TAB>passage, same pids and order as the input collection.
"""
import re
import json
import argparse
import unicodedata

from tqdm.auto import tqdm

TRAILING_PUNCT = ".。!！?？,，;；:：、"
# A unit containing a letter or combining mark repeated at least 3 times in a row
# (digits such as "000" are not a repetition loop).
REPEAT_RE = re.compile(r'(\S.{0,20}?)(?:\s*\1){2,}')


def is_degenerate(translation):
    return any(any(unicodedata.category(c)[0] in "LM" for c in m.group(1)) for m in REPEAT_RE.finditer(translation))


def read_collection(collection_path):
    """Yields (pid, passage, title or None), streaming. The passage is stripped, as in extract_keyphrase.py,
    whose spans index into it."""
    with open(collection_path) as f:
        for line in f:
            pid, passage, *rest = line.strip('\n\r ').split('\t')
            yield pid, passage.strip(), rest[0] if rest else None


def count_lines(path):
    with open(path, 'rb') as f:
        return sum(block.count(b'\n') for block in iter(lambda: f.read(1 << 24), b''))


def normalize(source, translation):
    translation = translation.replace('\t', ' ').replace('\n', ' ').strip()
    if source and source[-1] not in TRAILING_PUNCT:
        translation = translation.rstrip(TRAILING_PUNCT).strip()
    return translation


def switch_spans(text, spans):
    """spans: non-overlapping (start, end, translation). Replaces each span of `text` by its translation."""
    pieces, last = [], 0
    for start, end, translation in sorted(spans):
        pieces += [text[last:start], translation]
        last = end
    pieces.append(text[last:])
    return ''.join(pieces)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_lang_file", type=str, required=True, help="English collection tsv (pid<TAB>passage)")
    parser.add_argument("--keyword_file", type=str, required=True, help="output of translate_keyphrase.py")
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.keyword_file, encoding="utf-8") as f:
        keywords = json.load(f)
    if isinstance(next((items[0] for items in keywords.values() if items), None), dict):
        raise SystemExit(f"{args.keyword_file} has the old {{phrase: translation}} format: re-run extract_keyphrase.py and translate_keyphrase.py")

    n_passages, n_switched, n_degenerate, n_mismatch = 0, 0, 0, 0
    with open(args.output_file, 'w', encoding="utf-8") as out:
        for pid, text, title in tqdm(read_collection(args.source_lang_file), total=count_lines(args.source_lang_file), unit="passage"):
            spans = []
            for start, end, src, tgt in keywords.get(pid, []):
                tgt = normalize(src, tgt)
                if text[start:end] != src:      # spans come from another version of the collection
                    n_mismatch += 1
                elif is_degenerate(tgt):
                    n_degenerate += 1
                elif tgt:
                    spans.append((start, end, tgt))
            switched = switch_spans(text, spans)
            n_switched += len(spans)
            n_passages += 1
            out.write(f"{pid}\t{switched if title is None else title + ' | ' + switched}\n")

    print({"avg_n_switched_phrases": n_switched / max(1, n_passages), "n_degenerate_translations_skipped": n_degenerate})
    if n_mismatch:
        print(f"WARNING: {n_mismatch} keyphrase spans do not match the collection text and were skipped")
    print(f"Code-switched collection written to {args.output_file}")
