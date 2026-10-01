"""
Lexicon-based code-switching of the English MS MARCO collection (thesis Sec. 3.1.1).

Every n-gram (up to 5 tokens) whose lemma is in the lexicon is a candidate; overlapping candidates
are resolved (earliest, then longest); each candidate is replaced with probability `prob` by a
translation drawn at random from the lexicon.
Output: {output}/random_prob_{prob}_switching_1_{lang}_collection.tsv  (pid<TAB>passage)

Passages are processed by --workers processes. Each passage has its own random generator seeded with
(--seed, pid), so the output for a given seed does not depend on --workers.
"""
import os
import json
import random
import argparse
import itertools
import threading
import contextlib
import unicodedata
import multiprocessing as mp

from functools import lru_cache
from collections import Counter
from tqdm.auto import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer

en_stopwords = set(stopwords.words('english'))
tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()
lemmatizer = WordNetLemmatizer()


def is_letter_string(s):
    # Majority character class must be a letter (drops numbers, punctuation, symbols).
    categories = Counter(unicodedata.category(c)[0] for c in s)
    return bool(categories) and categories.most_common(1)[0][0] == "L"


class Lexicon:
    """Candidate lookup for extract_candidates.

    A multi-word string is never a WordNet form (WordNet joins words with "_"), so the lemma of an n-gram
    with n >= 2 is the n-gram itself, lower-cased. Its lexicon entry therefore starts with the lower-cased
    first token, which lets n-grams that cannot match be skipped without lemmatizing or joining them.
    """

    def __init__(self, lexicon, max_n_gram):
        self.lexicon = lexicon
        lengths = Counter(len(entry.split(' ')) for entry in lexicon)
        self.max_n_gram = min(max_n_gram, max(lengths))
        self.multiword_starts = {entry.split(' ')[0] for entry in lexicon if ' ' in entry}
        self.unigram = lru_cache(maxsize=1 << 18)(self._unigram)   # Zipfian vocabulary: high hit rate

    def _unigram(self, token):
        lower = token.lower()
        lemma = lemmatizer.lemmatize(lower)
        if lemma in self.lexicon and lower not in en_stopwords and is_letter_string(token):
            return self.lexicon[lemma]
        return None


def extract_candidates(tokens, lexicon):
    candidates = []
    for n_gram in range(lexicon.max_n_gram, 1, -1):
        for offset in range(len(tokens) - n_gram + 1):
            if tokens[offset].lower() not in lexicon.multiword_starts:
                continue
            word = ' '.join(tokens[offset:offset + n_gram])
            lower = word.lower()
            if lower in lexicon.lexicon and is_letter_string(word):   # multi-word strings are never stop words
                candidates.append({"translations": lexicon.lexicon[lower], "n_gram": n_gram, "offset": offset})
    for offset, token in enumerate(tokens):
        translations = lexicon.unigram(token)
        if translations is not None:
            candidates.append({"translations": translations, "n_gram": 1, "offset": offset})
    # Keep non-overlapping candidates: earliest first, longest first at the same position.
    candidates.sort(key=lambda x: (x['offset'], -x['n_gram']))
    kept, used = [], set()
    for c in candidates:
        span = range(c['offset'], c['offset'] + c['n_gram'])
        if not any(i in used for i in span):
            kept.append(c)
            used.update(span)
    return kept


def codeswitch(tokens, candidates, prob, rng=random):
    replacements = [(c, rng.sample(c['translations'], k=1)[0]) for c in candidates if rng.random() < prob]
    tokens = list(tokens)
    delete = []
    for c, target in replacements:
        tokens[c['offset']] = target
        delete.extend(range(c['offset'] + 1, c['offset'] + c['n_gram']))
    for i in sorted(delete, reverse=True):
        del tokens[i]
    return detokenizer.detokenize(tokens), len(replacements)


# Set in the main process before the workers are forked.
LEXICON, PROB, SEED = None, None, None


def process_lines(lines):
    """Code-switches a block of collection lines. Returns (output lines, (n_tokens, n_candidates, n_switched))."""
    output, n_tokens, n_candidates, n_switched = [], 0, 0, 0
    for line in lines:
        pid, passage, *rest = line.strip('\n\r ').split('\t')
        if rest:
            passage = rest[0] + ' | ' + passage
        tokens = tokenizer.tokenize(passage)
        candidates = extract_candidates(tokens, LEXICON)
        text, k = codeswitch(tokens, candidates, PROB, random.Random(f"{SEED}:{pid}"))
        output.append(f"{pid}\t{text}\n")
        n_tokens += len(tokens); n_candidates += len(candidates); n_switched += k
    return output, (n_tokens, n_candidates, n_switched)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', required=True, help="English collection tsv (pid<TAB>passage)")
    parser.add_argument('--lexicon', required=True, help="lexicon json {english word: [translations]}")
    parser.add_argument('--lang', required=True, help="used in the output name, e.g. en-zh or hau")
    parser.add_argument('--output', required=True, help="output directory")
    parser.add_argument('--prob', type=float, default=0.5)
    parser.add_argument('--n_gram', type=int, default=5, help="maximum n-gram length matched against the lexicon")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--block_size', type=int, default=1000, help="lines per worker task")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.SystemRandom().randrange(2 ** 32)
    print(f"seed: {args.seed}")
    with open(args.lexicon, encoding='utf-8') as f:
        LEXICON = Lexicon(json.load(f), args.n_gram)
    PROB, SEED = args.prob, args.seed
    lemmatizer.lemmatize("warmup")   # load WordNet once, before forking

    with open(args.collection, 'rb') as f:
        n_lines = sum(block.count(b'\n') for block in iter(lambda: f.read(1 << 24), b''))
    output_file = f"{args.output}/random_prob_{args.prob}_switching_1_{args.lang}_collection.tsv"
    n_passages = n_tokens = n_candidates = n_switched = 0
    with open(args.collection) as fin, open(output_file, 'w', encoding='utf-8') as fout, \
            tqdm(total=n_lines, unit="passage", desc=args.lang) as progress, \
            (mp.get_context("fork").Pool(args.workers) if args.workers > 1 else contextlib.nullcontext()) as pool:
        # Pool.imap reads its input as fast as it can: bound the number of blocks in flight.
        in_flight = threading.BoundedSemaphore(4 * args.workers)

        def blocks():
            while lines := list(itertools.islice(fin, args.block_size)):
                in_flight.acquire()
                yield lines

        for lines, (t, c, k) in (pool.imap(process_lines, blocks()) if pool else map(process_lines, blocks())):
            in_flight.release()
            fout.writelines(lines)
            n_passages += len(lines); n_tokens += t; n_candidates += c; n_switched += k
            progress.update(len(lines))

    print({"avg_n_token": n_tokens / n_passages, "avg_n_candidate": n_candidates / n_passages, "avg_n_switched": n_switched / n_passages})
    print(f"{output_file} is done")
