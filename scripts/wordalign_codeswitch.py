"""
Word-alignment code-switching of the English collection given its translation (thesis Sec. 3.1.2).
Both files must contain the same pids in the same order.
Output: {output_path}/en_{lang}_mixed_collection_with_parallel_{prob}.tsv

Both files are streamed in chunks. spaCy tokenization/tagging and detokenization run in --workers
processes, overlapping with the batched alignment on the GPU. Switching happens in the main process in
collection order, so a --seed draws the same random numbers whatever --workers is.
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
import random
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor

import spacy
from tqdm.auto import tqdm

from CodeSwitcher import load_english_pipeline, is_content_word, switch_content_words, detokenizer


def load_target_lang_pipeline(lang):
    if lang == 'zh':
        from spacy.lang.zh import Chinese
        cfg = {"segmenter": "jieba"}
        return Chinese.from_config({"nlp": {"tokenizer": cfg}})
    elif lang == 'yo':
        from spacy.lang.yo import Yoruba
        return Yoruba()
    elif lang == 'ru':
        from spacy.lang.ru import Russian
        return Russian()
    elif lang == 'fa':
        from spacy.lang.fa import Persian
        return Persian()
    else:
        # ha, so, sw (Latin script, space-delimited): English tokenization rules.
        return spacy.blank("en")


def count_lines(path):
    with open(path, 'rb') as f:
        return sum(block.count(b'\n') for block in iter(lambda: f.read(1 << 24), b''))


def read_chunks(source_file, target_file, chunk_size):
    """Yields (pids, english passages, target passages); tolerates empty passages (e.g. an empty machine translation)."""
    def parse(row):
        parts = row.rstrip('\n\r').split('\t')
        return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""

    with open(source_file) as source, open(target_file) as target:
        rows = itertools.zip_longest(source, target)
        while chunk := list(itertools.islice(rows, chunk_size)):
            assert all(s is not None and t is not None for s, t in chunk), \
                "The English collection and the translation must have the same number of lines."
            pids, english, translated = [], [], []
            for (pid, en), (tgt_pid, tgt) in ((parse(s), parse(t)) for s, t in chunk):
                assert pid == tgt_pid, f"The English collection and the translation must contain the same pids in the same order ({pid} != {tgt_pid})."
                pids.append(pid); english.append(en); translated.append(tgt)
            yield pids, english, translated


# Worker processes: spaCy pipelines are loaded once per worker.
en_nlp = target_nlp = None


def init_worker(target_lang):
    global en_nlp, target_nlp
    en_nlp, target_nlp = load_english_pipeline(), load_target_lang_pipeline(target_lang)


def tokenize_chunk(english, translated):
    """[(English words, content-word flags, target words)] per passage."""
    return [([t.text for t in en], [is_content_word(t) for t in en], [t.text for t in tgt])
            for en, tgt in zip(en_nlp.pipe(english, batch_size=256), target_nlp.pipe(translated, batch_size=256))]


def detokenize_chunk(sentences):
    return [detokenizer.detokenize(tokens) for tokens in sentences]


def align_chunk(aligner, tokenized):
    """Alignments for one chunk (pairs with an empty side get none)."""
    alignments = [{} for _ in tokenized]
    # Sorted by length so that the pair batches of the similarity step are padded little.
    todo = sorted((i for i, (words, _, target_words) in enumerate(tokenized) if words and target_words),
                  key=lambda i: len(tokenized[i][0]) + len(tokenized[i][2]))
    for i, alignment in zip(todo, aligner.align_word_indices([tokenized[i][0] for i in todo], [tokenized[i][2] for i in todo])):
        alignments[i] = alignment
    return alignments


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_lang", type=str, default="en")
    parser.add_argument("--source_lang_file", type=str, required=True, help="English collection tsv (pid<TAB>passage)")
    parser.add_argument("--target_lang", type=str, required=True, help="zh | fa | ru | ha | so | sw | yo (used for tokenization and the output name)")
    parser.add_argument("--target_lang_file", type=str, required=True, help="translated collection tsv, line-aligned with --source_lang_file")
    parser.add_argument("--output_path", type=str, default='./')
    parser.add_argument("--prob", type=float, default=0.5)
    parser.add_argument("--align_threshold", type=float, default=0.5)
    parser.add_argument("--align_layer", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256, help="sequences per Awesome-Align forward pass")
    parser.add_argument("--chunk_size", type=int, default=2000, help="passages per spaCy worker task")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count()), help="spaCy worker processes")
    parser.add_argument("--dtype", default="auto", help='"auto" (float16 on GPU, float32 on CPU), float32, float16 or bfloat16')
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.SystemRandom().randrange(2 ** 32)
    print(f"seed: {args.seed}")
    random.seed(args.seed)

    output_name = f"{args.output_path}/{args.source_lang}_{args.target_lang}_mixed_collection_with_parallel_{args.prob}"
    stats = Counter()
    # spawn: the workers must not inherit the CUDA context of the main process.
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=init_worker, initargs=(args.target_lang,)) as pool, \
            open(f"{output_name}.tsv", "w") as output_file, tqdm(total=count_lines(args.source_lang_file), unit="passage") as progress:
        from WordAligner import TextAligner   # torch only in the main process
        aligner = TextAligner(align_layer=args.align_layer, threshold=args.align_threshold, dtype=args.dtype, batch_size=args.batch_size)
        tokenizing, detokenizing = deque(), deque()

        def write_done(block):
            # Write finished chunks in collection order; block=True waits for the oldest one.
            while detokenizing and (block or detokenizing[0][1].done()):
                pids, future = detokenizing.popleft()
                output_file.writelines(f"{pid}\t{text}\n" for pid, text in zip(pids, future.result()))
                progress.update(len(pids))
                block = False

        def switch_next():
            pids, future = tokenizing.popleft()
            tokenized = future.result()
            sentences = []
            for (words, content, target_words), alignment in zip(tokenized, align_chunk(aligner, tokenized)):
                tokens, n_switched = switch_content_words(words, content, target_words, alignment, args.prob, args.target_lang)
                sentences.append(tokens)
                stats.update(token=len(words), switching=n_switched)
            stats.update(passage=len(pids))
            detokenizing.append((pids, pool.submit(detokenize_chunk, sentences)))
            write_done(block=len(detokenizing) > 2 * args.workers)

        for pids, english, translated in read_chunks(args.source_lang_file, args.target_lang_file, args.chunk_size):
            tokenizing.append((pids, pool.submit(tokenize_chunk, english, translated)))
            if len(tokenizing) > 2 * args.workers:   # keep the workers busy while the GPU aligns
                switch_next()
        while tokenizing:
            switch_next()
        while detokenizing:
            write_done(block=True)

    metadata = {'avg_n_token': stats['token'] / stats['passage'], 'avg_n_switching': stats['switching'] / stats['passage']}
    print(metadata)
    with open(f"{output_name}_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
