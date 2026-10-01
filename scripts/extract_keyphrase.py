"""
Keyphrase extraction and selection (thesis Sec. 3.1.3).

1. KeyBERT (sentence-transformers/all-MiniLM-L12-v2, MMR diversity 0.5, English stop words,
   1-3-gram) extracts up to 20 keyphrases per passage -> keyphrase_{n}gram.json
   (KeyBERT's extract_keywords is reimplemented below with the same computations and a faster MMR loop.)
2. Each keyphrase is located in the passage (its first occurrence; case-insensitive, not across punctuation),
   overlaps are resolved in KeyBERT rank order, and each remaining keyphrase is selected with probability 0.5
   -> keyphrase_{n}gram_selected.json  {pid: [[start, end, phrase with the passage's casing], ...]}
   (character offsets into the stripped passage; only this occurrence is replaced later)
"""
import re
import json
import random
import argparse
import os
import unicodedata

import numpy as np
from tqdm.auto import tqdm
from nltk.corpus import stopwords
from collections import Counter
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer

en_stopwords = set(stopwords.words('english'))

non_letter_categories = {
    "C": "Control",
    "M": "Mark",
    "N": "Number",
    "P": "Punctuation",
    "S": "Symbol",
    "Z": "Separator",
    "Unknown": "Unknown",
}

# Same token definition as KeyBERT's default CountVectorizer (words of >= 2 word characters).
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
# Characters allowed between two tokens of one keyphrase (no other punctuation).
GAP_CHARS_RE = re.compile(r"[\s\-'’\w]*")
# With stop_words="english" KeyBERT drops stop words before building n-grams. Inside a noun phrase only these
# may be skipped ("United States of America", "Indian cultures of the Northwest Coast"), at most 2 per gap.
NOUN_PHRASE_FUNCTION_WORDS = frozenset({"of", "the", "and"})
MAX_SKIPPED = 2


def character_info(c):
    category = unicodedata.category(c)
    return category[0] if not category.startswith("L") else unicodedata.name(c).split(" ")[0]

def string_info(s):
    categories = Counter(character_info(c) for c in s)
    return categories.most_common(1)[0][0] if categories else "Unknown"

def load_collection(file_path):
    collection = {}
    with open(file_path, "r") as f:
        for row in f:
            pid, passage, *rest = row.strip('\n\r ').split('\t')
            collection[pid] = passage.strip()
    return collection

def normalize_rows(x):
    # sklearn.preprocessing.normalize (as used by cosine_similarity), row by row.
    norms = np.sqrt(np.einsum("ij,ij->i", x, x))
    norms[norms == 0.0] = 1.0
    return x / norms[:, np.newaxis]

def mmr(doc_embedding, word_embeddings, top_n, diversity):
    """keybert._mmr.mmr on normalized embeddings, returning (index, rounded similarity) sorted like KeyBERT.
    Same float32 operations, so the same result; the max similarity to the selected keyphrases is kept up to
    date instead of being recomputed for every candidate in every step."""
    word_doc_similarity = (word_embeddings @ doc_embedding.T)[:, 0]
    word_similarity = word_embeddings @ word_embeddings.T
    keywords_idx = [int(np.argmax(word_doc_similarity))]
    available = np.ones(len(word_embeddings), dtype=bool)
    available[keywords_idx[0]] = False
    target_similarities = word_similarity[:, keywords_idx[0]].copy()
    for _ in range(min(top_n - 1, len(word_embeddings) - 1)):
        scores = (1 - diversity) * word_doc_similarity - diversity * target_similarities
        scores[~available] = -np.inf
        mmr_idx = int(np.argmax(scores))   # first maximum, as np.argmax over KeyBERT's ordered candidate list
        keywords_idx.append(mmr_idx)
        available[mmr_idx] = False
        np.maximum(target_similarities, word_similarity[:, mmr_idx], out=target_similarities)
    keywords = [(idx, round(float(word_doc_similarity[idx]), 4)) for idx in keywords_idx]
    return sorted(keywords, key=itemgetter(1), reverse=True)

def extract_keywords(model, docs, n_gram, top_n, diversity, stop_words, embed_batch_size):
    """KeyBERT(model).extract_keywords(docs, keyphrase_ngram_range=(1, n_gram), top_n=top_n, use_mmr=True,
    diversity=diversity, stop_words=stop_words), without KeyBERT's per-document overhead."""
    try:
        count = CountVectorizer(ngram_range=(1, n_gram), stop_words=stop_words).fit(docs)
    except ValueError:   # empty vocabulary
        return [[] for _ in docs]
    words = count.get_feature_names_out()
    df = count.transform(docs)   # CSR with sorted indices: candidates in vocabulary order, as in KeyBERT
    doc_embeddings = normalize_rows(model.encode(docs, batch_size=embed_batch_size, show_progress_bar=False))
    word_embeddings = normalize_rows(model.encode(words, batch_size=embed_batch_size, show_progress_bar=False))
    all_keywords = []
    for i in range(len(docs)):
        candidate_indices = df.indices[df.indptr[i]:df.indptr[i + 1]]
        if len(candidate_indices) == 0:
            all_keywords.append([])
            continue
        keywords = mmr(doc_embeddings[i:i + 1], word_embeddings[candidate_indices], top_n, diversity)
        all_keywords.append([(words[candidate_indices[idx]], score) for idx, score in keywords])
    return all_keywords

def process_keywords_in_batches(model, eng_collection, n_gram, n_candidates, diversity, batch_size=10, stop_words=None, embed_batch_size=32):
    keyword_dict = {}
    eng_values = list(eng_collection.values())
    eng_items = list(eng_collection.items())

    for i in tqdm(range(0, len(eng_values), batch_size), desc="Processing Batches"):
        batch_values = eng_values[i:i + batch_size]
        batch_items = eng_items[i:i + batch_size]

        # Extract keywords for the current batch
        batch_keywords = extract_keywords(model, batch_values, n_gram, n_candidates * 2, diversity, stop_words, embed_batch_size)

        # Process and filter keywords
        batch_keywords = [[x[0] for x in keywords if string_info(x[0]) not in non_letter_categories and x[0] not in en_stopwords]
                          for keywords, doc in zip(batch_keywords, batch_values)]

        # Limit the number of keywords per document/query
        batch_keywords = [keywords[:n_candidates] for keywords in batch_keywords]

        # Update keyword_dict with the processed keywords for each document
        for (doc_id, doc), keywords in zip(batch_items, batch_keywords):
            keyword_dict[doc_id] = keywords

    return keyword_dict

def valid_gap(gap, skippable):
    # Whitespace, hyphens, apostrophes, one-character words and skippable words only.
    return bool(GAP_CHARS_RE.fullmatch(gap)) and all(len(w) == 1 or w.lower() in skippable for w in re.findall(r"\w+", gap))

def locate_keyphrase(text, text_tokens, keyphrase, skippable=frozenset()):
    """(start, end) of the first occurrence of the lower-cased KeyBERT n-gram in `text`, or None."""
    kp_tokens = keyphrase.split()
    for i, token in enumerate(text_tokens):
        if token[0] != kp_tokens[0]:
            continue
        j, found = i, True
        for kp_token in kp_tokens[1:]:
            nxt = j + 1
            while nxt < len(text_tokens) and nxt - j <= MAX_SKIPPED and text_tokens[nxt][0] != kp_token and text_tokens[nxt][0] in skippable:
                nxt += 1
            if nxt >= len(text_tokens) or text_tokens[nxt][0] != kp_token or not valid_gap(text[text_tokens[j][2]:text_tokens[nxt][1]], skippable):
                found = False
                break
            j = nxt
        if found:
            return text_tokens[i][1], text_tokens[j][2]
    return None

def select_keyphrases(source_collection, keyword_dict, prob=0.5, skippable=frozenset()):
    selected_dict = {}
    n_candidates, n_selected = 0, 0

    for source_id, source_text in tqdm(source_collection.items(), total=len(source_collection)):
        if source_id not in keyword_dict:
            continue

        text_tokens = [(m.group(0).lower(), m.start(), m.end()) for m in TOKEN_RE.finditer(source_text)]
        used, spans = [], []
        for keyphrase in keyword_dict[source_id]:           # KeyBERT rank order
            span = locate_keyphrase(source_text, text_tokens, keyphrase, skippable)
            if span is None or any(span[0] < e and s < span[1] for s, e in used):
                continue
            used.append(span)
            spans.append(span)

        n_candidates += len(spans)
        chosen = [[s, e, source_text[s:e]] for s, e in spans if random.random() < prob]
        n_selected += len(chosen)
        selected_dict[source_id] = chosen

    print({"avg_n_non_overlapping_keyphrases": n_candidates / max(1, len(selected_dict)),
           "avg_n_selected_keyphrases": n_selected / max(1, len(selected_dict))})
    return selected_dict

def write_json(obj, path):
    # Write to a temporary file first so that an interrupted run never leaves a partial file.
    with open(path + ".tmp", 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_lang_file", type=str, required=True, help="English collection tsv (pid<TAB>passage)")
    parser.add_argument("--input_keyword", type=str, default=None, help="skip extraction and reuse an existing keyphrase_*.json")
    parser.add_argument("--output_path", type=str, default='./')
    parser.add_argument("--n_gram", type=int, default=3)
    parser.add_argument("--stop_words", type=str, default="english", help='"english" or "none"')
    parser.add_argument("--n_candidates", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--diversity", type=float, default=0.5)
    parser.add_argument("--prob", type=float, default=0.5, help="probability of translating each keyphrase")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--embed_batch_size", type=int, default=1024, help="sentence-transformers encode() batch size")
    parser.add_argument("--dtype", default="float32", help="float32 or float16 (GPU) for the embedding model")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.SystemRandom().randrange(2 ** 32)
    print(f"seed: {args.seed}")
    random.seed(args.seed)

    source_collection = load_collection(args.source_lang_file)
    stop_words = None if args.stop_words.lower() == "none" else args.stop_words

    if args.input_keyword:
        with open(args.input_keyword, 'r') as f:
            keyword_dict = json.load(f)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L12-v2")   # KeyBERT's embedding model
        if args.dtype == "float16":
            model.half()
        output_file = os.path.join(args.output_path, f"keyphrase_{args.n_gram}gram.json")
        print(f"Extracting keyphrases for {len(source_collection)} documents from {args.source_lang_file}")

        keyword_dict = process_keywords_in_batches(model, source_collection, args.n_gram, args.n_candidates, args.diversity, args.batch_size, stop_words, args.embed_batch_size)
        write_json(keyword_dict, output_file)

    skippable = NOUN_PHRASE_FUNCTION_WORDS if stop_words == "english" else frozenset()
    selected_keyword_dict = select_keyphrases(source_collection, keyword_dict, args.prob, skippable)
    output_file = os.path.join(args.output_path, f"keyphrase_{args.n_gram}gram_selected.json")
    write_json(selected_keyword_dict, output_file)
    print(f"keywords are saved to {output_file}")
