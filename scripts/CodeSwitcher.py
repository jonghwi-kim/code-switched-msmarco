"""
Word-alignment code-switching (thesis Sec. 3.1.2).

The English passage and its translation are aligned with Awesome-Align
(https://huggingface.co/aneuraz/awesome-align-with-co, layer 8, threshold 0.5). Each English content
word (noun/verb/adjective/adverb, not a stop word) that has an alignment is replaced by its aligned
target word(s) with probability 0.5. If several English words align to the same target word, it is
inserted only once.

This module holds the text side (spaCy, switching); the aligner is in WordAligner.py.
"""
import random
import unicodedata

import spacy
from nltk.tokenize import TreebankWordDetokenizer
from nltk.corpus import stopwords
from collections import Counter

detokenizer = TreebankWordDetokenizer()

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

content_word_tags = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
                     "RB", "RBR", "RBS", "NOUN", "VERB", "ADJ", "ADV"}

# Target languages written without spaces between words.
NO_SPACE_LANGS = {"zh", "zho", "ja", "jpn"}


def character_info(c):
    category = unicodedata.category(c)
    if category.startswith("L"):
        name = unicodedata.name(c)
        return name.split(" ")[0]
    else:
        return category[0]


def string_info(s):
    categories = Counter([character_info(c) for c in s])
    if len(categories) > 0:
        most_common, _ = categories.most_common(1)[0]
        return most_common
    else:
        return "Unknown"


def load_english_pipeline():
    # Keep tok2vec + tagger: the tagger listens to tok2vec, disabling it breaks POS tags.
    return spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "textcat"])


def is_content_word(token):
    return (token.tag_ in content_word_tags
            and string_info(token.text) not in non_letter_categories
            and token.text.lower() not in en_stopwords)


def switch_content_words(words, content, target_words, alignment, prob=0.5, target_lang=None):
    """Code-switches one passage. words/content: English tokens and their content-word flags; alignment from
    WordAligner.TextAligner. Uses the global `random` (one draw per aligned content word). Returns (tokens, n_switched)."""
    joiner = "" if target_lang in NO_SPACE_LANGS else " "
    new_sentence = []
    previous_replacement = None
    used_targets = set()
    cs_count = 0
    for idx, word in enumerate(words):
        if idx in alignment and content[idx] and random.random() < prob:
            tgt_indices = tuple(alignment[idx])
            if tgt_indices == previous_replacement:
                # Adjacent English words aligned to the same target word(s): merge into one.
                cs_count += 1
                continue
            if used_targets.intersection(tgt_indices):
                # Target word(s) already inserted for another English word: keep the English word.
                new_sentence.append(word)
                previous_replacement = None
                continue
            new_sentence.append(joiner.join(target_words[j] for j in tgt_indices))
            used_targets.update(tgt_indices)
            previous_replacement = tgt_indices
            cs_count += 1
        else:
            new_sentence.append(word)
            previous_replacement = None
    return new_sentence, cs_count
