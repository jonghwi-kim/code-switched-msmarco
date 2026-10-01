"""
Awesome-Align word alignment (https://huggingface.co/aneuraz/awesome-align-with-co, softmax extraction).
Sequences are encoded in batches of similar length, so that there is little padding.
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resolve_dtype(dtype):
    """"auto": float16 on GPU, float32 on CPU."""
    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return getattr(torch, dtype)


class TextAligner:
    def __init__(self, align_layer=8, threshold=0.5, dtype="auto", batch_size=256):
        self.model = AutoModel.from_pretrained("aneuraz/awesome-align-with-co", torch_dtype=resolve_dtype(dtype)).to(device).eval()
        # Only the output of layer `align_layer` is used: drop the layers above it and the pooler.
        self.model.encoder.layer = self.model.encoder.layer[:align_layer]
        self.model.pooler = None
        self.tokenizer = AutoTokenizer.from_pretrained("aneuraz/awesome-align-with-co")
        self.threshold = threshold
        self.batch_size = batch_size

    @torch.no_grad()
    def embed(self, word_lists):
        """Per word list: layer-`align_layer` states of its subwords ([CLS]/[SEP] removed) and the word index of
        each subword. Each sequence is truncated to the model's 512 positions, as in Awesome-Align."""
        encoding = self.tokenizer(word_lists, is_split_into_words=True, truncation=True)
        input_ids = encoding['input_ids']
        word_ids = [torch.tensor(encoding.word_ids(i)[1:-1], dtype=torch.long) for i in range(len(word_lists))]
        hidden = [None] * len(word_lists)
        order = sorted(range(len(word_lists)), key=lambda i: len(input_ids[i]))
        for k in range(0, len(order), self.batch_size):
            batch = order[k:k + self.batch_size]
            padded = self.tokenizer.pad({'input_ids': [input_ids[i] for i in batch]}, return_tensors='pt')
            states = self.model(input_ids=padded['input_ids'].to(device),
                                attention_mask=padded['attention_mask'].to(device)).last_hidden_state
            for row, i in enumerate(batch):
                hidden[i] = states[row, 1:len(input_ids[i]) - 1]
        return hidden, word_ids

    @torch.no_grad()
    def align_word_indices(self, src_batch, tgt_batch):
        """Aligns each (source words, target words) pair, none empty. Returns one
        {src word index: sorted list of aligned tgt word indices} per pair."""
        n = len(src_batch)
        hidden, word_ids = self.embed(src_batch + tgt_batch)
        alignments = []
        for k in range(0, n, self.batch_size):
            pairs = range(k, min(n, k + self.batch_size))
            src = pad_sequence([hidden[i] for i in pairs], batch_first=True).float()
            tgt = pad_sequence([hidden[n + i] for i in pairs], batch_first=True).float()
            src_ids = pad_sequence([word_ids[i] for i in pairs], batch_first=True, padding_value=-1)
            tgt_ids = pad_sequence([word_ids[n + i] for i in pairs], batch_first=True, padding_value=-1)
            valid = ((src_ids >= 0)[:, :, None] & (tgt_ids >= 0)[:, None, :]).to(device)

            dot_prod = torch.bmm(src, tgt.transpose(1, 2)).masked_fill(~valid, torch.finfo(torch.float32).min)
            softmax_inter = (dot_prod.softmax(dim=-1) > self.threshold) & (dot_prod.softmax(dim=-2) > self.threshold) & valid

            batch_alignments = [{} for _ in pairs]
            pair, i, j = (x.cpu() for x in softmax_inter.nonzero(as_tuple=True))
            for b, s, t in zip(pair.tolist(), src_ids[pair, i].tolist(), tgt_ids[pair, j].tolist()):
                batch_alignments[b].setdefault(s, set()).add(t)
            alignments.extend({s: sorted(ts) for s, ts in alignment.items()} for alignment in batch_alignments)
        return alignments
