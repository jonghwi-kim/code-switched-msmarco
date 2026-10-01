import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def resolve_dtype(dtype, device):
    """"auto": float16 on GPU, float32 on CPU."""
    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return getattr(torch, dtype)


class NLLBTranslation:
    """English -> target translation with NLLB-200 (https://huggingface.co/facebook/nllb-200-distilled-1.3B)."""

    def __init__(self, model_name="facebook/nllb-200-distilled-1.3B", target_lang="eng_Latn", n_beams=5, max_length=256, num_return=1, repetition_penalty=1.5, dtype="auto"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)   # source language: eng_Latn (default)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=resolve_dtype(dtype, self.device)).to(self.device).eval()
        self.target_lang = target_lang
        self.n_beams = n_beams
        self.max_length = max_length
        self.num_return = num_return
        self.repetition_penalty = repetition_penalty

    def translate(self, batch_list):
        """Returns one list of `num_return` (translation, finished) pairs per input text. `finished` is False when
        the generation reached max_length without emitting </s> (e.g. an NLLB repetition loop)."""
        tokenized_text = self.tokenizer(batch_list, max_length=self.max_length, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            translated = self.model.generate(
                input_ids=tokenized_text['input_ids'].to(self.device),
                attention_mask=tokenized_text['attention_mask'].to(self.device),
                max_length=self.max_length,
                num_beams=self.n_beams,
                num_return_sequences=self.num_return,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.target_lang),
                do_sample=False,
                repetition_penalty=self.repetition_penalty
            )
        # Position 0 is the decoder start token (also </s> for NLLB); a finished sequence has </s> after it.
        finished = (translated[:, 1:] == self.tokenizer.eos_token_id).any(dim=1).tolist()
        translated_words = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        pairs = list(zip(translated_words, finished))
        return [pairs[i:i+self.num_return] for i in range(0, len(pairs), self.num_return)]
