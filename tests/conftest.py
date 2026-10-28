import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HIDDEN = 64
LAYERS = 4


def make_tiny_model(dtype=torch.float32):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=HIDDEN, intermediate_size=128,
                      num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).to(dtype)


@pytest.fixture
def tiny_model():
    return make_tiny_model()


def make_tiny_tokenizer(texts):
    """Word-level tokenizer with a chat template, built offline."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    words = sorted({w for t in texts for w in t.split()})
    vocab = {"[UNK]": 0, "<pad>": 1, "<eos>": 2, "<user>": 3, "<assistant>": 4}
    for w in words:
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="<pad>",
                                  eos_token="<eos>")
    tok.chat_template = (
        "{% for m in messages %}<{{ m['role'] }}> {{ m['content'] }} "
        "{% if m['role'] == 'assistant' %}<eos> {% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<assistant> {% endif %}")
    return tok
