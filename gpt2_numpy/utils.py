from dataclasses import dataclass
from functools import cache
from transformers import GPT2Tokenizer, GPT2LMHeadModel


@cache
def load_tokenizer():
    tokenizer_hf = GPT2Tokenizer.from_pretrained(
        "openai-community/gpt2", cache_dir="./cache"
    )

    return tokenizer_hf


@cache
def load_hf_model():
    model_hf = GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2", cache_dir="./cache"
    )
    return model_hf


@cache
def load_model_params():
    model_hf = load_hf_model()
    return {k: v.numpy() for k, v in model_hf.state_dict().items()}
