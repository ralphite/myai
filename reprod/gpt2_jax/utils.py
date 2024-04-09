from functools import cache

import jax.numpy as jnp
from transformers import GPT2LMHeadModel, GPT2Tokenizer

_HF_CACHE_DIR = "./.hf_cache"


@cache
def load_hf_tokenizer():
    tokenizer_hf = GPT2Tokenizer.from_pretrained(
        "openai-community/gpt2", cache_dir=_HF_CACHE_DIR
    )

    return tokenizer_hf


@cache
def load_hf_model():
    model_hf = GPT2LMHeadModel.from_pretrained(
        "openai-community/gpt2", cache_dir=_HF_CACHE_DIR
    )
    return model_hf


@cache
def load_hf_pretrained_params():
    """
    Load the pretrained parameters from HuggingFace and convert them to Flax format as defined in `model.py`.
    """
    model_hf = load_hf_model()
    params_hf = {k: jnp.array(v.numpy()) for k, v in model_hf.state_dict().items()}

    params_flax = {
        "transformer": {
            "wte": {"embedding": params_hf["transformer.wte.weight"]},
            "wpe": {"embedding": params_hf["transformer.wpe.weight"]},
            "ln_f": {
                "scale": params_hf["transformer.ln_f.weight"],
                "bias": params_hf["transformer.ln_f.bias"],
            },
        },
        "lm_head": {"kernel": params_hf["lm_head.weight"].T},
    }

    for block_idx in range(12):
        block_params_hf = {}
        prefix = f"transformer.h.{block_idx}."
        for k, v in params_hf.items():
            if k.startswith(prefix):
                block_params_hf[k[len(prefix) :]] = v

        block_params = {
            "ln_1": {
                "scale": block_params_hf["ln_1.weight"],
                "bias": block_params_hf["ln_1.bias"],
            },
            "attn": {
                "c_attn": {
                    "kernel": block_params_hf["attn.c_attn.weight"],
                    "bias": block_params_hf["attn.c_attn.bias"],
                },
                "c_proj": {
                    "kernel": block_params_hf["attn.c_proj.weight"],
                    "bias": block_params_hf["attn.c_proj.bias"],
                },
            },
            "ln_2": {
                "scale": block_params_hf["ln_2.weight"],
                "bias": block_params_hf["ln_2.bias"],
            },
            "mlp": {
                "c_fc": {
                    "kernel": block_params_hf["mlp.c_fc.weight"],
                    "bias": block_params_hf["mlp.c_fc.bias"],
                },
                "c_proj": {
                    "kernel": block_params_hf["mlp.c_proj.weight"],
                    "bias": block_params_hf["mlp.c_proj.bias"],
                },
            },
        }

        params_flax["transformer"][f"block_{block_idx}"] = block_params

    return {"params": params_flax}
