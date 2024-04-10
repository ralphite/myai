from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class GPT2ModelConfig:
    vocab_size: int = 50257  # V
    context_size: int = 1024
    n_layer: int = 12  # D
    n_embd: int = 768  # C
    n_head: int = 12  # H


GPT2_SM_MODEL_CONFIG = GPT2ModelConfig()
RNG_KEY = jax.random.PRNGKey(12)
