from dataclasses import dataclass

@dataclass(frozen=True)
class GPT2SmallModelConfig:
    vocab_size: int = 50257
    context_size: int = 1024
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
