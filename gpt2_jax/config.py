from flax import struct

@struct.dataclass
class GPT2ModelConfig:
    vocab_size: int = 50257
    context_size: int = 1024
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
