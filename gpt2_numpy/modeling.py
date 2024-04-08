import numpy as np
from typing import Mapping

from .config import GPT2SmallModelConfig

class NumPyGPT2CausalLM:
    def __init__(
        self, model_config: GPT2SmallModelConfig, model_params: Mapping[str, np.ndarray]
    ) -> None:
        self.config = model_config
        self.params = model_params

    def __call__(self, input_ids: np.ndarray):
        """
        input_ids: (seq_len,) int64

        Note: We don't support batching here so there is no `batch_size`.
        """
        # Use breakpoint to inspect/debug.
        # breakpoint()
        return self.lm(input_ids)

    def lm(self, input_ids):
        """input_ids: (seq_len,) int64"""
        x = self.transformer(input_ids)

        # lm_head has no bias params.
        lm_logits = x @ self.params["lm_head.weight"].T
        return lm_logits

    def transformer(self, input_ids):
        """input_ids: (seq_len,) int64"""
        pos_emb = self.params["transformer.wpe.weight"][np.arange(len(input_ids))]
        tok_emb = self.params["transformer.wte.weight"][input_ids]
        x = tok_emb + pos_emb

        for block_idx in range(self.config.n_layer):
            # Extract params of this block
            block_params: Mapping[str, np.ndarray] = {}
            prefix = f"transformer.h.{block_idx}."
            for k in self.params.keys():
                if k.startswith(prefix):
                    block_params[k[len(prefix) :]] = self.params[k]

            x = self.block(x, block_params)
        return self.layer_norm(
            x,
            self.params["transformer.ln_f.weight"],
            self.params["transformer.ln_f.bias"],
        )

    def block(self, x, block_params):
        """x: (seq_len, n_embd)"""
        x = x + self.mha(
            self.layer_norm(x, block_params["ln_1.weight"], block_params["ln_1.bias"]),
            block_params,
        )
        x = x + self.mlp(
            self.layer_norm(x, block_params["ln_2.weight"], block_params["ln_2.bias"]),
            block_params,
        )
        return x

    def mha(self, x, block_params):
        """
        Multi-head causal self-attention.

        x: (seq_len, n_embd)
        """
        seq_len = x.shape[-2]
        x = self.linear(
            x, block_params["attn.c_attn.weight"], block_params["attn.c_attn.bias"]
        )
        q, k, v = np.split(x, 3, axis=-1)

        # Split heads to (n_head, seq_len, head_size)
        q_heads = np.array(np.split(q, self.config.n_head, axis=-1))
        k_heads = np.array(np.split(k, self.config.n_head, axis=-1))
        v_heads = np.array(np.split(v, self.config.n_head, axis=-1))

        causal_mask = (1 - np.tri(seq_len, dtype=x.dtype)) * -1e10
        x = self.attention(q_heads, k_heads, v_heads, causal_mask)
        x = np.hstack(x)

        x = self.linear(
            x,
            block_params["attn.c_proj.weight"],
            block_params["attn.c_proj.bias"],
        )
        return x

    def attention(self, q, k, v, mask):
        """
        Scaled dot product attention.

        q: (n_head, seq_len, head_size)
        k: (n_head, seq_len, head_size)
        v: (n_head, seq_len, head_size)
        mask: (seq_len, seq_len) with -1e10 for invalid positions on the top right traiangle.
        """

        # Considering n_head and head_size constants, the runtime complexity of the two matmuls is O(seq_len^2).

        d_k = k.shape[-1]  # head_size
        return self.softmax(q @ np.swapaxes(k, -1, -2) / np.sqrt(d_k) + mask) @ v

    def mlp(self, x, block_params):
        """
        Poswise feedforward network.

        x: (seq_len, n_embd)
        """
        # Project up 4x
        x = self.linear(
            x, block_params["mlp.c_fc.weight"], block_params["mlp.c_fc.bias"]
        )
        # Project back down
        return self.linear(
            self.gelu(x),
            block_params["mlp.c_proj.weight"],
            block_params["mlp.c_proj.bias"],
        )

    def linear(self, x, w, b):
        return x @ w + b

    def layer_norm(self, x, g, b, eps: float = 1e-5):
        """
        Layer normalization.

        x: (seq_len, n_embd)
        g: (n_embd,)
        b: (n_embd,)

        eps: avoid dividing by 0 with a small epsilon.
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        # Normalize x to have a mean of 0 and a variance of 1.
        x = (x - mean) / np.sqrt(variance + eps)

        # Scale and offset with learned gamma/beta params.
        return g * x + b

    def softmax(self, x):
        """Converts logits to probability distribution."""
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def gelu(self, x):
        """NewGELU activation."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
