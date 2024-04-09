import flax.linen as nn
import jax.numpy as jnp
from config import GPT2ModelConfig


class GPT2LMHeadModel(nn.Module):
    config: GPT2ModelConfig

    @nn.compact
    def __call__(self, input_ids):
        output = GPT2Model(name="transformer", config=self.config)(input_ids)

        lm_logits = nn.Dense(
            name="lm_head", features=self.config.vocab_size, use_bias=False
        )(output)
        return lm_logits


class GPT2Model(nn.Module):
    config: GPT2ModelConfig

    @nn.compact
    def __call__(self, input_ids):
        input_embeddings = nn.Embed(
            name="wte",
            num_embeddings=self.config.vocab_size,
            features=self.config.n_embd,
        )(input_ids)
        position_ids = jnp.arange(start=0, stop=input_ids.shape[-1])
        position_embeddings = nn.Embed(
            name="wpe",
            num_embeddings=self.config.context_size,
            features=self.config.n_embd,
        )(position_ids)
        x = input_embeddings + position_embeddings

        for block_id in range(self.config.n_layer):
            x = GPT2Block(name=f"block_{block_id}", config=self.config)(x)

        # final layer normalization
        x = nn.LayerNorm(name="ln_f")(x)
        return x


class GPT2Block(nn.Module):
    config: GPT2ModelConfig

    @nn.compact
    def __call__(self, x):
        x = x + GPT2MultiHeadAttention(name="attn", config=self.config)(
            nn.LayerNorm(name="ln_1")(x)
        )
        x = x + GPT2MLP(name="mlp", config=self.config)(nn.LayerNorm(name="ln_2")(x))
        return x


class GPT2MultiHeadAttention(nn.Module):
    config: GPT2ModelConfig

    def setup(self):
        self.causal_mask = jnp.tril(
            jnp.ones((self.config.context_size, self.config.context_size))
        )

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(name="c_attn", features=self.config.n_embd * 3)(x)

        q, k, v = jnp.split(x, 3, axis=-1)
        new_shape = x.shape[:-1] + (
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
        )
        q = jnp.reshape(q, new_shape).transpose((0, 2, 1, 3))
        k = jnp.reshape(k, new_shape).transpose((0, 2, 1, 3))
        v = jnp.reshape(v, new_shape).transpose((0, 2, 1, 3))

        attn_weights = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(
            self.config.n_embd // self.config.n_head
        )
        attn_weights = jnp.where(
            self.causal_mask[: x.shape[-2], : x.shape[-2]].astype(bool),
            attn_weights,
            -1e4,
        )
        attn_weights = nn.softmax(attn_weights, axis=-1)
        attn_weights = attn_weights.astype(v.dtype)
        x = jnp.matmul(attn_weights, v)
        x = jnp.transpose(x, (0, 2, 1, 3))
        x = jnp.reshape(x, x.shape[:-2] + (self.config.n_embd,))

        x = nn.Dense(name="c_proj", features=self.config.n_embd)(x)
        return x


class GPT2MLP(nn.Module):
    config: GPT2ModelConfig

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(name="c_fc", features=self.config.n_embd * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(name="c_proj", features=self.config.n_embd)(x)
        return x
