import math
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from config import GPT2ModelConfig
from sample import greedy_sample
from utils import load_pretrained_params, load_hf_tokenizer


class GPT2LMHeadModel(nn.Module):
    config: GPT2ModelConfig
    pretrained_params: dict = None

    @nn.compact
    def __call__(self, input_ids):
        output = GPT2Model(
            name="transformer",
            config=self.config,
            pretrained_params=self.pretrained_params,
        )(input_ids)

        lm_head_weight = get(self.pretrained_params, "lm_head.weight")
        lm_logits = linear(
            name="lm_head",
            features=self.config.vocab_size,
            weight=lm_head_weight.T if lm_head_weight is not None else None,
            use_bias=False,
        )(output)
        return lm_logits


class GPT2Model(nn.Module):
    config: GPT2ModelConfig
    pretrained_params: dict = None

    @nn.compact
    def __call__(self, input_ids):
        input_embeddings = embedding(
            name="wte",
            num_embeddings=self.config.vocab_size,
            features=self.config.n_embd,
            weight=get(self.pretrained_params, "transformer.wte.weight"),
        )(input_ids)
        position_ids = jnp.arange(start=0, stop=input_ids.shape[-1])
        position_embeddings = embedding(
            name="wpe",
            num_embeddings=self.config.context_size,
            features=self.config.n_embd,
            weight=get(self.pretrained_params, "transformer.wpe.weight"),
        )(position_ids)
        x = input_embeddings + position_embeddings

        for block_id in range(self.config.n_layer):
            if self.pretrained_params is None:
                block_params = None
            else:
                block_params = {}
                prefix = f"transformer.h.{block_id}."
                for key in self.pretrained_params:
                    if key.startswith(prefix):
                        block_params[key[len(prefix) :]] = self.pretrained_params[key]
            x = GPT2Block(
                name=f"block_{block_id}",
                config=self.config,
                block_pretrained_params=block_params,
            )(x)

        # final layer normalization
        x = layer_norm(
            name="ln_f",
            scale=get(self.pretrained_params, "transformer.ln_f.weight"),
            bias=get(self.pretrained_params, "transformer.ln_f.bias"),
        )(x)
        return x


class GPT2Block(nn.Module):
    config: GPT2ModelConfig
    block_pretrained_params: dict = None

    @nn.compact
    def __call__(self, x):
        x = x + GPT2MultiHeadAttention(
            name="attn",
            config=self.config,
            block_pretrained_params=self.block_pretrained_params,
        )(
            layer_norm(
                name="ln_1",
                scale=get(self.block_pretrained_params, "ln_1.weight"),
                bias=get(self.block_pretrained_params, "ln_1.bias"),
            )(x)
        )
        x = x + GPT2MLP(
            name="mlp",
            config=self.config,
            block_pretrained_params=self.block_pretrained_params,
        )(
            layer_norm(
                name="ln_2",
                scale=get(self.block_pretrained_params, "ln_2.weight"),
                bias=get(self.block_pretrained_params, "ln_2.bias"),
            )(x)
        )
        return x


class GPT2MultiHeadAttention(nn.Module):
    config: GPT2ModelConfig
    block_pretrained_params: dict = None

    def setup(self):
        self.causal_mask = jnp.tril(
            jnp.ones((self.config.context_size, self.config.context_size))
        )

    @nn.compact
    def __call__(self, x):
        x = linear(
            name="c_attn",
            features=self.config.n_embd * 3,
            use_bias=True,
            weight=get(self.block_pretrained_params, "attn.c_attn.weight"),
            bias=get(self.block_pretrained_params, "attn.c_attn.bias"),
        )(x)

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

        x = linear(
            name="c_proj",
            features=self.config.n_embd,
            use_bias=True,
            weight=get(self.block_pretrained_params, "attn.c_proj.weight"),
            bias=get(self.block_pretrained_params, "attn.c_proj.bias"),
        )(x)

        return x


class GPT2MLP(nn.Module):
    config: GPT2ModelConfig
    block_pretrained_params: dict = None

    @nn.compact
    def __call__(self, x):
        x = linear(
            name="c_fc",
            features=self.config.n_embd * 4,
            use_bias=True,
            weight=get(self.block_pretrained_params, "mlp.c_fc.weight"),
            bias=get(self.block_pretrained_params, "mlp.c_fc.bias"),
        )(x)
        x = gelu_new(x)
        x = linear(
            name="c_proj",
            features=self.config.n_embd,
            use_bias=True,
            weight=get(self.block_pretrained_params, "mlp.c_proj.weight"),
            bias=get(self.block_pretrained_params, "mlp.c_proj.bias"),
        )(x)
        return x


def embedding(
    name=None, num_embeddings=None, features=None, weight=None, dtype="float32"
):
    kwargs = {
        "name": name,
        "num_embeddings": num_embeddings,
        "features": features,
        "dtype": dtype,
    }
    if weight is not None:
        kwargs["embedding_init"] = lambda *_: jnp.array(weight)
    return nn.Embed(**kwargs)


def layer_norm(name=None, scale=None, bias=None, epsilon=1e-5, dtype="float32"):
    kwargs = {
        "name": name,
        "epsilon": epsilon,
        "dtype": dtype,
    }
    if scale is not None:
        kwargs["scale_init"] = lambda *_: jnp.array(scale)
    if bias is not None:
        kwargs["bias_init"] = lambda *_: jnp.array(bias)
    return nn.LayerNorm(**kwargs)


def linear(name=None, features=None, weight=None, bias=None, use_bias=True):
    kwargs = {
        "name": name,
        "features": features,
        "use_bias": use_bias,
    }
    if weight is not None:
        kwargs["kernel_init"] = lambda *_: jnp.array(weight)
    if bias is not None:
        kwargs["bias_init"] = lambda *_: jnp.array(bias)
    return nn.Dense(**kwargs)


def gelu_new(x):
    return (
        0.5
        * x
        * (1.0 + nn.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.power(x, 3.0))))
    )


def get(params: Optional[dict], key):
    if params is None or key not in params:
        return None
    return params[key]


if __name__ == "__main__":
    tokenizer_hf = load_hf_tokenizer()
    input_ids = jnp.array(
        tokenizer_hf.encode(
            "Alan Turing theorized that computers would one day become",
            return_tensors="np",
        )
    )

    lm = GPT2LMHeadModel(
        config=GPT2ModelConfig(),
        pretrained_params=load_pretrained_params(),
    )

    params = lm.init(jax.random.PRNGKey(0), input_ids=input_ids)

    for _ in range(8):
        logits = lm.apply(params, input_ids=input_ids)
        next_id = greedy_sample(logits)
        input_ids = jnp.array([jnp.append(input_ids, next_id).tolist()])
        print(tokenizer_hf.decode(next_id), end="")
    print()
