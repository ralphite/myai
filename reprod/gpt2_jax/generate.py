from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import jax.random as random
from config import GPT2_SM_MODEL_CONFIG, RNG_KEY
from model import GPT2LMHeadModel
from utils import load_hf_pretrained_params, load_hf_tokenizer


class BaseDecoder(ABC):
    @abstractmethod
    def __call__(self, logits: jax.Array):
        pass


class GreedySearch(BaseDecoder):
    def __call__(self, logits: jax.Array):
        assert logits.ndim >= 2
        return jnp.argmax(logits[..., -1, :], axis=-1)


class Sample(BaseDecoder):
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        num_samples: int = 1,
        rng_key: jax.random.PRNGKey = RNG_KEY,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.num_samples = num_samples
        self.rng_key = rng_key

    def __call__(self, logits: jax.Array):
        assert logits.ndim >= 2
        assert self.num_samples >= 1

        if self.temperature == 0:
            return GreedySearch()(logits)

        x = logits[..., -1, :]  # (B, C)

        # Apply temperature scaling
        x = x / self.temperature

        # Apply top-k filtering
        if self.top_k > 0:
            top_k = min(self.top_k, logits.shape[-1])
            _, top_k_indices = jax.lax.top_k(x, k=top_k)
            top_k_indices = top_k_indices.astype(jnp.uint16)
            one_hot_top_k = jax.nn.one_hot(
                top_k_indices, axis=-1, num_classes=logits.shape[-1]
            )
            top_k_mask = jnp.sum(one_hot_top_k, axis=-2)
            x = jnp.where(top_k_mask, x, -jnp.inf)

        # # Apply top-p filtering
        if self.top_p < 1.0:
            raise NotImplementedError()

        # Sample from the filtered logits
        sample = random.categorical(self.rng_key, x, shape=(self.num_samples,), axis=-1)
        return sample


class BeamSearch(BaseDecoder):
    def __call__(self, logits: jax.Array):
        raise NotImplementedError()


class Generator:
    def __init__(self, decoder: BaseDecoder = GreedySearch()) -> None:
        self._tokenizer = load_hf_tokenizer()
        self._lm = GPT2LMHeadModel(config=GPT2_SM_MODEL_CONFIG)
        self._pretrained_params = load_hf_pretrained_params()
        self._decoder = decoder

    def __call__(
        self,
        prompt: str,
        max_length: int = 128,
    ) -> str:
        input_ids = jnp.array(self._tokenizer.encode(prompt, return_tensors="np"))
        self._lm.init(RNG_KEY, input_ids=input_ids)
        while input_ids.shape[-1] < max_length:
            logits = self._lm.apply(self._pretrained_params, input_ids=input_ids)
            next_id = self._decoder(logits)
            input_ids = jnp.array([jnp.append(input_ids, next_id).tolist()])
        return self._tokenizer.decode(input_ids.squeeze().tolist())


if __name__ == "__main__":
    greedy_gen = Generator()
    sample_gen = Generator(decoder=Sample(temperature=1.5, top_k=8))

    for gen in [greedy_gen, sample_gen]:
        output = gen(
            "Alan Turing theorized that computers would one day become",
            max_length=32,
        )
        print(output)
