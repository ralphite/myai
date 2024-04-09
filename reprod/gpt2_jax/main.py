import jax
import jax.numpy as jnp
from config import GPT2ModelConfig
from sample import greedy_sample
from utils import load_hf_pretrained_params, load_hf_tokenizer
from model import GPT2LMHeadModel

if __name__ == "__main__":
    tokenizer_hf = load_hf_tokenizer()
    input_ids = jnp.array(
        tokenizer_hf.encode(
            "Alan Turing theorized that computers would one day become",
            return_tensors="np",
        )
    )
    pretrained_params_hf = load_hf_pretrained_params()

    lm = GPT2LMHeadModel(config=GPT2ModelConfig())

    init_params = lm.init(jax.random.PRNGKey(0), input_ids=input_ids)

    for _ in range(8):
        logits = lm.apply(pretrained_params_hf, input_ids=input_ids)
        next_id = greedy_sample(logits)
        input_ids = jnp.array([jnp.append(input_ids, next_id).tolist()])
        print(tokenizer_hf.decode(next_id), end="")
    print()
