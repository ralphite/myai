import jax.numpy as jnp


def greedy_sample(logits):
    """
    logits: (batch_size, seq_len, vocab_size)
    """
    return int(jnp.argmax(logits, axis=-1)[-1][-1])
