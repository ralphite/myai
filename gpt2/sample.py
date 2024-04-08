import numpy as np


def greedy_sample(logits):
    return np.argmax(logits[-1])
