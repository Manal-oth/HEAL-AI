"""Seed handling. Seeds are replacement seeds; the article does not disclose its five seeds."""
import os, random
import numpy as np

SEEDS = [42, 123, 2024, 2025, 2026]

def set_seed(seed: int) -> np.random.RandomState:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.RandomState(seed)
