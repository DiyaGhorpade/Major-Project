# src/utils.py

import os
import random
import numpy as np
import torch
from config import SEED

def set_seed(seed: int = SEED) -> None:
    """Seed everything for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # makes CUDA ops deterministic (slight speed cost, worth it)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[seed] Everything seeded to {seed}")
