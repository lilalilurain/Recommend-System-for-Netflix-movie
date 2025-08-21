from pathlib import Path
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
