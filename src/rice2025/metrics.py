import numpy as np

def euclidean(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.sum((a - b) ** 2)))

def manhattan(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sum(np.abs(a - b)))
