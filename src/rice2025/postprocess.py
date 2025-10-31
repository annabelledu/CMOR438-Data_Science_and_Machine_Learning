import numpy as np
from typing import Sequence, Any

def majority_label(labels: Sequence[Any]):
    """Most frequent label; ties broken by smallest value for determinism."""
    labels = np.asarray(labels)
    vals, counts = np.unique(labels, return_counts=True)
    winners = vals[counts == counts.max()]
    return winners.min()

def average_label(labels: Sequence[float]) -> float:
    return float(np.mean(labels))
