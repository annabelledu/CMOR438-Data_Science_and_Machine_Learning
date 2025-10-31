import numpy as np
from rice2025.knn import knn_predict
from rice2025.metrics import manhattan

def test_knn_classification():
    Xtr = np.array([[0,0],[0,1],[1,0],[1,1]], float)
    ytr = np.array([0,0,1,1])
    Xte = np.array([[0.1,0.2],[0.9,0.8]], float)
    preds = knn_predict(Xtr, ytr, Xte, k=3, classify=True)
    assert list(preds) == [0,1]

def test_knn_regression():
    Xtr = np.array([[0],[1],[2],[3]], float)
    ytr = np.array([10,20,30,40], float)
    Xte = np.array([[1.2]], float)
    preds = knn_predict(Xtr, ytr, Xte, k=2, classify=False, distance=manhattan)
    assert 20 <= preds[0] <= 30
