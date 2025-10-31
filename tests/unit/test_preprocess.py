import numpy as np
from rice2025.preprocess import normalize, minmax_scale, train_test_split

def test_normalize_zero_mean_unit_var():
    X = np.array([[1,2],[3,4],[5,6]], float)
    Z = normalize(X)
    np.testing.assert_allclose(Z.mean(axis=0), [0,0], atol=1e-7)
    np.testing.assert_allclose(Z.std(axis=0),  [1,1], atol=1e-7)

def test_minmax_scale_in_0_1():
    X = np.array([[2,5],[4,9]], float)
    S = minmax_scale(X)
    assert 0 <= S.min() and S.max() <= 1

def test_train_test_split_sizes():
    X = np.arange(20).reshape(10,2); y = np.arange(10)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    assert len(Xte) == 3 and len(Xtr) == 7 and len(ytr) == 7 and len(yte) == 3
