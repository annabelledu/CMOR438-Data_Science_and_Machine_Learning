from rice2025.postprocess import majority_label, average_label

def test_majority_label_tiebreak_smallest():
    assert majority_label([1,1,2,2]) == 1

def test_average_label():
    assert average_label([1,2,3,4]) == 2.5
