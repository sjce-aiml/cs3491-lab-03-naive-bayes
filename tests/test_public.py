"""Public tests for Lab 03. Hidden tests are harder!"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import lab03 as lab  # noqa: E402

import numpy as np

def test_two_clusters():
    X = np.array([[0, 0], [0.2, 0.1], [5, 5], [5.1, 4.9]])
    y = np.array([0, 0, 1, 1])
    m = lab.NaiveBayes().fit(X, y)
    assert list(m.predict(X)) == [0, 0, 1, 1]
