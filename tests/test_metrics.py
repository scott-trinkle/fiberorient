import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from context import fiberorient as fo


def test_ang_distance():
    vec1 = np.zeros((5, 3))
    vec1[..., 0] = 1

    vec2 = np.zeros((5, 3))
    vec2[..., 0] = 1 / np.sqrt(2)
    vec2[..., 1] = 1 / np.sqrt(2)

    test_angs = fo.metrics.angular_distance(vec1, vec2)
    true_angs = 45 * np.ones(5)
    assert_array_almost_equal(test_angs, true_angs)


def test_calc_acc():
    np.random.seed(2021)
    c1 = np.random.random(28)
    c2 = np.random.random(28)
    acc = fo.metrics.calc_ACC(c1, c2)
    assert np.round(acc, 6) == 0.808932
