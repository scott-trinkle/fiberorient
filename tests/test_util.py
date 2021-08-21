import numpy as np
from numpy.testing import assert_array_equal

from context import fiberorient as fo


def test_rescale():
    a = np.array([1., 2., 3.])
    a_sc = fo.util.rescale(a)
    ans = [0., 0.5, 1.]
    assert_array_equal(a_sc, ans)
