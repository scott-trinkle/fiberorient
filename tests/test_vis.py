import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from context import fiberorient as fo


def test_img_to_dec(img, vectors):
    true_dec = np.zeros_like(vectors)
    true_dec[..., 0] = fo.util.rescale(img, scale=255).astype(np.uint8)
    test_dec = fo.vis.img_to_dec(img, vectors)
    assert_array_equal(true_dec, test_dec)
