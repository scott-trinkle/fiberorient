import os
import pytest
import numpy as np

from context import fiberorient as fo


@pytest.fixture(scope='package')
def img():

    nx, ny, nz = (4, 4, 4)
    P = 1

    x, y, z = np.meshgrid(np.linspace(-nx, nx, nx),
                          np.linspace(-ny, ny, ny),
                          np.linspace(-nz, nz, nz),
                          indexing='ij')

    k = [1, 0, 0]
    phantom = np.cos(2 * np.pi / P * y) + np.cos(2 * np.pi / P * z)

    return phantom


@pytest.fixture(scope='package')
def st_obj(img):
    st_obj = fo.StructureTensor(d_sigma=1, n_sigma=1, verbose=True)
    st_obj.fit(img)
    return st_obj


@pytest.fixture(scope='package')
def vectors(st_obj):
    return st_obj.vectors
