import os
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from context import fiberorient as fo

COEFS = {
    'precompute': np.array([2.82094792e-01, -3.15367177e-01,  5.54660495e-03,  3.45978810e-05,
                            5.46217628e-01,  6.81451374e-03,  3.17274844e-01, -1.01891511e-02,
                            -6.35565433e-05, -4.72952994e-01, -5.90047722e-03,  8.98489696e-03,
                            1.68151864e-04,  6.25608696e-01,  1.56123951e-02, -3.17673982e-01,
                            1.47868538e-02,  9.22354869e-05,  4.60341252e-01,  5.74313538e-03,
                            -1.40263401e-02, -2.62502202e-04, -5.04238850e-01, -1.25835466e-02,
                            1.20085266e-02,  3.74642639e-04,  6.82652914e-01,  2.55605778e-02]),
    'delta': np.array([2.82094792e-01, -3.15391565e-01,  4.77568105e-08,  0.00000000e+00,
                       5.46274215e-01,  0.00000000e+00,  3.17356641e-01, -8.77348631e-08,
                       0.00000000e+00, -4.73087348e-01,  0.00000000e+00,  7.73748763e-08,
                       0.00000000e+00,  6.25835735e-01,  0.00000000e+00, -3.17846011e-01,
                       1.27335948e-07,  0.00000000e+00,  4.60602630e-01,  0.00000000e+00,
                       -1.20801487e-07,  0.00000000e+00, -5.04564901e-01,  0.00000000e+00,
                       1.03448213e-07,  0.00000000e+00,  6.83184105e-01,  0.00000000e+00]),
    500: np.array([2.82094792e-01, -3.15387781e-01,  2.18197901e-03, -1.16605370e-04,
                   5.43160769e-01, -5.82194947e-02,  3.17343947e-01, -4.00851398e-03,
                   2.14215745e-04, -4.70377854e-01,  5.04181496e-02,  3.49492689e-03,
                   -5.64612627e-04,  6.11613799e-01, -1.32637332e-01, -3.17819313e-01,
                   5.81776073e-03, -3.10902233e-04,  4.57944488e-01, -4.90854608e-02,
                   -5.45637283e-03,  8.81488252e-04, -4.93077113e-01,  1.06930930e-01,
                   4.56554482e-03, -1.24855815e-03,  6.48422433e-01, -2.15122450e-01])
}


@pytest.mark.parametrize('method', ['delta', 'precompute', 500])
def test_odf_coef(method, vectors):
    odf = fo.odf.ODF(degree=6, method=method).fit(vectors)
    assert_array_almost_equal(COEFS[method], odf.coef)


def test_wrong_method():
    with pytest.raises(ValueError) as e_info:
        odf = fo.odf.ODF(degree=8, method='wrongmethod')
    assert str(e_info.value) == 'Invalid method'


def test_odd_degree():
    with pytest.raises(ValueError) as e_info:
        odf = fo.odf.ODF(degree=5)
    assert str(e_info.value) == 'degree must be even'


def test_to_sphere(vectors):
    sph = fo.util.make_sphere(20)
    odf = fo.odf.ODF(degree=6, method='precompute')
    odf.fit(vectors)
    test_odf = odf.to_sphere(sph)

    true_odf = np.array([0.09798854,  0.15862607, -0.10990833,  0.10908317, -0.21412536,
                         -0.24362026,  0.04500221,  0.19488849,  0.57580163,  0.47802291,
                         0.2509023,  0.17588233, -0.07137362,  0.30557336,  0.08907693,
                         -0.10890658, -0.00474367, -0.11996911,  0.15262168, -0.12269684])
    assert_array_almost_equal(test_odf, true_odf)


def test_sphere_without_coefs_error():
    with pytest.raises(ValueError) as e_info:
        sph = fo.util.make_sphere(20)
        odf = fo.odf.ODF(degree=6, method='precompute')
        odf_on_sphere = odf.to_sphere(sph)
    assert str(e_info.value) == 'Please fit ODF object first'


def test_precompute_sh():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    true_pre = np.load(f'{root}/fiberorient/data/sh_deg20_n6500.npy')
    test_pre = fo.odf._precompute_SH(N=6500, degree=20)

    assert_array_almost_equal(true_pre, test_pre)
