import numpy as np
from .util import _prep_vectors


def angular_distance(vectors0, vectors1):
    '''Takes two vector arrays and returns the angular distance
    between each direction

    Parameters
    __________
    vectors0, vectors 1 : ndarray, shape=(N,3)
        Array of vectors to compare

    Returns
    _______
    ang : ndarray in [0,90]
        Array of angular distances between each element of vectors0 and
        vectors1

    '''

    # Sort peaks
    vectors0 = _prep_vectors(vectors0)
    vectors1 = _prep_vectors(vectors1)

    # Calculate angle in degrees
    ang = np.arccos((vectors0 * vectors1).sum(axis=-1)) * 180 / np.pi

    # Restrict to being between 0-90 degrees
    ang = np.where(ang > 90, 180 - ang, ang)
    return ang


def calc_ACC(c1, c2):
    '''Calcualtes the angular correlation coefficient between two ODFs
    expressed as SH coefficients.

    Parameters
    __________
    c1, c2 : ndarray
        ODF SH coefficients

    Returns
    _______
    ACC : float
        Angular correlation coefficient
    '''

    c1_norm = np.sqrt((abs(c1)**2).sum())
    c2_norm = np.sqrt((abs(c2)**2).sum())

    ACC = (c1 * c2).sum() / (c1_norm * c2_norm)
    return ACC
