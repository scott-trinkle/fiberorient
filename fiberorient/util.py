import numpy as np


def get_westin(evals):
    '''
    Calculates westin certainty metric for orientations.

    Parameters
    __________
    evals : ndarray, shape = (...,3)
        Array of eigenvalues of structure tensors

    Returns
    _______
    westin : ndarray
        Array of westin certainty values
    '''

    t2 = evals[..., 2]  # largest
    t1 = evals[..., 1]  # middle
    t0 = evals[..., 0]  # smallest

    with np.errstate(invalid='ignore'):
        westin = np.where(t2 != 0, (t1 - t0) / t2, np.zeros_like(t2))
    return westin


def rescale(arr, scale=1.0):
    '''
    Rescales an array from 0 to `scale`

    Parameters
    __________
    arr : ndarray
        Input array
    scale : float
        Maximum value of rescaled array

    Returns
    _______
    rescaled : ndarray
        Input array scaled to be within 0 to `scale`

    '''

    dtype = arr.dtype
    rescaled = (scale * (arr - arr.min()) /
                (arr.max() - arr.min())).astype(dtype)
    return rescaled


def split_comps(vectors):
    '''
    Utility function to splits vectors into individual components

    Parameters
    __________
    vectors : ndarray
        Input vectors. shape = (...,3)

    Returns
    _______
    v0, v1, v2 : ndarray
        Three component images of `vectors`

    '''

    v0 = vectors[..., 0]
    v1 = vectors[..., 1]
    v2 = vectors[..., 2]
    return v0, v1, v2
