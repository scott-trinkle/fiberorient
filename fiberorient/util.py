import numpy as np
from dipy.core.sphere import Sphere


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

    rescaled = scale * (arr - arr.min()) / (arr.max() - arr.min())
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


def make_sphere(n):
    '''Returns dipy Sphere object with n points determined using Fibonacci
    Sampling with `n` "equally" spaced points on a unit sphere in
    spherical coordinates. http://stackoverflow.com/a/26127012/5854689

    Parameters
    __________
    n : int
        Number of points on the sphere

    Returns
    _______
    sphere : dipy Sphere object
    '''

    z = np.linspace(1 - 1 / n, -1 + 1 / n, num=n)
    polar = np.arccos(z)
    azim = np.mod((np.pi * (3.0 - np.sqrt(5.0))) *
                  np.arange(n), 2 * np.pi) - np.pi
    azim[azim < 0] += 2 * np.pi  # sph_harm functions require azim in [0, 2pi]

    sphere = Sphere(theta=polar, phi=azim)
    return sphere


def cart_to_spherical(vectors):
    '''Takes [...,3] ndarray of vectors and returns flat lists of
    polar and azim values in spherical coordinates.

    Parameters
    __________
    vectors : ndarray, shape=(N,3)
        Array of vectors

    Returns
    _______
    polar : ndarray
        Array of polar angles in [0,pi]
    azim : ndarray
        Array of azimuth angle in [0,2pi]

    '''

    v0, v1, v2 = split_comps(vectors)
    r = np.sqrt(v0**2 + v1**2 + v2**2)
    polar = np.arccos(v2 / r)  # z / r
    azim = np.arctan2(v1, v0)  # y / x
    azim[azim < 0] += 2 * np.pi  # sph_harm functions require azim in [0,2pi]

    return polar, azim


def _prep_vectors(vectors):
    '''Utility function to takes in array of direction coordinates and
    make all x-coordinates positive and sorted by x

    Parameters
    __________
    vectors : ndarray, shape=(...,3)
        Array of vectors

    Returns
    _______
    vectors : ndarray, shape=(...,3)
        Array of vectors, sorted for comparison

    '''

    vectors[vectors[..., 0] < 0] *= -1  # make all x's positive
    if vectors.ndim > 1:
        vectors = vectors[vectors[..., 0].argsort()]  # sort by x
    return vectors
