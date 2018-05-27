import math
import numpy as np
from scipy.special import sph_harm
from dipy.core.sphere import Sphere
from dipy.direction.peaks import peak_directions
from .util import split_comps


def make_sphere(n):
    '''
    Fibonacci Sampling:
    Returns "equally" spaced points on a unit sphere in spherical coordinates.
    http://stackoverflow.com/a/26127012/5854689
    '''

    z = np.linspace(1 - 1 / n, -1 + 1 / n, num=n)
    theta = np.arccos(z)
    phi = np.mod((np.pi * (3.0 - np.sqrt(5.0))) *
                 np.arange(n), 2 * np.pi) - np.pi
    phi[phi < 0] += 2 * np.pi  # sph_harm functions require phi in [0, 2pi]

    sphere = Sphere(theta=theta, phi=phi)
    return sphere


def cart_to_spherical(vectors):
    '''
    Takes [...,3] ndarray of vectors and returns flat lists of
    theta and phi values in spherical coordinates.

    Note:
    theta is in [0, pi]
    phi is in [0, 2pi]
    '''
    vz, vy, vx = split_comps(vectors)
    r = np.sqrt(vz**2 + vy**2 + vx**2)
    theta = np.arccos(vz / r)
    phi = np.arctan2(vy, vx)
    phi[phi < 0] += 2 * np.pi  # sph_harm functions require phi in [0,2pi]

    return theta, phi


def check_degree(degree):
    if degree % 2 != 0:
        raise ValueError('SH Degree must be even')


def get_SH_loop_ind(degree):
    '''
    Get indices for looping the even-n, positive-m SHs
    '''
    check_degree(degree)
    mn = [(m, n) for n in range(0, degree + 1, 2)
          for m in range(0, n + 1)]
    return mn


def real_sph_harm(m, n, theta, phi):
    '''
    Assumes m is positive, calculates sph_harm for +m and -m using
    conjugate symmetry
    '''
    sh = sph_harm(m, n, phi, theta)
    if m != 0:
        # Implements conjugate symmetry as in Dipy.
        # Note: it is faster to include sqrt(2) factor when
        # calculating the coefficients.
        real_neg = sh.real
        real_pos = sh.imag
        return real_neg, real_pos
    else:
        return sh.real


def get_SH_coeffs(degree, theta, phi):
    '''
    Calculate even-degree SH coefficients up to 'degree'
    Order of output is given by:

    (n, m)
    ______
    (0, 0)
    (2, 0)
    (2, -1)
    (2, 1)
    (2, -2)
    (2, 2)
    (3, 0)
      .
      .
      .
    '''
    check_degree(degree)
    mn = get_SH_loop_ind(degree)
    c = []
    app = c.append
    K = theta.size
    for m, n in mn:
        if m == 0:
            app(real_sph_harm(m, n, theta, phi).sum() / K)
        else:
            neg, pos = real_sph_harm(m, n, theta, phi)
            app(math.sqrt(2) * neg.sum() / K)
            app(math.sqrt(2) * pos.sum() / K)

    return c


def get_odf(coeffs, sphere):
    '''
    Calculates odf as linear combination of real SH using coeffs,
    evaluated on sample points defined by sphere.
    '''
    degree = int((math.sqrt(8 * len(coeffs) + 1) - 3) // 2)
    mn = get_SH_loop_ind(degree)
    odf = np.zeros(sphere.phi.size)
    i = 0
    for m, n in mn:
        if m == 0:
            odf += coeffs[i] * real_sph_harm(m, n, sphere.theta, sphere.phi)
            i += 1
        else:
            Y_neg, Y_pos = real_sph_harm(m, n, sphere.theta, sphere.phi)
            odf += coeffs[i] * Y_neg
            i += 1
            odf += coeffs[i] * Y_pos
            i += 1
    return odf


def get_peaks(odf, sphere, threshold=0.2, minsep=10):
    dirs, vals, inds = peak_directions(odf, sphere,
                                       relative_peak_threshold=threshold,
                                       min_separation_angle=minsep)
    return dirs, vals, inds


def prep_dirs(dirs):
    # Takes in Nx3 array of direction coordinates
    # makes all x-coordinates positive and sorts by x-coordinate
    dirs[dirs[..., 0] < 0] *= -1  # make all x's positive
    if dirs.ndim > 1:
        dirs = dirs[dirs[..., 0].argsort()]  # sort by x
    return dirs


def get_ang_distance(dirs0, dirs1):
    # Takes two direction coordinate arrays and returns angular distance
    # between each unique direction

    # Sort peaks
    dirs0 = prep_dirs(dirs0)
    dirs1 = prep_dirs(dirs1)

    # Calculate angle in degrees
    ang = np.arccos((dirs0 * dirs1).sum(axis=-1)) * 180 / np.pi

    # Restrict to being between 0-90 degrees
    ang = np.where(ang > 90, 180 - ang, ang)
    return ang
