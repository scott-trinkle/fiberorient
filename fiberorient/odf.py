import math
import numpy as np
from scipy.special import sph_harm
from sklearn.neighbors import NearestNeighbors

from .util import make_sphere, cart_to_spherical
import pkg_resources
data_path = pkg_resources.resource_filename('fiberorient', 'data/')


def get_SH_loop_ind(degree):
    '''Get indices for looping the even-n, positive-m SHs

    Parameters
    __________
    degree : int
        Maximum SH degree. Must be even.

    Returns
    _______
    mn : list of tuples
        (m,n) tuples up to n=`degree`
    '''

    mn = [(m, n) for n in range(0, degree + 1, 2)
          for m in range(0, n + 1)]
    return mn


def real_sph_harm(m, n, polar, azim):
    '''Returns spherical harmonic function. Assumes m is positive and
    calculates sph_harm for +m and -m using conjugate symmetry

    Parameters
    __________
    m : int
        SH order
    n : int
        SH degree
    polar : ndarray
        Array of polar angles in [0,pi]
    azim : ndarray
        Array of azimuth angle in [0,2pi]

    Returns
    _______
    real_neg, real_pos : tuple of ndarrays
        Returns -m and +m SH functions if m != 0
    sh.real : ndarray
        Returns sh.real SH function if m==0

    '''
    sh = sph_harm(m, n, azim, polar)
    if m != 0:
        # Implements conjugate symmetry as in Dipy.
        # Note: sqrt(2) factor is implemented later on when we calculate
        # the coefficients
        real_neg = sh.real
        real_pos = sh.imag
        return real_neg, real_pos
    else:
        return sh.real


class ODF:
    '''Class to express an array of vectors as an orientation distribution
    function on a basis of real spherical harmonic (SH) functions. The order of
    the SH coefficients is given by:

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

    Parameters
    __________
    degree : int
        Maximum SH degree. Must be even.
    precompute : bool
        Speed transform by using precomputed SH functions
    delta : bool
        Calculate SH transform using the exact vector points
    n_bins : int
        Calculate SH transform by first binning vectors into angular histogram
    '''

    def __init__(self, degree, method='precompute'):

        self.degree = degree
        if degree % 2 != 0:
            raise ValueError('degree must be even')

        self.n_coef = int(((self.degree * 2 + 3)**2 - 1) / 8)

        if method == 'precompute':
            self.sh_pre = np.load(
                data_path + 'sh_deg20_n6500.npy')[:self.n_coef]
            self.n_bins = 6500
        elif method == 'delta':
            self.n_bins = None
        elif type(method) == int:
            self.n_bins = method
        else:
            raise ValueError('Invalid method')

        self.method = method

        self._mn_sym = get_SH_loop_ind(self.degree)
        self.coef = None

        mn = []
        for n in range(0, degree+1, 2):
            for m in range(-n, n+1):
                mn.append((n, m))
        self.mn = np.array(mn)

    def fit(self, vectors, K=None):
        '''Perform even, real SH transform, compute SH coefficients
        from vectors.

        Parameters
        __________
        vectors : ndarray, shape=(N,3)
            Array of vectors to compute ODF
        K : float or int
            Normalization factor for SH coefficients. Default is N. 

        '''

        if (vectors.ndim != 2) | (vectors.shape[-1] != 3):
            vectors = vectors.reshape((-1, 3))
        if K is None:
            K = vectors.shape[0]

        if self.method == 'delta':
            self._fit_delta(vectors, K)
        elif self.method == 'precompute':
            self._fit_hist_pre(vectors, K)
        else:
            self._fit_hist(vectors, K)
        return self

    def _fit_delta(self, vectors, K):
        '''SH transform treating vectors as sum of delta functions'''
        polar, azim = cart_to_spherical(vectors)
        c = []
        app = c.append
        for m, n in self._mn_sym:
            if m == 0:
                app(real_sph_harm(m, n, polar, azim).sum() / K)
            else:
                neg, pos = real_sph_harm(m, n, polar, azim)
                app(math.sqrt(2) * neg.sum() / K)
                app(math.sqrt(2) * pos.sum() / K)

        self.coef = np.array(c)

    def _fit_hist_pre(self, vectors, K):
        '''SH transform with precomputed SH values'''
        hist = self._vector_to_hist(vectors)
        self.coef = (self.sh_pre * hist[None, :]).sum(axis=1) / K

    def _fit_hist(self, vectors, K):
        '''SH transform with angular binning'''
        hist = self._vector_to_hist(vectors)
        sphere = make_sphere(self.n_bins)
        c = []
        app = c.append
        for m, n in self._mn_sym:
            if m == 0:
                app((hist * real_sph_harm(m, n,
                                          sphere.theta, sphere.phi)).sum() / K)
            else:
                neg, pos = real_sph_harm(m, n, sphere.theta, sphere.phi)
                app(math.sqrt(2) * (hist * neg).sum() / K)
                app(math.sqrt(2) * (hist * pos).sum() / K)
        self.coef = np.array(c)

    def to_sphere(self, sphere):
        '''Calculates ODF as a linear combination of real SH functions
        evaluated on sample points defined by sphere.

        Parameters
        __________
        sphere: dipy Sphere object
            Used to define grid of angular points for ODF

        Returns
        _______
        odf : ndarray
            Value of ODF on sample points defined by sphere

        '''

        if self.coef is None:
            raise ValueError('Please fit ODF object first')

        odf = np.zeros(sphere.phi.size)
        i = 0
        for m, n in self._mn_sym:
            if m == 0:
                odf += self.coef[i] * \
                    real_sph_harm(m, n, sphere.theta, sphere.phi)
                i += 1
            else:
                Y_neg, Y_pos = real_sph_harm(m, n, sphere.theta, sphere.phi)
                odf += self.coef[i] * Y_neg
                i += 1
                odf += self.coef[i] * Y_pos
                i += 1
        return odf

    def _vector_to_hist(self, vectors):
        '''Bins vectors as spherical histogram counts.

        Parameters
        __________
        vectors : ndarray, shape=(N,3)
            Array of vectors

        Returns
        _______
        hist : ndarray
            Counts of vectors within each angular bin

        '''

        sphere = make_sphere(self.n_bins)
        hist_points = np.stack((sphere.x, sphere.y, sphere.z), axis=-1)
        nbrs = NearestNeighbors(n_neighbors=1,
                                algorithm='ball_tree',
                                leaf_size=5).fit(hist_points)
        indices = nbrs.kneighbors(vectors, return_distance=False)
        hist = np.bincount(indices.flatten(), minlength=sphere.theta.size)
        return hist


def _precompute_SH(N=6500, degree=20, path=None):
    '''Utility function to precompute SH functions on grid of 6500 points.'''

    sphere = make_sphere(N)
    mn = get_SH_loop_ind(degree)
    num_coeffs = int(((degree * 2 + 3)**2 - 1) / 8)
    sh = np.zeros((num_coeffs, N))
    count = 0
    for m, n in mn:
        if m == 0:
            sh[count] = real_sph_harm(m, n, sphere.theta, sphere.phi)
            count += 1
        else:
            neg, pos = real_sph_harm(m, n, sphere.theta, sphere.phi)
            sh[count] = math.sqrt(2) * neg
            count += 1
            sh[count] = math.sqrt(2) * pos
            count += 1
    if path is not None:
        np.save(path + 'sh_deg{}_n{}'.format(degree, N), sh)
    else:
        return sh
