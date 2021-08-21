import logging

import numpy as np
from scipy.ndimage import gaussian_filter
from concurrent.futures import ProcessPoolExecutor


class StructureTensor(object):
    """Class for calculating the 3D structure tensor at every voxel of an
    image. Contains methods for performing eigenanalysis on the resulting
    tensors, which can be used to estimate the orientation of local fiber-like
    structures.

    Parameters
    __________
    d_sigma : float
        Sigma for the gaussian derivative filters
    n_sigma : float
        Sigma for the gaussian neighborhood filters
    gaussargs : dict
        Keyword arguments for scipy.ndimage.gaussian_filter. Defaults to
        {'mode' : nearest', 'cval' : 0}
    n_jobs : int
        Number of CPU processes for eigenanalysis. Defaults to None, which
        uses all available cores.
    verbose : bool
        Verbosity mode. Default is False

    """

    def __init__(self, d_sigma, n_sigma,
                 gaussargs=None, n_jobs=None, verbose=False):

        self.d_sigma = d_sigma
        self.n_sigma = n_sigma
        if gaussargs is None:
            self.gaussargs = {'mode': 'nearest', 'cval': 0}
        self.n_jobs = n_jobs

        if verbose:
            logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger()

        self.S = None
        self.evals = None
        self.vectors = None
        self.confidence = None

    def fit(self, img):
        """Compute structure tensor array and perform eigenanalysis, extracting the
        local structure orientation vectors along with an orientation confidence
        metric

        Parameters
        __________
        img : ndarray
            3D image array
        """
        self.S = self._structure_tensor(img)
        logging.info('Calculating eigenvectors/values')
        evals = np.zeros(self.S.shape[:4])
        evectors = np.zeros_like(self.S)

        with ProcessPoolExecutor(self.n_jobs) as pool:
            evals, evectors = zip(
                *[eig for eig in pool.map(np.linalg.eigh, self.S)])
        self.evals = np.array(evals)
        self.vectors = np.array(evectors)[..., 0]
        self.confidence = self._get_westin()
        logging.info('Done!')

        return self

    def _structure_tensor(self, img):
        """
        Computes the structure tensor array

        Parameters
        __________
        img : ndarray
            3D image array

        """

        img = np.squeeze(img).astype('float32')

        logging.info('Computing gradient')
        imz, imy, imx = self._compute_gradient(img)

        logging.info('Forming ST elements:')

        logging.info('Szz')
        Szz = gaussian_filter(imz * imz, self.n_sigma, **self.gaussargs)

        logging.info('Szy')
        Szy = gaussian_filter(imz * imy, self.n_sigma, **self.gaussargs)

        logging.info('Szx')
        Szx = gaussian_filter(imz * imx, self.n_sigma, **self.gaussargs)

        logging.info('Syy')
        Syy = gaussian_filter(imy * imy, self.n_sigma, **self.gaussargs)

        logging.info('Syx')
        Syx = gaussian_filter(imy * imx, self.n_sigma, **self.gaussargs)

        logging.info('Sxx')
        Sxx = gaussian_filter(imx * imx, self.n_sigma, **self.gaussargs)

        S = np.array([[Szz, Szy, Szx],
                      [Szy, Syy, Syx],
                      [Szx, Syx, Sxx]])

        # np.linalg.eigh requires shape = (...,3,3)
        S = np.moveaxis(S, [0, 1], [3, 4])

        return S

    def _compute_gradient(self, img):
        """
        Computes the 3D image gradient using convolution with Gaussian
        partial derivatives

        Parameters
        __________
        img : ndarray
            3D image array

        """

        logging.info('Imz')
        imz = gaussian_filter(img, self.d_sigma, order=[1, 0, 0],
                              **self.gaussargs)

        logging.info('Imy')
        imy = gaussian_filter(img, self.d_sigma, order=[0, 1, 0],
                              **self.gaussargs)

        logging.info('Imx')
        imx = gaussian_filter(img, self.d_sigma, order=[0, 0, 1],
                              **self.gaussargs)

        return imz, imy, imx

    def _get_westin(self):
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

        t2 = self.evals[..., 2]  # largest
        t1 = self.evals[..., 1]  # middle
        t0 = self.evals[..., 0]  # smallest

        with np.errstate(invalid='ignore'):
            westin = np.where(t2 != 0, (t1 - t0) / t2, np.zeros_like(t2))
        return westin
