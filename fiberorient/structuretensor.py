import logging

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from concurrent.futures import ProcessPoolExecutor

img = np.random.random((25, 25, 25))


class StructureTensor(object):

    def __init__(self, im, d_sigma=7.5 / 1.2, n_sigma=6.5 / 1.2,
                 gaussmode='nearest', cval=0, n=None, verbose=False):
        if verbose:
            logging.basicConfig(level=logging.INFO, format='%(message)s')

        self.evals, self.orientations = structure_tensor_eig(image=im,
                                                             d_sigma=d_sigma,
                                                             n_sigma=n_sigma,
                                                             mode=gaussmode,
                                                             cval=cval,
                                                             n=n,
                                                             verbose=verbose)

    def get_anisotropy_index(self, metric='westin'):
        '''
        Calculates westin image from the eigenvalues image.
        Eigenvalues are ordered from smallest to largest, so t1 > t2 > t3.
        '''

        # t1 = self.evals[..., 2]  # largest
        # t2 = self.evals[..., 1]  # middle
        # t3 = self.evals[..., 0]  # smallest

        with np.errstate(invalid='ignore'):
            if metric == 'westin':
                AI = np.where(self.evals[..., 2] != 0, (self.evals[..., 1] -
                                                        self.evals[..., 0]) / self.evals[..., 2], np.zeros_like(self.evals[..., 2]))
            elif metric == 'fa':
                norm2 = self.evals[..., 2]**2 + \
                    self.evals[..., 1]**2 + self.evals[..., 0]**2
                AI = np.where(norm2 != 0,
                              np.sqrt(((self.evals[..., 2] - self.evals[..., 1])**2 +
                                       (self.evals[..., 1] - self.evals[..., 0])**2 +
                                       (self.evals[..., 2] - self.evals[..., 0])**2) / (2 * norm2)),
                              np.zeros_like(self.evals[..., 2]))

        return AI

    def results(self, metric='westin'):
        '''
        Quick method to return anisotropy index and orientation vectors
        '''
        if self.verbose:
            logging.info('Calculating FA')
        return self.get_anisotropy_index(metric=metric), self.orientations


def structure_tensor_eig(image, d_sigma=15 / 1.2, n_sigma=13 / 1.2,
                         mode='nearest', cval=0, n=None, verbose=False):
    '''
    Returns the eigenvalues and eigenvectors of the structure tensor
    for an image.
    '''

    if verbose:
        logging.info('Calculating ST elements')
    Szz, Szy, Szx, Syy, Syx, Sxx = structure_tensor_elements(
        image, d_sigma=d_sigma, n_sigma=n_sigma, mode=mode, cval=cval,
        verbose=verbose)

    S = np.array([[Szz, Szy, Szx],
                  [Szy, Syy, Syx],
                  [Szx, Syx, Sxx]])

    # np.linalg.eigh requires shape = (...,3,3)
    S = np.moveaxis(S, [0, 1], [3, 4])

    if verbose:
        logging.info('Calculating eigenvectors/values')

    evals = np.zeros(S.shape[:4])
    evectors = np.zeros(S.shape)

    with ProcessPoolExecutor(n) as pool:
        evals, evectors = zip(
            *[eig for eig in pool.map(np.linalg.eigh, S)])
    return np.array(evals), np.array(evectors)[..., 0]


def structure_tensor_elements(image, d_sigma=15 / 1.2, n_sigma=13 / 1.2,
                              mode='nearest', cval=0, verbose=False):
    """
    Computes the structure tensor elements
    """

    image = np.squeeze(image)
    # prevents overflow for uint8 xray data
    image = img_as_float(image).astype('float32')

    if verbose:
        logging.info('Computing gradient')
    imz, imy, imx = compute_derivatives(
        image, d_sigma=d_sigma, mode=mode, cval=cval, verbose=verbose)

    if verbose:
        logging.info('Forming ST elements:')
    # structure tensor
    if verbose:
        logging.info('Szz')
    Szz = gaussian_filter(imz * imz, n_sigma, mode=mode, cval=cval)
    if verbose:
        logging.info('Szy')
    Szy = gaussian_filter(imz * imy, n_sigma, mode=mode, cval=cval)
    if verbose:
        logging.info('Szx')
    Szx = gaussian_filter(imz * imx, n_sigma, mode=mode, cval=cval)
    if verbose:
        logging.info('Syy')
    Syy = gaussian_filter(imy * imy, n_sigma, mode=mode, cval=cval)
    if verbose:
        logging.info('Syx')
    Syx = gaussian_filter(imy * imx, n_sigma, mode=mode, cval=cval)
    if verbose:
        logging.info('Sxx')
    Sxx = gaussian_filter(imx * imx, n_sigma, mode=mode, cval=cval)

    return Szz, Szy, Szx, Syy, Syx, Sxx


def compute_derivatives(image, d_sigma=15 / 1.2, mode='nearest', cval=0,
                        verbose=False):
    """
    Compute derivatives in row, column and plane directions using convolution
    with Gaussian partial derivatives
    """

    if verbose:
        logging.info('Imz')
    imz = gaussian_filter(
        image, d_sigma, order=[1, 0, 0], mode=mode, cval=cval)
    if verbose:
        logging.info('Imy')
    imy = gaussian_filter(
        image, d_sigma, order=[0, 1, 0], mode=mode, cval=cval)
    if verbose:
        logging.info('Imx')
    imx = gaussian_filter(
        image, d_sigma, order=[0, 0, 1], mode=mode, cval=cval)

    return imz, imy, imx
