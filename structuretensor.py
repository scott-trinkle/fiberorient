import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import img_as_float


class StruutilctureTensor(object):
    def __init__(self, im, d_sigma=15.0 / 2.4, n_sigma=13 / 2.4,
                 gaussmode='nearest', cval=0):
        self.evals, self.evectors = structure_tensor_eig(image=im,
                                                         d_sigma=d_sigma,
                                                         n_sigma=n_sigma,
                                                         mode=gaussmode,
                                                         cval=cval)

    def get_orientations(self):
        '''
        Returns principal orientation vector from a 3D volume
        using the structure tensor.
        '''

        # taking vector w/ smallest eval
        self.orientations = self.evectors[..., 0]
        return self.orientations

    def get_anisotropy_index(self, metric='fa'):
        '''
        Calculates fractional anisotropy image from the eigenvalues image.
        Eigenvalues are ordered from smallest to largest, so t1 > t2 > t3.
        '''

        t1 = self.evals[..., 2]  # largest
        t2 = self.evals[..., 1]  # middle
        t3 = self.evals[..., 0]  # smallest

        with np.errstate(invalid='ignore'):
            if metric == 'westin':
                self.AI = np.where(t1 != 0, (t2 - t3) / t1, np.zeros_like(t1))
            elif metric == 'fa':
                norm2 = t1**2 + t2**2 + t3**2
                self.AI = np.where(norm2 != 0,
                                   np.sqrt(((t1 - t2)**2 +
                                            (t2 - t3)**2 +
                                            (t1 - t3)**2) / (2 * norm2)),
                                   np.zeros_like(t1))

        return self.AI

    def results(self, metric='fa'):
        '''
        Quick method to return anisotropy index and orientation vectors
        '''
        return self.get_anisotropy_index(metric=metric), self.get_orientations()


def structure_tensor_eig(image, d_sigma=7.5 / 1.2, n_sigma=6.5 / 1.2,
                         mode='nearest', cval=0):
    '''
    Returns the eigenvalues and eigenvectors of the structure tensor
    for an image.
    '''

    Szz, Szy, Szx, Syy, Syx, Sxx = structure_tensor_elements(
        image, d_sigma=d_sigma, n_sigma=n_sigma, mode=mode, cval=cval)

    S = np.array([[Szz, Szy, Szx],
                  [Szy, Syy, Syx],
                  [Szx, Syx, Sxx]])

    # np.linalg.eigh requires shape = (...,3,3)
    S = np.moveaxis(S, [0, 1], [3, 4])

    evals, evectors = np.linalg.eigh(S)
    return evals, evectors


def structure_tensor_elements(image, d_sigma=7.5 / 1.2, n_sigma=6.5 / 1.2,
                              mode='nearest', cval=0):
    """
    Computes the structure tensor elements
    """

    image = np.squeeze(image)
    image = img_as_float(image)  # prevents overflow for uint8 xray data

    imz, imy, imx = compute_derivatives(
        image, d_sigma=d_sigma, mode=mode, cval=cval)

    # structure tensor
    Szz = gaussian_filter(imz * imz, n_sigma, mode=mode, cval=cval)
    Szy = gaussian_filter(imz * imy, n_sigma, mode=mode, cval=cval)
    Szx = gaussian_filter(imz * imx, n_sigma, mode=mode, cval=cval)
    Syy = gaussian_filter(imy * imy, n_sigma, mode=mode, cval=cval)
    Syx = gaussian_filter(imy * imx, n_sigma, mode=mode, cval=cval)
    Sxx = gaussian_filter(imx * imx, n_sigma, mode=mode, cval=cval)

    return Szz, Szy, Szx, Syy, Syx, Sxx


def compute_derivatives(image, d_sigma=7.5 / 1.2, mode='nearest', cval=0):
    """
    Compute derivatives in row, column and plane directions using convolution
    with Gaussian partial derivatives
    """

    imz = gaussian_filter(
        image, d_sigma, order=[1, 0, 0], mode=mode, cval=cval)
    imy = gaussian_filter(
        image, d_sigma, order=[0, 1, 0], mode=mode, cval=cval)
    imx = gaussian_filter(
        image, d_sigma, order=[0, 0, 1], mode=mode, cval=cval)

    return imz, imy, imx
