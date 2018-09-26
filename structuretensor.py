import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
try:
    import cupy as cp
    from .cuda_kernels import cu_eigh
except:
    pass


class StructureTensor(object):
    def __init__(self, im, d_sigma=15.0 / 1.2, n_sigma=13 / 1.2,
                 gaussmode='nearest', cval=0, cuda=False, par_cpu=True, n=None):

        self.evals, self.orientations = structure_tensor_eig(image=im,
                                                             d_sigma=d_sigma,
                                                             n_sigma=n_sigma,
                                                             mode=gaussmode,
                                                             cval=cval,
                                                             cuda=cuda,
                                                             par_cpu=par_cpu,
                                                             n=n)

    def get_anisotropy_index(self, metric='fa'):
        '''
        Calculates fractional anisotropy image from the eigenvalues image.
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

    def results(self, metric='fa'):
        '''
        Quick method to return anisotropy index and orientation vectors
        '''
        return self.get_anisotropy_index(metric=metric), self.orientations


def structure_tensor_eig(image, d_sigma=15 / 1.2, n_sigma=13 / 1.2,
                         mode='nearest', cval=0, cuda=False, par_cpu=False,
                         n=None):
    '''
    Returns the eigenvalues and eigenvectors of the structure tensor
    for an image.
    '''

    Szz, Szy, Szx, Syy, Syx, Sxx = structure_tensor_elements(
        image, d_sigma=d_sigma, n_sigma=n_sigma, mode=mode, cval=cval)

    if cuda:
        # Copying ST elements to GPU
        a = cp.array(Szz)
        b = cp.array(Szy)
        c = cp.array(Szx)
        d = cp.array(Syy)
        e = cp.array(Syx)
        f = cp.array(Sxx)

        # Initialize individual eig elements
        eig1 = cp.zeros_like(Sxx)
        eig2 = cp.zeros_like(Sxx)
        eig3 = cp.zeros_like(Sxx)

        # Kernel only calculates smallest eigenvector
        vec1 = cp.zeros_like(Sxx)
        vec2 = cp.zeros_like(Sxx)
        vec3 = cp.zeros_like(Sxx)

        cu_eigh(a, b, c, d, e, f, eig1, eig2, eig3, vec1, vec2, vec3)

        evals = np.stack((eig3.get(), eig2.get(), eig1.get()),
                         axis=-1)
        evecs = np.stack((vec1.get(), vec2.get(), vec3.get()),
                         axis=-1)

        return evals, evecs
    else:

        S = np.array([[Szz, Szy, Szx],
                      [Szy, Syy, Syx],
                      [Szx, Syx, Sxx]])

        # np.linalg.eigh requires shape = (...,3,3)
        S = np.moveaxis(S, [0, 1], [3, 4])

        del Szz, Szy, Szx, Syy, Syx, Sxx

        if par_cpu:
            from concurrent.futures import ProcessPoolExecutor
            evals = np.zeros(S.shape[:4])
            evectors = np.zeros(S.shape)

            with ProcessPoolExecutor(n) as pool:
                evals, evectors = zip(
                    *[eig for eig in pool.map(np.linalg.eigh, S)])
                return np.array(evals), np.array(evectors)[..., 0]
        else:
            evals, evectors = np.linalg.eigh(S)
            return evals, evectors[..., 0]


def structure_tensor_elements(image, d_sigma=15 / 1.2, n_sigma=13 / 1.2,
                              mode='nearest', cval=0):
    """
    Computes the structure tensor elements
    """

    image = np.squeeze(image)
    # prevents overflow for uint8 xray data
    image = img_as_float(image).astype('float32')

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


def compute_derivatives(image, d_sigma=15 / 1.2, mode='nearest', cval=0):
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
