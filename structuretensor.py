import numpy as np
from .util import structure_tensor_eig


class StructureTensor(object):
    def __init__(self, im, d_sigma=1.0, n_sigma=1.0,
                 gaussmode='nearest', cval=0):
        self.evals, self.evectors = structure_tensor_eig(im=im,
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
