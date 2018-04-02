import numpy as np
from scipy import ndimage as ndi
import tifffile


def structure_tensor_eig(im, d_sigma, n_sigma, mode='constant', cval=0):
    '''
    Returns the eigenvalues and eigenvectors of the structure tensor
    for an image.
    '''

    Spp, Spr, Spc, Srr, Src, Scc = structure_tensor(
        im, d_sigma=d_sigma, n_sigma=n_sigma, mode=mode, cval=cval)

    S = np.array([[Spp, Spr, Spc],
                  [Spr, Srr, Src],
                  [Spc, Src, Scc]])

    # np.linalg.eigh requires shape = (...,3,3)
    S = np.moveaxis(S, [0, 1], [3, 4])

    evals, evectors = np.linalg.eigh(S)
    return evals, evectors


def structure_tensor(image, d_sigma=1, n_sigma=1, mode='constant', cval=0):
    """
    Computes the structure tensor elements
    """

    imp, imr, imc = compute_derivatives(
        image, d_sigma=d_sigma, mode=mode, cval=cval)

    # structure tensor
    Spp = ndi.gaussian_filter(imp * imp, n_sigma, mode=mode, cval=cval)
    Spr = ndi.gaussian_filter(imp * imr, n_sigma, mode=mode, cval=cval)
    Spc = ndi.gaussian_filter(imp * imc, n_sigma, mode=mode, cval=cval)
    Srr = ndi.gaussian_filter(imr * imr, n_sigma, mode=mode, cval=cval)
    Src = ndi.gaussian_filter(imr * imc, n_sigma, mode=mode, cval=cval)
    Scc = ndi.gaussian_filter(imc * imc, n_sigma, mode=mode, cval=cval)

    return Spp, Spr, Spc, Srr, Src, Scc


def rescale(a, scale=1.0, dtype=None):
    '''
    Rescales an array to 0-scale
    '''
    if not dtype:
        dtype = a.dtype

    return (scale * (a - a.min()) / (a - a.min()).max()).astype(dtype)


def compute_derivatives(image, d_sigma=1.0, mode='constant', cval=0):
    """
    Compute derivatives in row, column and plane directions using convolution
    with Gaussian partial derivatives
    """

    imp = ndi.gaussian_filter(
        image, [d_sigma, 0, 0], order=1, mode=mode, cval=cval)
    imr = ndi.gaussian_filter(
        image, [0, d_sigma, 0], order=1, mode=mode, cval=cval)
    imc = ndi.gaussian_filter(
        image, [0, 0, d_sigma], order=1, mode=mode, cval=cval)

    return imp, imr, imc


def make_rgb(a):
    '''
    Converts an array to 0-255 scaled 8-bit for export to RGB
    '''

    scaled = rescale(vects, scale=255, dtype=np.int8)
    return scaled


def save_tiff(fn, im, rgb=False):
    '''
    Saves `im` as a .tiff file, with option for RGB format for principal
    orientation vectors
    '''
    if rgb:
        tifffile.imsave(fn, make_rgb(im))
    else:
        tifffile.imsave(fn, im)


def split_comps(vectors):
    '''
    Splits vectors into individual components
    '''
    return vectors[..., 0], vectors[..., 1], vectors[..., 2]


def make_prc(vectors):
    plc, row, col = vectors.shape[:3]
    p, r, c = np.mgrid[0:plc, 0:row, 0:col]
    return p, r, c
