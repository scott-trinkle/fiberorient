import numpy as np
from scipy import ndimage as ndi
from skimage import io


def structure_tensor_eig(im, d_sigma=1, n_sigma=1, mode='constant', cval=0):
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


def rescale(a, scale=1.0, dtype=None, maxperc=None):
    '''
    Rescales an array to 0-scale
    '''
    if dtype is None:
        dtype = a.dtype

    shifted = a - a.min()

    if maxperc is not None:
        thresh = np.percentile(shifted, maxperc)
        shifted[shifted > thresh] = thresh

    return (scale * (shifted / shifted.max())).astype(dtype)


def compute_derivatives(image, d_sigma=1.0, mode='constant', cval=0):
    """
    Compute derivatives in row, column and plane directions using convolution
    with Gaussian partial derivatives
    """

    imp = ndi.gaussian_filter(
        image, d_sigma, order=[1, 0, 0], mode=mode, cval=cval)
    imr = ndi.gaussian_filter(
        image, d_sigma, order=[0, 1, 0], mode=mode, cval=cval)
    imc = ndi.gaussian_filter(
        image, d_sigma, order=[0, 0, 1], mode=mode, cval=cval)

    return imp, imr, imc


def make_rgb(im, scalar=None, maxperc=None):
    '''
    Converts an array to 0-255 8-bit, scaled by AI for export to RGB
    '''

    if scalar is not None:
        rgb = (rescale(im, scale=255) *
               np.expand_dims(scalar, -1)).astype(np.uint8)
    else:
        rgb = rescale(im, scale=255)

    if maxperc is not None:
        return rescale(rgb, scale=255, maxperc=maxperc)
    else:
        return rgb.astype(np.uint8)


def imread(fn):
    return io.imread(fn)


def imsave(fn, im, rgb=False, scalar=None,
           maxperc=None, dtype=None):

    if rgb:
        im = make_rgb(im, scalar, maxperc)
    if dtype is not None:
        im = im.astype(dtype)

    io.imsave(fn, im)


def split_comps(vectors):
    '''
    Splits vectors into individual components
    '''
    return vectors[..., 0], vectors[..., 1], vectors[..., 2]


def make_prc(vectors):
    return np.mgrid[0:vectors.shape[0],
                    0:vectors.shape[1],
                    0:vectors.shape[2]]


def fibonacci_sphere(n):
    # Returns "equally" spaced points on a unit sphere in spherical coordinates
    # http://stackoverflow.com/a/26127012/5854689

    n -= 2  # Currently adding two points
    z = np.linspace(1 - 1/n, -1 + 1/n, num=n)
    theta = np.arccos(z)
    phi = np.mod((np.pi*(3.0 - np.sqrt(5.0)))*np.arange(n), 2*np.pi) - np.pi
    theta = np.append(theta, [0, np.pi])
    phi = np.append(phi, [0, np.pi])
    theta.sort()
    phi.sort()
    return theta, phi


def make_sphere(n):
    theta, phi = fibonacci_sphere(n)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)

    return x, y, z, T, P


def make_ODF(n, vectors):
    x, y, z, T, P = make_sphere(n)
    v_z, v_y, v_x = split_comps(vectors)
    r = np.sqrt(v_z ** 2 + v_y ** 2 + v_x ** 2)
    theta = np.arccos(v_z / r)  # might need to add exception handling later
    phi = np.arctan2(v_y, v_x)

    H, t_edges, p_edges = np.histogram2d(theta.flatten(),
                                         phi.flatten(),
                                         bins=(T[:, 0], P[0]))

    # H has shape (n-1, n-1)
    s = np.insert(H, 0, np.zeros_like(H[0]), axis=0)
    s = np.insert(s, 0, np.zeros_like(s[:, 0]), axis=1)

    # Scale by bin count
    x, y, z = [i * s for i in [x, y, z]]

    return x, y, z


def colormap(x, y, z):
    colors = np.stack((make_rgb(x.flatten()),
                       make_rgb(y.flatten()),
                       make_rgb(z.flatten()),
                       255 * np.ones(x.size)),  # alpha = 255
                      axis=1).astype(np.uint8)

    # scalar identifier for each point
    s = np.arange(x.flatten().size).reshape(x.shape)

    return colors, s
