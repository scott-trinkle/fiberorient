import numpy as np
from skimage import io


def imread(fn):
    return io.imread(fn)


def imsave(fn, im, rgb=False, scalar=None,
           maxperc=None, dtype=None):

    if rgb:
        im = make_rgb(im, scalar, maxperc)
    if dtype is not None:
        im = im.astype(dtype)

    io.imsave(fn, im)


def read_tif_stack(base, start=0, stop=10000):
    im = []
    app = im.append
    for i in range(start, stop+1):
        fn = base + '{:0>4d}'.format(i) + '.tif'
        try:
            temp = imread(fn)
            app(temp)
        except FileNotFoundError:
            if i == start:
                raise ValueError('Choose a higher start index.')
            break
    return np.array(im)


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


def make_rgb(image, scalar=None, maxperc=None):
    '''
    Converts an array to 0-255 8-bit, scaled by AI for export to RGB
    '''

    if scalar is not None:
        rgb = (rescale(image, scale=255) *
               np.expand_dims(scalar, -1)).astype(np.uint8)
    else:
        rgb = rescale(image, scale=255)

    if maxperc is not None:
        return rescale(rgb, scale=255, maxperc=maxperc)
    else:
        return rgb.astype(np.uint8)


def split_comps(vectors):
    '''
    Splits vectors into individual components
    '''
    return vectors[..., 0], vectors[..., 1], vectors[..., 2]


def make_xyz(vectors):
    '''
    Makes xyz index grid for mayavi plots
    '''
    return np.mgrid[0:vectors.shape[0],
                    0:vectors.shape[1],
                    0:vectors.shape[2]]


def colormap(x, y, z):
    colors = np.stack((make_rgb(x.flatten()),
                       make_rgb(y.flatten()),
                       make_rgb(z.flatten()),
                       255 * np.ones(x.size)),  # alpha = 255
                      axis=1).astype(np.uint8)

    # scalar identifier for each point
    s = np.arange(x.flatten().size).reshape(x.shape)

    return colors, s
