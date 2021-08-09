import numpy as np
from fury import actor, window

from .util import rescale


def img_to_dec(img, vectors):
    '''Takes a scalar image and converts it to a "directionally encoded color"
    image: an RGB image with red, blue, and green channels corresponding to the
    three dimensions of the `vectors` image.

    Parameters
    __________
    img : ndarray
        Scalar image. shape=(n0,n1,n2)
    vectors : ndarray
        Image of vectors. shape=(n0,n1,n2,3).

    Returns
    dec : ndarray
        3D RGB image

    '''
    vectors = rescale(vectors)  # check between 0-1
    scalar = rescale(img, scale=255)
    dec = (np.expand_dims(scalar, -1) * vectors).astype(np.uint8)
    return dec


def show_vectors_2D(img, vectors, certainty, max_length=0.25,
                    linewidth=2, min_pix=1000, offset=-0.25,
                    interactive=False, out_path=None):
    '''Visualizes 2D image data with vectors at each pixel.

    Parameters
    __________
    img : ndarray, shape=(n1,n2)
        2D image array
    vectors : ndarray, shape=(n1,n2,3)
        2D vector array
    certainty : ndarray, shape=(n1,n2) or None
        Used to scale vectors. If None: scale all by 1.0
    max_length : float, default=0.25
        Maximum length of vectors
    linewidth : float, default=2
        Width of vectors
    min_pix : int, default=1000
        Minimum display pixels along smallest dimension
    offset : float, default=-0.25
        Offset distance between image and vectors in vtk
    interactive : bool, default=False
        Display interactive scene
    out_path : str, default=None
        Filename to save image

    Returns
    _______
    scene : fury.window.Scene
        Returns scene if interactive is false and no out_path is provided
    '''

    if type(min_pix) != int:
        min_pix = int(min_pix)

    if img.ndim != 2:
        raise ValueError('img must be 2D')
    if vectors.ndim != 3:
        raise ValueError('vectors must be 3D')
    if certainty is None:
        certainty = np.ones_like(img)
    elif certainty.ndim != 2:
        raise ValueError('certainty must be 2D or None')

    certainty /= certainty.max()

    ny, nx = img.shape
    if ny <= nx:
        outy = min_pix
        outx = int(nx/ny * min_pix)
    else:
        outx = min_pix
        outy = int(ny/nx * min_pix)
    size = (outx, outy)

    img = np.expand_dims(img, -1)
    vectors = np.expand_dims(vectors, (2, 3))
    certainty = max_length * np.expand_dims(certainty, (2, 3))

    scene = window.Scene()
    vecs_actor = actor.peak_slicer(vectors, certainty,
                                   colors=None,
                                   linewidth=linewidth)
    aff = np.eye(4)
    aff[2, 3] = offset
    image_actor = actor.slicer(img, interpolation='nearest', affine=aff)
    image_actor.display(None, None, 0)

    scene.add(image_actor)
    scene.add(vecs_actor)

    scene.reset_camera_tight(margin_factor=1.0)

    if interactive:
        window.show(scene, size=size, reset_camera=False)
    elif out_path is not None:
        window.record(scene, size=size, reset_camera=False, out_path=out_path)
    else:
        return scene
