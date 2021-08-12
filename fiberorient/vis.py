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
    vectors = abs(vectors)  # +- symmetry
    vectors /= vectors.max()
    scalar = rescale(img, scale=255)
    dec = (np.expand_dims(scalar, -1) * vectors).astype(np.uint8)
    return dec


def show_vectors_2D(img, vectors, scale=None, max_length=0.25,
                    linewidth=2, min_pix=1000, offset=-0.25,
                    interactive=False, out_path=None):
    '''Visualizes 2D image data with vectors at each pixel.

    Parameters
    __________
    img : ndarray, shape=(n1,n2)
        2D image array
    vectors : ndarray, shape=(n1,n2,3)
        2D vector array
    scale : ndarray, shape=(n1,n2) or None
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
    if scale is None:
        scale = np.ones_like(img)
    elif scale.ndim != 2:
        raise ValueError('Scale must be 2D or None')

    scale /= scale.max()

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
    scale = max_length * np.expand_dims(scale, (2, 3))

    scene = window.Scene()
    scene.SetBackground(1, 1, 1)
    vecs_actor = actor.peak_slicer(vectors, scale,
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


def show_odf(odf_array, sphere, slicer_args=None, min_pix=1000,
             interactive=False, out_path=None):
    '''Visualizes 2D image data with ODFs.

    Parameters
    __________
    odf_array : ndarray, shape=(n_points_on_sphere)
        ODF expressed on spherical points
    sphere : dipy Sphere
        Sphere used to construct ODFs
    slicer_args : dict
        Additional arguments to actor.odf_slicer
    min_pix : int, default=1000
        Minimum display pixels along smallest dimension
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

    if odf_array.ndim != 1:
        raise ValueError('odf_array must have shape=(n_points)')

    if slicer_args is None:
        slicer_args = {
            'norm': False,
            'colormap': None,
            'scale': 0.5
        }

    size = (min_pix, min_pix)

    odf_array = np.expand_dims(odf_array, (0, 1, 2))

    scene = window.Scene()
    scene.SetBackground(1, 1, 1)

    odf_actor = actor.odf_slicer(odf_array, sphere=sphere, **slicer_args)

    scene.add(odf_actor)
    scene.reset_camera_tight(margin_factor=1.0)

    if interactive:
        window.show(scene, size=size, reset_camera=False)
    elif out_path is not None:
        window.record(scene, size=size, reset_camera=False, out_path=out_path)
    return scene


def show_odf_img_2D(odf_array, sphere, img=None, min_pix=1000,
                    slicer_args=None, offset=0.25, interactive=False,
                    out_path=None):
    '''Visualizes 2D image data with ODFs.

    Parameters
    __________
    odf_array : ndarray, shape=(nx,ny,n_points_on_sphere)
        2D grid of ODFs
    sphere : dipy Sphere
        Sphere used to construct ODFs
    img : ndarray or None
        2D image to plot beneath ODFs
    min_pix : int, default=1000
        Minimum display pixels along smallest dimension
    slicer_args : dict
        Additional arguments to actor.odf_slicer
    offset : float, default=-0.25
        Offset distance between image and ODFs in vtk
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

    if odf_array.ndim != 3:
        raise ValueError('odf_array must have shape=(nx,ny,n_points0')

    if slicer_args is None:
        slicer_args = {
            'norm': False,
            'colormap': None,
            'scale': 0.15
        }

    scene = window.Scene()
    scene.SetBackground(1, 1, 1)

    odf_nx, odf_ny = odf_array.shape[:2]

    if img is not None:
        if img.ndim != 2:
            raise ValueError('img must be 2D')
        nx, ny = img.shape
        img = np.expand_dims(img, -1)
        aff = np.eye(4)
        aff[0, 0] = odf_nx / nx
        aff[1, 1] = odf_ny / ny
        aff[0, 3] = -0.5
        aff[1, 3] = -0.5
        aff[2, 3] = offset
        image_actor = actor.slicer(img, interpolation='nearest', affine=aff)
        image_actor.display(None, None, 0)
        scene.add(image_actor)
    else:
        nx, ny = odf_nx, odf_ny

    if ny <= nx:
        outy = min_pix
        outx = int(nx/ny * min_pix)
    else:
        outx = min_pix
        outy = int(ny/nx * min_pix)
    size = (outx, outy)

    odf_array = np.expand_dims(odf_array, 2)
    odf_actor = actor.odf_slicer(odf_array, sphere=sphere, **slicer_args)

    scene.add(odf_actor)

    x0 = (odf_nx-1)/2
    y0 = (odf_ny-1)/2

    scene.set_camera(focal_point=[x0, y0, 0],
                     view_up=[0, -1, 0])
    angle = np.pi * scene.camera().GetViewAngle() / 180
    dist = max(odf_nx, odf_ny) / np.sin(angle/2.) / 3
    scene.set_camera(position=[x0, y0, -dist],
                     focal_point=[x0, y0, 0],
                     view_up=[0, -1, 0])

    if interactive:
        window.show(scene, size=size, reset_camera=False)
    elif out_path is not None:
        window.record(scene, size=size, reset_camera=False, out_path=out_path)
    return scene
