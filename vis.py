import sys
import numpy as np
from .util import split_comps, make_rgb, make_prc, make_ODF, colormap


def plot_field(vectors, scalars, lw=2.0, sf=0.5, cm='jet', colorbar=False):
    '''
    This creates a mayavi quiver plot of a vector field
    '''
    if [sys.version_info[0], sys.version_info[1]] == [3, 6]:
        raise ValueError('Mayavi not compatible with Python 3.6')

    import mayavi.mlab as m

    p, r, c = make_prc(vectors)
    u, v, w = split_comps(vectors)
    field = m.quiver3d(p, r, c,
                       u, v, w,
                       scalars=scalars,
                       line_width=lw,
                       scale_factor=sf,
                       colormap=cm)

    field.glyph.color_mode = 'color_by_scalar'
    field.glyph.glyph_source.glyph_source.center = [0, 0, 0]
    if colorbar:
        m.colorbar()
    m.orientation_axes()


def save_vtu(fn, vectors, AI):
    if sys.version_info[0] > 2:
        raise ValueError('Pyevtk not compatible with Python 3')
        return

    from pyevtk.hl import pointsToVTK

    p, r, c = make_prc(vectors)
    u, v, w = split_comps(vectors)
    R, G, B = split_comps(make_rgb(vectors, scalar=AI, maxperc=95))

    pointsToVTK(fn,
                c.astype('float'), r.astype('float'), p.astype('float'),
                data={'uvw': (u, v, w), 'FA': AI, 'rgb': (R, G, B)})


def plot_ODF(vectors, n=400, scalar=None, fignum=1, save=False, fn=None):

    from mayavi import mlab
    mlab.close(fignum)

    if scalar is not None:
        print('scalar is not none')
        if scalar.ndim != vectors.ndim:
            print('scalar and vector have different dims')
            data = vectors * np.expand_dims(scalar, -1)
        else:
            print('scalar and vectors do NOT have different dims')
            data = vectors * scalar
    else:
        print('Scalar is none')
        data = vectors

    x, y, z = make_ODF(n, data)
    colors, s = colormap(x, y, z)
    fig = mlab.figure(fignum,
                      bgcolor=(1, 1, 1),
                      fgcolor=(0, 0, 0),
                      size=(1920, 1080))
    ODF = mlab.mesh(x, y, z, scalars=s)
    ODF.module_manager.scalar_lut_manager.lut.table = colors
    mlab.outline()
    mlab.orientation_axes()

    if save:
        mlab.savefig(fn)

    return fig
