import sys
from .util import split_comps, make_rgb, make_prc


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
    r = make_rgb(u)
    g = make_rgb(v)
    b = make_rgb(w)

    pointsToVTK(fn,
                p.astype('float'), r.astype('float'), c.astype('float'),
                data={'uvw': (u, v, w), 'FA': AI, 'rgb': (r, g, b)})
