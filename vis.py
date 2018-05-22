import sys
from .util import (split_comps, make_rgb, make_xyz)


def plot_field(vectors, scalars, lw=2.0, sf=0.5, cm='jet', colorbar=False):
    '''
    This creates a mayavi quiver plot of a vector field
    '''
    if [sys.version_info[0], sys.version_info[1]] == [3, 6]:
        raise ValueError('Mayavi not compatible with Python 3.6')

    import mayavi.mlab as m

    z, y, x = make_xyz(vectors)
    u, v, w = split_comps(vectors)
    field = m.quiver3d(x, y, z,
                       w, v, u,
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

    z, y, x = make_xyz(vectors)
    w, v, u = split_comps(vectors)
    R, G, B = split_comps(make_rgb(vectors, scalar=AI, maxperc=95))

    pointsToVTK(fn,
                x.astype('float'), y.astype('float'), z.astype('float'),
                data={'uvw': (u, v, w), 'FA': AI, 'rgb': (R, G, B)})


def show_ODF(odf, sphere, ren=None, interactive=True, save=False, fn=None, ax_scale=1.5):
    from dipy.viz import actor, window
    if ren is None:
        ren = window.Renderer()
    odf_actor = actor.odf_slicer(odf.reshape((1, 1, 1, -1)), sphere=sphere)
    axes = actor.axes(scale=(ax_scale, ax_scale, ax_scale))
    ren.add(odf_actor, axes)
    ren.set_camera(position=(1, 0.5, 0.5),
                   focal_point=(0, 0, 0),
                   view_up=(0, 0, 1))
    if interactive:
        window.show(ren)
    if save:
        window.record(ren, out_path=fn, size=(1200, 1200))
    return ren


def show_peaks(dirs, vals, sphere, ren=None, interactive=True, save=False, fn=None):
    from dipy.viz import actor, window
    if ren is None:
        ren = window.Renderer()
    peaks_actor = actor.peak_slicer(dirs, vals)
    ax_scale = vals.min() * 0.9
    axes = actor.axes(scale=(ax_scale, ax_scale, ax_scale))
    ren.add(peaks_actor, axes)
    ren.set_camera(position=(1, 0.5, 0.5),
                   focal_point=(0, 0, 0),
                   view_up=(0, 0, 1))
    if interactive:
        window.show(ren)
    if save:
        window.record(ren, out_path=fn, size=(1200, 1200))
    return ren
