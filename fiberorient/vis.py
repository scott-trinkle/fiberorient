import sys
from .util import (split_comps, make_rgb, make_xyz)


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
        window.clear(ren)

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
