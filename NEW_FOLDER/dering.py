import numpy as np
from scipy.ndimage import median_filter, map_coordinates
from multiprocessing import Pool
from math import sqrt


def get_max_radius(x0, y0, nx, ny):
    radii = []
    rapp = radii.append
    for x in [0, nx]:
        for y in [0, ny]:
            rapp(sqrt((x - x0)**2 + (y - y0)**2))
    return max(radii)


def cart2pol(fxy, x0=None, y0=None, nr=None, nt=None):
    ny, nx = fxy.shape

    if (x0 is None) & (y0 is None):
        x0 = nx // 2
        y0 = ny // 2
    if np.sum([nr is None, nt is None]) == 2:  # both are None
        nr = nt = max(ny, nx)
    elif np.sum([nr is None, nt is None]) == 1:  # one is None
        if nr is None:
            nr = nt
        if nt is None:
            nt = nr

    finalRadius = get_max_radius(x0, y0, nx, ny)

    R, T = np.meshgrid(np.linspace(0, finalRadius, nr, endpoint=False),
                       np.linspace(0, 2 * np.pi, nt, endpoint=False))

    # Adding 3 for border padding
    x = R * np.cos(T) + x0 + 3
    y = R * np.sin(T) + y0 + 3

    fxy = np.pad(fxy, 3, 'edge')  # for constant border
    frt = map_coordinates(fxy, [y, x], mode='constant',
                          cval=0, order=3)
    return frt


def pol2cart(frt, x0, y0, nx, ny):

    nt, nr = frt.shape

    final_radius = get_max_radius(x0, y0, nx, ny)
    scale_radius = nr / final_radius
    scale_angle = nt / (2 * np.pi)

    x, y = np.meshgrid(np.arange(nx), np.arange(ny))

    # Adding 3 for border padding
    r = np.sqrt((x-x0)**2 + (y-y0)**2) * scale_radius + 3
    theta = np.arctan2(y-y0, x-x0)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta) * scale_angle + 3
    frt = np.pad(frt, 3, 'edge')

    fxy = map_coordinates(frt, [theta, r], order=3,
                          mode='constant', cval=0)
    return fxy


def partial_median(im, M_rad, dim, region, p13, p23):

    if region == 1:
        w = int(M_rad / 3)
        buf = (w - 1) // 2 + 1

        im1 = 0
        im2 = p13 + buf
        out1 = 0
        out2 = p13
    if region == 2:
        w = int(2 * M_rad / 3)
        buf = (w - 1) // 2 + 1

        im1 = p13 - buf
        im2 = p23 + buf
        out1 = buf
        out2 = -buf
    if region == 3:
        w = int(3 * M_rad / 3)
        buf = (w - 1) // 2 + 1

        im1 = p23 - buf
        im2 = None
        out1 = buf
        out2 = None

    size = [1, w] if dim == 'rad' else [w, 1]

    return median_filter(im[:, im1:im2], size)[:, out1:out2]


def de_ring(im, M_rad, M_azi, x0=None, y0=None, thresh_0=None,
            nr=None, nt=None, art_thresh=None, parallel=True):

    ny, nx = im.shape

    # 13, 151
    if thresh_0 is not None:  # only threshold if the variable is set
        im = np.clip(im, thresh_0[0], thresh_0[1])
    if x0 is None:
        x0 = nx // 2
    if y0 is None:
        y0 = ny // 2

    frt = cart2pol(im, x0=x0, y0=y0, nr=nr, nt=nt)
    if (nr is None) & (nt is None):
        nt, nr = frt.shape

    p13 = int(nr / 3)
    p23 = int(2 * nr / 3)

    if parallel:
        pool = Pool(processes=3)
        med_r = np.hstack(pool.starmap(
            partial_median, [(frt, M_rad, 'rad', reg, p13, p23)
                             for reg in range(1, 4)],
            chunksize=1))
    else:
        med_r = np.zeros_like(frt)
        med_r[:, 0:p13] = partial_median(frt, M_rad, 'rad', 1, p13, p23)
        med_r[:, p13:p23] = partial_median(frt, M_rad, 'rad', 2, p13, p23)
        med_r[:, p23:nr] = partial_median(frt, M_rad, 'rad', 3, p13, p23)

    art = frt - med_r

    if art_thresh is not None:  # only threshold if variable is set
        art = np.clip(art, art_thresh[0], art_thresh[1])

    if parallel:
        med_ra = np.hstack(pool.starmap(
            partial_median, [(art, M_azi, 'azi', reg, p13, p23)
                             for reg in range(1, 4)],
            chunksize=1))
    else:
        med_ra = np.zeros_like(art)
        med_ra[:, 0:p13] = partial_median(art, M_azi, 'azi', 1, p13, p23)
        med_ra[:, p13:p23] = partial_median(art, M_azi, 'azi', 2, p13, p23)
        med_ra[:, p23:nr] = partial_median(art, M_azi, 'azi', 3, p13, p23)

    med_ra = pol2cart(med_ra, x0=x0, y0=y0, nx=nx, ny=ny)

    return im - med_ra
