import numpy as np
from scipy.ndimage import median_filter, map_coordinates


def get_max_radius(x0, y0, nx, ny):
    radii = []
    for x in [0, nx]:
        for y in [0, ny]:
            radii.append(np.linalg.norm([x - x0, y - y0]))
    return np.max(radii)


def cart2pol(fxy, x0, y0):
    ny, nx = fxy.shape

    finalRadius = get_max_radius(x0, y0, nx, ny)
    nr = max(nx, ny)
    nt = max(nx, ny)

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


def de_ring(im, M_rad, M_azi, x0=None, y0=None, thresh_0=None,
            nr=None, nt=None, art_thresh=None):

    ny, nx = im.shape

    if thresh_0 is None:
        thresh_0 = [im.min(), im.max()]
    if x0 is None:
        x0 = nx // 2
    if y0 is None:
        y0 = ny // 2
    if np.sum([nr is None, nt is None]) == 2:  # both are None
        nr = nt = max(ny, nx)
    elif np.sum([nr is None, nt is None]) == 1:  # one is None
        if nr is None:
            nr = nt
        if nt is None:
            nt = nr

    im_th = np.clip(im, thresh_0[0], thresh_0[1])

    frt = cart2pol(im_th, x0=x0, y0=y0)

    med_r = np.zeros((nr, nt), dtype=im.dtype)
    a = np.zeros((nr, nt))
    b = np.zeros((nr, nt))
    c = np.zeros((nr, nt))
    for i in range(nt):
        a[i, :] = median_filter(frt[i, :], int(1/3 * M_rad))
        b[i, :] = median_filter(frt[i, :], int(2/3 * M_rad))
        c[i, :] = median_filter(frt[i, :], int(3/3 * M_rad))

    p13 = int(nr/3)
    p23 = int(2*nr/3)

    med_r[:, 0:p13] = a[:, 0:p13]
    med_r[:, p13:p23] = b[:, p13:p23]
    med_r[:, p23:nr] = c[:, p23:nr]

    art = frt - med_r
    if art_thresh is None:
        art_thresh = [art.min(), art.max()]
    art = np.clip(art, art_thresh[0], art_thresh[1])

    med_ra = np.zeros((nr, nt), dtype=im.dtype)
    for i in range(p13):
        med_ra[:, i] = median_filter(art[:, i], int(1/3 * M_azi))
    for i in range(p13, p23):
        med_ra[:, i] = median_filter(art[:, i], int(2/3 * M_azi))
    for i in range(p23, nr):
        med_ra[:, i] = median_filter(art[:, i], int(3/3 * M_azi))

    med_ra = pol2cart(med_ra, x0=x0, y0=y0, nx=nx, ny=ny)

    return im - med_ra
