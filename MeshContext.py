import Art
import ArtCollection
import numpy as np


def angle(segment1, segment2):
    v1 = segment1[1] - segment1[0]
    v2 = segment2[1] - segment2[0]

    if np.linalg.norm(v2) < 1.e-4 or np.linalg.norm(v1) < 1.e-4:
        return 0
    return np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))


def get_closest_pixel(point, scale, offset):
    assert (scale > 0)
    return np.floor((point[0] - offset[0])/scale), np.floor((point[1] - offset[1])/scale)


def l_inf_norm(v):
    return np.max(np.abs(v))


def initialize(vx, vy, art, step, scale, offset):
    for bezier in art.get_beziers():
        for t in np.arange(0, 1 + step, step):
            ij = get_closest_pixel(point=bezier.get_point(t), scale=scale, offset=offset)
            g = bezier.get_gradient(t)
            vx[ij] = g[0]
            vy[ij] = g[1]


def get_adjacent(ij, adj_len, shape):
    i,j = ij
    max_x = shape[0]
    max_y = shape[1]
    top_left = i - adj_len, j - adj_len
    bottom_right = i + adj_len, j + adj_len
    ret = []
    for k in range(max(0, top_left[0]), min(bottom_right[0], max_x)):
        for l in range(max(0, top_left[1]), min(bottom_right[1], max_y)):
            ret.append((k,l))
    return ret


def compute_curl(vx, vy, iter_lim, adj_len):
    iter = 0
    assert (vx.shape == vy.shape)
    shape = vx.shape
    vx_ = np.zeros(shape=shape)
    vy_ = np.zeros(shape=shape)
    while iter < iter_lim and (l_inf_norm(vx)):
        iter = iter + 1
        for ij in np.ndindex(shape[:2]):
            vx[ij] = np.average([vx[kl] for kl in get_adjacent(ij, adj_len=adj_len, shape=shape)])
            vy[ij] = np.average([vy[kl] for kl in get_adjacent(ij, adj_len=adj_len, shape=shape)])


def build_curl(art, step, scale, conv_size, iter_lim):
    bbox = art.get_bounding_box()
    offset = bbox[0]
    length = bbox[1] - bbox[0]
    shape = (length[0]/scale, length[1]/scale, 1)
    vx, vy = np.zeros(shape=shape), np.zeros(shape=shape)

    initialize(vx=vx, vy=vy, art=art, step=step, scale=scale, offset=offset)
    compute_curl(vx=vx, vy=vy, iter_lim=iter_lim, adj_len=conv_size)
    return vx, vy


