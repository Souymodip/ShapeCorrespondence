import numpy as np
import Art

def get_line_list(polygon):
    Ls = []
    l = len(polygon.points)
    if l > 0:
        p = polygon.points[0]
        for i in range(1, l):
            Ls.append([p, polygon.points[i]])
            p = polygon.points[i]
    if polygon.isClosed:
        Ls.append([polygon.points[l - 1], polygon.points[0]])
    return Ls


def get_raster_map(qr, scale):
    return np.zeros((int((qr[1][0] - qr[0][0])*scale), int((qr[1][1] - qr[0][1])*scale), 1)), (qr[0][0], qr[0][1])


def line_integ_HD(line, qx, qy):
    ax, ay = line[0]
    bx, by = line[1]

    if by == ay :
        return 0

    if ax == bx:
        if 0 <= (qy - ay)/(by - ay) <= 1 and qx <= ax:
            return np.sign(by - ay)
        else:
            return 0
    else:
        t = (qy - ay) / (by - ay)
        if 0 <= t <= 1 and  ax + (bx - ax)* t >= qx:
            return np.sign(by - ay)
        else:
            return 0


def is_y_tangent(Ls, qy):
    for l in Ls:
        ay, by = l[0][1], l[1][1]
        if qy == ay == by:
            return 0
    else:
        return 1


def area_polygon(polygon, qr, scale):
    Ls = get_line_list(polygon)
    anp, offset = get_raster_map(qr, scale)

    a = 0
    for ij in np.ndindex(anp.shape[:2]):
        qx, qy = ij[0] / scale + offset[0], ij[1] / scale + offset[1]
        K1 = np.array([line_integ_HD(l, qx, qy) for l in Ls])

        v = np.abs(sum(K1)) * is_y_tangent(Ls, qy)
        anp[ij] = v
        a = a + v

    return a/scale, anp, offset

def overlap_polygon(polygon1, polygon2, qr, scale):
    Ls1 = get_line_list(polygon1)
    Ls2 = get_line_list(polygon2)
    anp, offset = get_raster_map(qr, scale)

    a = 0
    for ij in np.ndindex(anp.shape[:2]):
        qx, qy = ij[0] / scale + offset[0], ij[1] / scale + offset[1]
        K1 = np.array([line_integ_HD(l, qx, qy) for l in Ls1])
        v1 = np.abs(sum(K1)) * is_y_tangent(Ls1, qy)
        if v1 :
            K2 = np.array([line_integ_HD(l, qx, qy) for l in Ls2])
        else:
            K2 = np.zeros((len(Ls2)))
        v2 = np.abs(sum(K2)) * is_y_tangent(Ls2, qy)
        v = v1 * v2
        anp[ij] = v
        a = a + v

    return a/scale, anp, offset


def D(line):
    ax, ay = line[0]
    bx, by = line[1]
    return (bx - ax)/(by - ay) if by != ay else 0


def grad_overlap_polygon(polygon1, polygon2, qr, scale):
    Ls1 = get_line_list(polygon1)
    Ls2 = get_line_list(polygon2)
    anp, offset = get_raster_map(qr, scale)

    a = 0
    for ij in np.ndindex(anp.shape[:2]):
        qx, qy = ij[0] / scale + offset[0], ij[1] / scale + offset[1]
        K1 = np.array([line_integ_HD(l, qx, qy) for l in Ls1])
        v1 = np.abs(sum(K1)) * is_y_tangent(Ls1, qy)
        if v1 :
            K2 = np.array([D(l) * line_integ_HD(l, qx, qy) for l in Ls2])
        else:
            K2 = np.zeros((len(Ls2)))

        v2 = np.abs(sum(K2)) * is_y_tangent(Ls2, qy)
        v = v1 * v2
        anp[ij] = v
        a = a + v

    return a / scale, anp, offset

def test():
    scale = 2
    p2 = Art.Polygon([(2, 2), (2, -2), (-2, -2), (-2, 2)])
    theta = -np.pi / 4
    R = Art.Rotate(theta)
    p2.apply(R)
    Ls = get_line_list(p2)

    # px, py = -2.5, 2.5
    # for L in Ls:
    #     print ("Line : {}, Point: {} := {}".format(L, (px, py), line_integ_HD(L, px, py)))


    a, anp, offset = area_polygon(p2, [(-20, -20), (20, 20)], scale)

    d = Art.Draw()
    d.add_art(p2)
    d.add_raster(anp, offset, scale)
    d.draw()
