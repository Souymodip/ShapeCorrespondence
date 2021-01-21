import numpy as np
import Art
import testsLevel1 as testsuite
import copy
import matplotlib.pyplot as plt
import DFT
import FunctionSimilarity


ERROR = 1.e-12
def is_diff(p1, p2):
    return np.linalg.norm(p1 - p2) > ERROR


def sub(vec, l, size):
    if l + size <= vec.shape[0]:
        return vec[l : l + size]
    else:
        return np.append(vec[l:], vec[: l + size - vec.shape[0]], axis=0)


# radian at p1
def turn_angle(p0, p1, p2):
    a, b = p1 - p0, p2 - p1

    if np.linalg.norm(a) * np.linalg.norm(b) == 0:
        print(p0, p1, p2)
    assert (np.linalg.norm(a) * np.linalg.norm(b) > 0 )
    return np.arcsin(np.cross(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def draw_polygon(polygon, d, color =(0, 0, 0), open = False):
    iter_len = polygon.shape[:1][0] if not open else polygon.shape[:1][0] - 1
    for i in range(iter_len):
        c = Art.Circle(polygon[i], 0.1)
        c.set_color((255, 0, 0))
        d.add_art(c)
        l = Art.Line(polygon[i], polygon[(i + 1) % polygon.shape[0]])
        l.set_color(color)
        d.add_art(l)


def piecewise_bezier_to_polygon(art):
    polygon = []
    # for i in range(art.no_of_beziers()):
    #     art.split_bezier_in_parts(i, 30)

    for b in art.get_beziers():
        # ex = b.get_extremes()
        if len(polygon) == 0 or is_diff(polygon[-1], b.controls[0]):
            polygon.append(b.controls[0])
        # for e in ex:
        #     if len(polygon) == 0 or is_diff(polygon[-1], b.controls[0]):
        #         polygon.append(e)
    return np.array(polygon)


# polygon is a counter clockwise sequence of vertices
def poly_to_turn_v_length(polygon, closed=True):
    assert (len(polygon) > 2)
    radians, length = np.zeros((len(polygon)), dtype=float), np.zeros((len(polygon)), dtype=float)
    for i in range(len(polygon)):
        p0 = polygon[i-1]
        p1 = polygon[i]
        p2 = polygon[(i+1) % polygon.shape[0]]

        if i != 0:
            length[i] = length[i-1] + np.linalg.norm(p1 - p2)

        if closed or (i != 0 and i != len(polygon)-1): # first and last have zero
            radians[i] = turn_angle(p0, p1, p2)
    return radians, length


def dft_descriptor_diff(poly1, i, poly2, j):
    piece_size = max(poly1.shape[0], poly2.shape[0])
    dft_dim = piece_size

    def vectorize(polygon, dimensions):
        angle, distance = poly_to_turn_v_length(polygon, closed=False)
        return DFT.DFT_SIG(angle, distance, dimensions)

    sub_poly1, sub_poly2 = sub(poly1, i, piece_size), sub(poly2, j, piece_size)
    return np.linalg.norm(vectorize(sub_poly1, dft_dim) - vectorize(sub_poly2, dft_dim))


def enclosed_area(poly1, i, poly2, j):
    sub_poly1, sub_poly2 = sub(poly1, i, poly1.shape[0]), sub(poly2, j, poly2.shape[0])
    a1, d1 = poly_to_turn_v_length(sub_poly1, closed=False)
    a2, d2 = poly_to_turn_v_length(sub_poly2, closed=False)

    d1 = d1/d1[-1]
    d2 = d2/d2[-1]
    fs = [(a1, d1), (a2, d2)]
    return FunctionSimilarity.diff(fs[0], fs[1])


def cut_and_measure(polygon1, polygon2):
    mat = np.zeros(shape=(polygon1.shape[0], polygon2.shape[0], 1))
    old = 0
    for ij in np.ndindex(mat.shape[:2]):
        i, j = ij
        if old != i:
            old = i
            print(ij)

        mat[ij] = enclosed_area(poly1=polygon1, i=i, poly2=polygon2, j=j)
    return mat


def test_single(ij, poly1, poly2):
    polygon1, polygon2 = poly1, poly2
    i,j = ij
    piece_size = 10
    dft_dim = 10

    sub_poly1, sub_poly2 = sub(polygon1, i, piece_size), sub(polygon2, j, piece_size)
    a1, d1 = poly_to_turn_v_length(sub_poly1, closed=False)
    a2, d2 = poly_to_turn_v_length(sub_poly2, closed=False)

    dft1, dft2 = DFT.DFT_SIG(a1, d1, dft_dim), DFT.DFT_SIG(a2, d2, dft_dim)
    print(dft1)
    print(dft2)
    print(np.linalg.norm(dft1 - dft2))

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    ax1.plot(d1, a1)
    ax2.plot(d2, a2)
    plt.show()


def get_min(mat):
    min_ind = 0, 0
    minimum = np.inf
    for ij in np.ndindex(mat.shape[:2]):
        if mat[ij] == np.nan:
            continue
        if minimum > mat[ij]:
            minimum = mat[ij]
            min_ind = ij
    return min_ind, mat[min_ind]


def marching(poly1, poly2, iter, draw):
    count = 0
    polygon1, polygon2 = copy.deepcopy(poly1), copy.deepcopy(poly2)
    mat = cut_and_measure(polygon1, polygon2)

    while count < iter:
        count = count + 1
        min_ind, min_val = get_min(mat)
        print("Min[{}] := {}".format(min_ind, min_val))
        mat[min_ind] = np.inf
        k, l = min_ind
        p,q = polygon1[k], polygon2[l]
        c1, c2 = Art.Circle(p, 0.2), Art.Circle(q, 0.2)
        c1.set_fill_color((255,0, 0)), c2.set_fill_color((255, 0, 0))
        draw.add_art(c1), draw.add_art(c2)
        l = Art.Line(p, q)
        l.set_color((0, 0, 255))
        draw.add_art(l)


def squint(polygon, is_closed, tolerance):
    new_polygon = []
    for i in range(len(polygon)):
        if not is_closed and (i == 0 or i == len(polygon) - 1):
            new_polygon.append(polygon[i])
            continue
        else:
            p0 = polygon[(i + 1) % len(polygon)]
            p1 = polygon[i]
            p2 = polygon[i - 1]

            if np.abs(turn_angle(p0, p1, p2)) >= tolerance:
                new_polygon.append(p1)
    return np.array(new_polygon)


def test_at(poly1, poly2, i, j, draw):
    c1 = Art.Circle(poly1[i], 0.2)
    c1.set_fill_color((255, 0, 0))
    c2 = Art.Circle(poly2[j], 0.2)
    c2.set_fill_color((255, 0, 0))

    draw.add_art(c1)
    draw.add_art(c2)
    draw_polygon(poly1, draw)
    draw_polygon(poly2, draw)

    draw.draw()

    sub_poly1, sub_poly2 = sub(poly1, i, poly1.shape[0]), sub(poly2, j, poly2.shape[0])
    a1, d1 = poly_to_turn_v_length(sub_poly1, closed=False)
    a2, d2 = poly_to_turn_v_length(sub_poly2, closed=False)

    d1 = d1/d1[-1]
    d2 = d2/d2[-1]
    fs = [(a1, d1), (a2, d2)]

    integral = FunctionSimilarity.diff(fs[0], fs[1], debug=False)
    print(integral)


def main():
    d = Art.Draw()
    art1, art2 = testsuite.get_test(1)

    polygon1, polygon2 = piecewise_bezier_to_polygon(art=art1), piecewise_bezier_to_polygon(art=art2)
    print(len(polygon1), len(polygon2))

    importance_angle = 5

    n_p1, n_p2 = squint(polygon1, True, np.deg2rad(importance_angle)), squint(polygon2, True, np.deg2rad(importance_angle))
    print(len(n_p1), len(n_p2))
    draw_polygon(n_p1, d)
    draw_polygon(n_p2, d)

    # test_at(n_p1, n_p2, 18, 23, d)
    marching(poly1=n_p1, poly2=n_p2, iter=10, draw=d)
    d.draw()


main()
