import numpy as np
import Art
import testsLevel1 as testsuite
import copy
import matplotlib.pyplot as plt
import DFT
import FunctionSimilarity
import LPMatching
import FunctionTransform as ft
import LPMatching as lp

ERROR = 1.e-12
def is_diff(p1, p2):
    return np.linalg.norm(p1 - p2) > ERROR


def sub(vec, l, size):
    if l + size <= vec.shape[0]:
        return vec[l : l + size]
    else:
        return np.append(vec[l:], vec[: l + size - vec.shape[0]], axis=0)


def draw_polygon(polygon, d, color =(0, 0, 0), open = False):
    iter_len = polygon.shape[:1][0] if not open else polygon.shape[:1][0] - 1
    for i in range(iter_len):
        c = Art.Circle(polygon[i], 0.1)
        c.set_color((255, 0, 0))
        d.add_art(c)
        l = Art.Line(polygon[i], polygon[(i + 1) % polygon.shape[0]])
        l.set_color(color)
        d.add_art(l)


def dft_descriptor_diff(poly1, i, poly2, j):
    piece_size = max(poly1.shape[0], poly2.shape[0])
    dft_dim = piece_size

    def vectorize(polygon, dimensions):
        angle, distance = ft.poly_to_turn_v_length(polygon, closed=False)
        return DFT.DFT_SIG(angle, distance, dimensions)

    sub_poly1, sub_poly2 = sub(poly1, i, piece_size), sub(poly2, j, piece_size)
    return np.linalg.norm(vectorize(sub_poly1, dft_dim) - vectorize(sub_poly2, dft_dim))


def poly_to_func(poly, start_index=0):
    sub_poly = sub(poly, start_index, poly.shape[0])
    a, d = ft.poly_to_turn_v_length(sub_poly, closed=True)
    d = d/d[-1]
    return a, d


def enclosed_area(poly1, i, poly2, j):
    a1, d1 = poly_to_func(poly1, i)
    a2, d2 = poly_to_func(poly2, j)
    fs = [(a1, d1), (a2, d2)]
    return FunctionSimilarity.diff(fs[0], fs[1], debug=False)


def cut_and_measure(polygon1, polygon2, debug):
    mat = np.zeros(shape=(polygon1.shape[0], polygon2.shape[0], 1))
    old = 0
    for ij in np.ndindex(mat.shape[:2]):
        i, j = ij
        if debug and old != i:
            old = i
            print(ij)

        mat[ij] = enclosed_area(poly1=polygon1, i=i, poly2=polygon2, j=j)
    return mat


def get_min(mat, already_matched):
    min_ind = 0, 0
    minimum = np.inf
    for ij in np.ndindex(mat.shape[:2]):
        if mat[ij] == np.nan:
            continue
        i, j = ij
        if i in already_matched:
            continue
        if minimum > mat[ij]:
            minimum = mat[ij]
            min_ind = ij
    return min_ind, mat[min_ind]


def poly_to_func_space(size, start_index, index):
    if index >= start_index:
        return index - start_index
    else:
        return size - start_index + index


def func_to_poly_space(size, start_index, index):
    if index + start_index < size-1:
        return index + start_index
    else:
        return index + start_index - size +1


def lean_marching(poly1, poly2, draw=None):
    sub_pair = []
    for i in range(poly1.shape[0]):
        if i % int(poly1.shape[0]/3) == 0:
            f1_i = poly_to_func(poly1, i)
            mat = np.zeros((poly2.shape[0]))
            for j in range(poly2.shape[0]):
                f2_j = poly_to_func(poly2, j)
                mat[j] = FunctionSimilarity.diff(f1_i, f2_j, debug=False)
            min_j = np.argmin(mat)
            f2 = poly_to_func(poly2, min_j)
            sub_pair.append((i, min_j))

    print(sub_pair)
    # if draw:
    #     pl1 = sub(poly1, 0, poly1.shape[0])
    #     pl2 = sub(poly2, min_j, poly1.shape[0])
    #     pairs = ft.match(f1, f2, [(0, 0)])
    #     print(pairs)
    #     Art.draw_match(pl1, pl2, pairs, draw)

    if draw:
        for p,q in sub_pair:
            p, q = poly1[p], poly2[q]
            c1, c2 = Art.Circle(p, 0.2), Art.Circle(q, 0.2)
            c1.set_fill_color((255, 0, 0)), c2.set_fill_color((255, 0, 0))
            draw.add_art(c1), draw.add_art(c2)
            l = Art.Line(p, q)
            l.set_color((0, 0, 255))
            draw.add_art(l)

    return sub_pair[0]


def marching(poly1, poly2, iter, draw=None): # list of pair of matching indexes and the corresponding measure
    count = 0
    polygon1, polygon2 = copy.deepcopy(poly1), copy.deepcopy(poly2)
    mat = cut_and_measure(polygon1, polygon2, draw)
    ret = []
    already_matched = dict()
    """The top iter least pair"""
    while count < iter:
        count = count + 1
        min_ind, min_val = get_min(mat, already_matched)
        if draw:
            print("Min[{}] := {}".format(min_ind, min_val))
        ret.append([min_ind, min_val[0]])
        already_matched[min_ind[0]] = min_ind[1]
        # mat[min_ind] = np.inf
        if draw :
            k, l = min_ind
            p,q = polygon1[k], polygon2[l]
            c1, c2 = Art.Circle(p, 0.2), Art.Circle(q, 0.2)
            c1.set_fill_color((255,0, 0)), c2.set_fill_color((255, 0, 0))
            draw.add_art(c1), draw.add_art(c2)
            l = Art.Line(p, q)
            l.set_color((0, 0, 255))
            draw.add_art(l)

    pairs = [p[0] for p in ret]
    least_i, least_j = pairs[0]
    f1, f2 = poly_to_func(polygon1, least_i), poly_to_func(polygon2, least_j)
    f_pairs = [(poly_to_func_space(len(f1[0]), least_i, p[0]), poly_to_func_space(len(f2[0]), least_j, p[1]))  for p in pairs]
    f_match = ft.match(f1, f2, forced_match=f_pairs)
    all_matching = [(func_to_poly_space(len(f1[0]), least_i, p[0]), func_to_poly_space(len(f2[0]), least_j, p[1])) for p in f_match]

    Art.draw_match(polygon1, polygon2, all_matching, d=draw)
    return ret


def test_at(poly1, poly2, i, j, draw):
    c1 = Art.Circle(poly1[i], 0.2)
    c1.set_fill_color((255, 0, 0))
    c2 = Art.Circle(poly2[j], 0.2)
    c2.set_fill_color((255, 0, 0))

    draw.add_art(c1)
    draw.add_art(c2)
    # draw_polygon(poly1, draw)
    # draw_polygon(poly2, draw)

    sub_poly1, sub_poly2 = sub(poly1, i, poly1.shape[0]), sub(poly2, j, poly2.shape[0])
    a1, d1 = ft.poly_to_turn_v_length(sub_poly1, closed=False)
    a2, d2 = ft.poly_to_turn_v_length(sub_poly2, closed=False)

    d1 = d1/d1[-1]
    d2 = d2/d2[-1]
    fs = [(a1, d1), (a2, d2)]

    integral = FunctionSimilarity.diff(fs[0], fs[1], debug=False)
    print(integral)


def measure(art1, art2, d):
    """list containing a pair of matching indexes and the corresponding measure which is least among all matches """
    art1.set_color((0, 0, 100))
    art2.set_color((0, 0, 100))

    polygon1, polygon2 = ft.piecewise_bezier_to_polygon(art=art1), ft.piecewise_bezier_to_polygon(art=art2)
    importance_angle = 10
    n_p1, n_p2 = ft.squint(polygon1, True, np.deg2rad(importance_angle)), ft.squint(polygon2, True, np.deg2rad(importance_angle))

    if d:
        print(len(polygon1), len(polygon2))
        print(len(n_p1), len(n_p2))
        # draw_polygon(n_p1, d)
        # draw_polygon(n_p2, d)

    # if d: test_at(n_p1, n_p2, 26, 23, d)
    return marching(poly1=n_p1, poly2=n_p2, iter=5, draw=d)


def lean_measure(art1, art2, d):
    """Gives the best match for index 0 of art1 with some index j of art2"""
    art1.set_color((0, 0, 100))
    art2.set_color((0, 0, 100))
    polygon1, polygon2 = ft.piecewise_bezier_to_polygon(art=art1), ft.piecewise_bezier_to_polygon(art=art2)
    importance_angle = 10
    n_p1, n_p2 = ft.squint(polygon1, True, np.deg2rad(importance_angle)), ft.squint(polygon2, True,
                                                                                    np.deg2rad(importance_angle))
    if d:
        print(len(polygon1), len(polygon2))
        print(len(n_p1), len(n_p2))

    pair = lean_marching(poly1=n_p1, poly2=n_p2, draw=d)
    pairs = ft.match(poly_to_func(n_p1, pair[0]), poly_to_func(n_p2, pair[1]), [(0, 0)])
    return pairs


def main():
    d = Art.Draw()
    ags = testsuite.get_test(5)
    art1, art2 = ags[0], ags[1]
    # art1.apply(Art.Translate([-10, 0]))
    # art2.apply(Art.Translate([10, 0]))

    d.add_art(art1)
    d.add_art(art2)
    # pair = measure(art1, art2, d)
    pair = lean_measure(art1, art2, d)
    print(pair)
    d.draw()


if __name__ == '__main__':
    main()
