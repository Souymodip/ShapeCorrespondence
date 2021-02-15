import numpy as np
from scipy.optimize import linprog
import testsLevel1 as testsuite
import ShapeSimilarity as ss
import FunctionTransform as ft
import Art
import copy as cp


def get_lhs(start, width, total):
    assert (total % width == 0)
    ret = []
    for i in range(int(total / width)):
        if i == start:
            ret = ret + width * [1.0]
        else:
            ret = ret + width * [0.0]
    return ret


def march(point_list1, point_list2, draw=None):
    # pl1, pl2 = copy.deepcopy(point_set1), copy.deepcopy(point_set2)
    pl1, pl2 = point_list1, point_list2
    print(len(pl1), len(pl2))
    flipped = False
    if len(pl1) > len(pl2):
        tmp = pl1
        pl1 = pl2
        pl2 = tmp
        flipped = True

    row, col = len(pl1), len(pl2)
    c1, c2 = np.mean(pl1, axis=0), np.mean(pl2, axis=0)
    print(c1, c2)
    dim = row * col
    obj = np.zeros(shape=(dim))
    for k in range(dim):
        i = int(k / col)
        j = k % col
        obj[k] = np.abs(np.linalg.norm(pl1[i] - c1) - np.linalg.norm(pl2[j] - c2))

    lhs_ineq = [np.ones(shape=(dim))]
    rhs_ineq = [len(pl1)]

    lhs_eq = [get_lhs(i, len(pl2), dim) for i in range(row)]
    rhs_eq = np.ones(shape=(row))

    bnd = [(0, 1.0) for i in range(dim)]

    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd, method="revised simplex")

    x = np.zeros(shape=(len(point_list1), len(point_list2)))
    match = []
    for i in range(row):
        for j in range(col):
            opt_ij = opt.x[i * col + j]
            if opt_ij > 0.5:
                match.append([j, i] if flipped else [i, j])
            x[j, i] = opt_ij

    if draw:
        for i in range(len(lhs_ineq)):
            print ("InEQ: {} <= {}".format(lhs_ineq[i], rhs_ineq[i]))
        for i in range(len(lhs_eq)):
            print ("EQ  : {} = {}".format(lhs_eq[i], rhs_eq[i]))
        for i in range(len(point_list1)):
            print(x[i])

    return np.array(match, dtype=int)


def solve_min_energy(mat, force_matching_list):
    map = dict(force_matching_list)
    left, right = mat.shape[0], mat.shape[1]
    assert (left <= right)
    obj = mat.flatten()

    max_match = min(left, right)
    lhs_ineq = [np.ones(shape=(left*right))]
    rhs_ineq = [max_match]

    def get_one_at(i, j):
        k = np.zeros((left*right))
        k[i*right + j] = 1.0
        return k

    lhs_eq = []
    rhs_eq = []
    for i in range(left):
        if i in map:
            j = map[i]
            lhs_eq.append(get_one_at(i, j))
        else:
            lhs_eq.append(get_lhs(i, right, left*right))
        rhs_eq.append(1.0)

    bnd = [(0, 1.0) for i in range(left*right)]
    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd, method="revised simplex")

    x = np.zeros(shape=(left, right))
    match = []
    for i in range(left):
        for j in range(right):
            opt_ij = opt.x[i * right + j]
            if opt_ij > 0.5:
                match.append([j, i])
            x[j, i] = opt_ij

    print(x)
    return match


def main():
    art1, art2 = testsuite.get_test(4)
    d = Art.Draw()
    art1.set_color((0, 0, 100))
    art2.set_color((0, 0, 100))

    d.add_art(art1)
    d.add_art(art2)

    polygon1, polygon2 = ft.piecewise_bezier_to_polygon(art=art1), ft.piecewise_bezier_to_polygon(art=art2)
    importance_angle = 15
    n_p1 = ft.squint(polygon1, True, np.deg2rad(importance_angle))
    n_p2 = ft.squint(polygon2, True, np.deg2rad(importance_angle))

    match = march(point_list1=n_p1, point_list2=n_p2)

    for k in range(int(len(match)/3)):
        k = 3*k
        i, j = match[k][0], match[k][1]
        p, q = n_p1[i], n_p2[j]
        c1, c2 = Art.Circle(p, 0.2), Art.Circle(q, 0.2)
        c1.set_fill_color((255, 0, 0)), c2.set_fill_color((255, 0, 0))
        d.add_art(c1), d.add_art(c2)
        l = Art.Line(p, q)
        l.set_color((0, 0, 255))
        d.add_art(l)

    d.draw()


if __name__ == '__main__':
    main()
