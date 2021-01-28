import matplotlib.pyplot as plt
import Art
import numpy as np
import testsLevel1 as testsuite
import ShapeSimilarity as ss
import FunctionSimilarity as fs
import copy as cp


def sub_sampler(f, fraction):
    y, x = f
    size = int(x.size * fraction)
    indexes = np.random.choice(np.arange(x.size), size)
    new_x , new_y = [], []
    for i in indexes:
        new_x.append(x[i])
        new_y.append(x[i])
    return np.array(new_y), np.array(new_x)


def perturb_x(f, px, interval):
    yx, xx = f
    y = yx #cp.deepcopy(yx)
    x = xx #cp.deepcopy(xx)
    assert (x.size == y.size)
    r = np.random.rand(x.size)
    for i in range(1, x.size-1):
        if r[i] <= px:
            low = max (x[i-1], x[i] - interval/2)
            high = min (x[i+1], x[i] + interval/2)
            x[i] = np.random.uniform(low, high, 1)[0]
    return y, x


def perturb_y(f, py, interval):
    yx, xx = f
    y = yx #cp.deepcopy(yx)
    x = xx #cp.deepcopy(xx)
    assert (x.size == y.size)
    r = np.random.rand(x.size)
    for i in range(1, x.size-1):
        if r[i] <= py:
            v = interval/3
            y[i] = y[i] + np.random.normal(0, v, 1)[0]
    return y, x


def cycle(f):
    y, x = f
    assert(x.size == y.size)
    index = np.random.randint(0, x.size, 1)[0]
    if index == 0:
        return y, x

    n_x = np.concatenate((x[index : ] -x[index], x[1: index] + 1 -x[index]))
    n_y = np.concatenate((y[index : ], y[1: index]))

    n_x = np.append(n_x, 1)
    n_y = np.append(n_y, n_y[0])

    return n_y, n_x


def random(f_dim, low_y, high_y):
    assert (f_dim > 2)
    x = np.random.sample((f_dim-1,))
    x[0]= 0.0
    x = np.append(x, 1.0)
    x = np.sort(x)
    assert(len(set(x)) == x.size)
    assert (high_y > low_y)
    y = (high_y - low_y) * np.random.rand(f_dim) + low_y
    y[0] = y[-1]
    return y, x



def main():
    d = Art.Draw()
    art1, art2 = testsuite.get_test(4)
    polygon1, polygon2 = ss.piecewise_bezier_to_polygon(art=art1), ss.piecewise_bezier_to_polygon(art=art2)
    importance_angle = 15

    n_p1 = ss.squint(polygon1, True, np.deg2rad(importance_angle))
    a1, d1 = ss.poly_to_turn_v_length(n_p1, closed=False)
    d1 = d1 / d1[-1]

    for i in range(1):
        a = cp.deepcopy(a1)
        d = cp.deepcopy(d1)
        a2, d2 = perturb_x((a, d), 0.9, 0.2)
        a3, d3 = perturb_y((a2, d2), 0.9, 0.1)
        a4, d4 = cycle((a3, d3))
        a5, d6 = random(a4.size, np.min(a4), np.max(a4))
        fs.draw_graph([(a, d), (a4, d4), (a5, d6)])



if __name__ == '__main__':
    main()