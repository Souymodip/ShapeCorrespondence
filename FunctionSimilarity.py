import numpy as np
import matplotlib.pyplot as plt


def draw_graph(fs, last_scatter=False):
    for i in range(len(fs)):
        f = fs[i]
        ys, xs = f
        assert (xs.shape[0] == ys.shape[0])
        if not last_scatter or (i != len(fs) - 1 or i <= 1):
            plt.plot(xs, ys, 'o-')
        else:
            plt.scatter(xs, ys)
    plt.show()


def y_at(y_, y, x_, x, x_at):
    if x == x_:
        return y
    return y_ + (y - y_) * (x_at - x_) / (x - x_)


def merge(ys1, xs1, ys2, xs2):
    i, j = 0, 0
    ys, xs = [], []
    while i < len(xs1) or j < len(xs2):
        if i < len(xs1) and j < len(xs2):
            if xs1[i] == xs2[j]:
                xs.append(xs1[i])
                ys.append([ys1[i], ys2[j]])
                i = i + 1
                j = j + 1
            elif xs1[i] < xs2[j]:
                if len(ys) == 0:
                    y_, x_ = 0, 0
                else:
                    y_, x_ = ys[-1][1], xs[-1]
                xs.append(xs1[i])
                ys.append([ys1[i], y_at(y_, ys2[j], x_, xs2[j], xs1[i])])
                i = i + 1
                j = j
            else:
                if len(ys) == 0:
                    y_, x_ = 0, 0
                else:
                    y_, x_ = ys[-1][0], xs[-1]
                xs.append(xs2[j])
                ys.append([ y_at(y_, ys1[i], x_, xs1[i], xs2[j]), ys2[j]] )
                i = i
                j = j + 1
        elif i < len(xs1):
            xs.append(xs1[i])
            ys.append([ys1[i], 0])
            i = i + 1
            j = j + 1
        elif j < len(xs2):
            xs.append(xs2[j])
            ys.append([0, ys2[j]])
            i = i + 1
            j = j + 1
        else:
            i = i + 1
            j = j + 1
    return np.array(ys), np.array(xs)


def enclosed_area(f1, f2, debug):
    y1, x1 = f1
    y2, x2 = f2

    ys , xs = merge(y1, x1, y2, x2)
    assert (len(xs) == len(ys))
    if debug:
        fy , fx = [ys[0]], [xs[0]]
    else:
        fy, fx = [], []

    integral = 0
    for i in range(1, len(xs)):
        y1_, y2_, x_ = ys[i-1][0], ys[i-1][1], xs[i-1]
        y1, y2, x = ys[i][0], ys[i][1], xs[i]
        if (y1_ - y2_) * (y1 - y2) < 0: # intersection
            x_in = x_ + (y2_ - y1_) *(x - x_) / ( (y1 - y1_) - (y2 - y2_) )
            y_in = y_at(y1_, y1, x_, x, x_in)
            if debug:
                fy.append([y_in, y_in])
                fx.append(x_in)
            integral = integral + 0.5 * (np.abs((y1_ - y2_) * (x_in - x_)) + np.abs((y1 - y2) * (x_in - x)))
        else:
            integral = integral + np.abs( (x - x_) * ((y1 + y1_)/2 - (y2 + y2_)/2))

        if debug:
            fy.append(ys[i])
            fx.append(x)

    return fy, fx, integral


def diff(f1, f2, debug=False):
    fy, fx, integral = enclosed_area(f1, f2, debug)
    if debug:
        y1 = np.array([y[0] for y in fy])
        y2 = np.array([y[1] for y in fy])
        x = np.array(fx)
        draw_graph([(y1, x), (y2, x)])
    return integral


def main():
    y1, x1 = np.array([0, 3]), np.array([0, 1])
    y2, x2 = np.array([3, 0.5]), np.array([0, 1.5])

    ys, xs = merge(y1, x1, y2, x2)
    xs = np.array(xs)
    ys0 = np.array([y[0] for y in ys])
    ys1 = np.array([y[1] for y in ys])

    draw_graph([(ys0, xs), (ys1, xs)])

    print(ys0)
    print(ys1)
    print(xs)

    print()

    fy, fx, integral = enclosed_area((y1, x1), (y2, x2))

    y1 = np.array([y[0] for y in fy])
    y2 = np.array([y[1] for y in fy])
    x = np.array(fx)
    print(integral)
    print(y1)
    print(y2)
    print(x)
    draw_graph([(y1, x), (y2, x)])
