import numpy as np
import matplotlib.pyplot as plt
import testsLevel2 as testsuite
import Art


def y_at(y_, y, x_, x, x_at):
    if x == x_:
        return y
    return y_ + (y - y_) * (x_at - x_) / (x - x_)


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


def get_enclosing_range(f, val_x, index):
    y, x = f
    assert (index < len(x))
    i = index
    val_y = None
    while i < len(index) and x[i] < val_x:
        i = i + 1
    if i == len(index):
        val_y = y[-1]
    elif x[i] == val_x:
        val_y = y[i]
    else:
        if i == 0:
            assert(x[i] > 0)
            val_y = 0 + (y[i] - 0) * (val_x - 0) / (x[i] - 0)
        else:
            assert (x[i] > x[i-1])
            val_y = y[i-1] + (y[i] - y[i-1]) * (val_x - x[i-1]) / (x[i] - x[i-1])
    return val_y, i


def merge_along_x(fs):
    xs = set()
    for f in fs:
        xs = xs + set(f[1])

    xs = sorted(xs)

    indexes = np.zeros((len(fs)), dtype=int)
    ys = [[] for i in range(len(fs))]
    for x in xs:
        for i in range(len(fs)):
            ind_i = indexes[i]
            f_i = fs[i]
            if x == f_i[1][ind_i]:
                ys[i].append(f_i[0][ind_i])
                indexes[i] = indexes[i] + 1
            else:
                y, index = get_enclosing_range(f_i, x, ind_i)
                assert(y)
                ys[i].append(y)
                indexes[i] = index
    return ys, xs


def change_suport(f, new_x):
    """ returns a function whose support is new_x"""
    new_y = []
    index = 0
    for x_val in new_x:
        y_val, index = get_enclosing_range(f, x_val, index)
        assert (y_val)
        new_y.append(y_val)

    return np.array(new_y), new_x


ERROR = 1.e-12
def is_diff(p1, p2):
    return np.linalg.norm(p1 - p2) > ERROR


def piecewise_bezier_to_polygon(art):
    polygon = []
    for b in art.get_beziers():
        ex = b.get_extremes()
        if len(polygon) == 0 or is_diff(polygon[-1], b.controls[0]):
            polygon.append(b.controls[0])
        for e in ex:
            if len(polygon) == 0 or is_diff(polygon[-1], e):
                polygon.append(e)
    return np.array(polygon)


def turn_angle(p0, p1, p2):
    """radian at p1"""
    a, b = p1 - p0, p2 - p1

    def angle(x, y):
        if x == 0: return np.pi/2 if y >=0 else np.pi * 3 / 2
        else:
            theta = np.arctan(np.abs(y)/np.abs(x))
            if y >=0 and x >=0:
                return theta
            if y >=0 and x < 0:
                return np.pi - theta
            if y < 0 and x < 0:
                return np.pi + theta
            else:
                return 2 * np.pi - theta

    if np.linalg.norm(a) * np.linalg.norm(b) == 0:
        return 0.0
    else:
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        atheta, btheta = angle(a[0], a[1]), angle(b[0], b[1])
        t = btheta - atheta
        t = t if t >=0 else 2*np.pi + t
        return t


def squint(polygon, is_closed, tolerance):
    if tolerance == 0.0:
        return polygon
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


def closed_poly_to_turn_v_lenght(polygon):
    assert(len(polygon) > 2)
    radians, length = np.zeros((len(polygon) + 1), dtype=float), np.zeros((len(polygon) + 1), dtype=float)
    radians[0] = turn_angle(polygon[-1], polygon[0], polygon[1])
    length[0] = 0.0

    for i in range(1, len(polygon) + 1):
        p0 = polygon[i - 1]
        p1 = polygon[i%len(polygon)]
        p2 = polygon[(i + 1) % len(polygon)]

        radians[i] = turn_angle(p0, p1, p2)
        length[i] = length[i-1] + np.linalg.norm(p1 - p0)
    return radians, length/length[-1]


# polygon is a counter clockwise sequence of vertices
def poly_to_turn_v_length(polygon, closed=True):
    assert (len(polygon) > 2)
    if closed:
        return closed_poly_to_turn_v_lenght(polygon)
    else:
        radians, length = np.zeros((len(polygon)), dtype=float), np.zeros((len(polygon)), dtype=float)
        length[0] = 0.0
        radians[0] = 0.0
        for i in range(1, len(polygon)-1):
            p0 = polygon[i-1]
            p1 = polygon[i]
            p2 = polygon[(i+1)]
            length[i] = length[i-1] + np.linalg.norm(p1 - p0)
            radians[i] = turn_angle(p0, p1, p2)
        length[-1] = length[-2] + np.linalg.norm(polygon[-2] - polygon[-1])
        radians[-1] = 0.0
        return radians, length/length[-1]


def roll(poly, amount):
    amount = amount % poly.shape[0]
    if amount == 0:
        return poly
    else:
        return np.append(poly[amount:], poly[:amount], axis=0)


def art_to_function(art, importance_angle = 15):
    polygon = piecewise_bezier_to_polygon(art=art)

    n_p = squint(polygon, True, np.deg2rad(importance_angle))
    a1, d1 = poly_to_turn_v_length(n_p, closed=True)
    d1 = d1 / d1[-1]
    return a1, d1


def func_to_poly(f, mult):
    y, x = f
    size = len(y)
    assert size == len(x)
    poly = [[0.0, 0.0]]
    direction = y[0]
    for i in range(1, size):
        r = (x[i] - x[i-1]) * mult
        last = poly[-1]
        # print("{}:D:{:.2f}".format(i, np.rad2deg(direction)))
        next = last[0] + r * np.cos(direction), last[1] +  r * np.sin(direction)
        poly.append(next)
        direction = y[i] + direction
    return np.array(poly)


def func_to_polygon(f, mult):
    return Art.Polygon(func_to_poly(f, mult))


def main():
    a1, a2 = testsuite.get_test(1)
    d = Art.Draw()
    f1, f2 = art_to_function(a1), art_to_function(a2)
    # draw_graph([f1, f2])
    # print(p)
    # d.add_art(a1)
    # d.add_art(a2)
    # d.draw()


if __name__ == '__main__':
    main()