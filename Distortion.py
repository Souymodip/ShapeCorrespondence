import numpy as np
import Art

def distance(p1, point_list):
    n = [np.linalg.norm(p1 - p) for p in point_list]
    # print("||{} - {}|| := {}".format(p1, point_list, n))
    return np.sum(n)


def context(pl):
    return np.array([distance(p, pl) for p in pl])


def match(pl1, pl2):
    target = context(pl2)
    print ("Target Context {}".format(target))
    source = context(pl1)
    print("Source Context {}".format(source))
    distortions = np.array([np.linalg.norm(np.roll(source, i) - target) for i in range(len(pl1))])
    print(distortions)
    return np.argmin(distortions)

def main():
    d = Art.Draw()

    '''Creating two arts at different location by defining the list of [anchor, in, out]'''
    l1 = Art.PieceWiseBezier(np.array([
        [[-1, 0.5], [-2, 0.5], [-3, 0]],
        [[-3, -2], [-3, -1.5], [-3, -6]],
        [[3, -2], [3, -6], [3, -1.5]],
        [[1, 0.5], [3, 0], [2, 0.5]],
        [[1.5, 2.5], [2, 1.5], [1, 3]],
        [[-1.5, 2.5], [-1, 3], [-2, 1.5]]
    ]), is_closed=True, show_control=False)

    T = Art.Translate([10, 10])
    l1.apply(T)
    d.add_art(l1)

    l2 = Art.PieceWiseBezier(np.array([
        [[-3, -2], [-3, -1.5], [-3, -2]],
        [[3, -2], [3, -2], [3, -1.5]],
        [[1, 0.5], [3, 0], [2, 0.5]],
        [[1.5, 2.5], [2, 1.5], [1, 3]],
        [[-1.5, 2.5], [-1, 3], [-2, 1.5]],
        [[-1, 0.5], [-2, 0.5], [-3, 0]]
    ]), is_closed=True, show_control=False)

    d.add_art(l2)
    R = Art.Rotate(np.deg2rad(90))
    l2.apply(R)

    ''' We are choosing the list of vertices for distortion calculation. 
    Dynamic programming based distortion minimization can be used to remove this restriction. Furthermore, we are using 
    only the anchors of the curve. Once can envisage distortion calculation involving the control points as well'''
    v2 = l2.get_vertices()
    v1 = l1.get_vertices()
    m = match(v2, v1)
    print(m)

    print(v2)
    v2 = np.roll(v2, m, axis=0)
    print(v2)

    for p, q in zip(v2, v1):
        line = Art.Line([p[0], p[1]], [q[0], q[1]])
        line.set_color((255, 0, 0))
        d.add_art(line)

    p, q = v2[0], v1[0]
    line = Art.Line([p[0], p[1]], [q[0], q[1]])
    line.set_color((0, 255, 0))
    d.add_art(line)

    d.draw()
