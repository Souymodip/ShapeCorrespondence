import numpy as np
import ArtCollection
import Art
import matplotlib.pyplot as plt

SIGMA = 1
SUPPORT = 8


def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.))) / (np.sqrt(2 * np.pi) * sigma)


def get_context(points, point):
    c = np.array([np.linalg.norm(point - p) for p in points])
    n = normalize(c)

    def G(d):
        return gaussian(d, 0, SIGMA)
    gc = n #np.vectorize(G)(n)
    # gc[::-1].sort()
    gc.sort()
    return gc


def get_context_bezier(beziers, index):
    assert (index <= len(beziers))
    #     left
    cl = []
    last = 0
    for i in range(index - 1, -1, -1):
        last = last + beziers[i].length()
        cl.insert(0, last)
    #   right
    if index < len(beziers):
        last = beziers[index].length()
        cr = [last]
    else:
        last = 0
        cr = []
    for i in range(index + 1, len(beziers)):
        last = last + beziers[i].length()
        cr.append(last)

    c = np.sort(np.array(cl + cr))
    def G(d):
        return gaussian(d, 0, SIGMA)
    gc = np.vectorize(G)(c)
    return gc[:SUPPORT]


def main():
    t1= ArtCollection.tree1
    t1.apply(Art.Translate([-3, 2]))
    t2 = ArtCollection.tree2
    t2.apply(Art.Translate([0, 2]))
    t1.apply(Art.Scale(6))
    t2.apply(Art.Scale(6))

    bs1 = t1.get_beziers()
    bs2 = t2.get_beziers()

    norms = []
    for i in range(len(bs1) + 1):
        c1 = get_context_bezier(bs1, i)
        c2 = get_context_bezier(bs2, i)
        n = np.linalg.norm(c1 - c2)
        print("c1 :{}\nc2 :{}\n L2Norm :{}\n".format(c1, c2, n))
        norms.append(n)
    #
    #  Vertices
    # points1 = t1.get_vertices()
    # points2 = t2.get_vertices()
    #
    # norms = []
    # for p1, p2 in zip(points1, points2):
    #     c1 = get_context(points1, p1)
    #     c2 = get_context(points2, p2)
    #     n = np.linalg.norm(c1 - c2)
    #     print("c1 :{}\nc2 :{}\n L2Norm :{}\n".format(c1, c2, n))
    #     norms.append(n)


    plt.plot(list(range(len(norms))), norms)
    plt.show()


    d = Art.Draw()
    d.add_art(t1)
    d.add_art(t2)
    d.draw()

main()