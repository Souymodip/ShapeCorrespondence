import numpy as np
import ArtCollection
import Art
import matplotlib.pyplot as plt


SIGMA = 1
SUPPORT = 5
THRESHOLD = 0.07


def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.))) / (np.sqrt(2 * np.pi) * sigma)


def get_context_distance(art, index, support):
    points = art.get_vertices()
    point = art.get_vertex(index)
    c = np.array([np.linalg.norm(point - p) for p in points])
    c.sort()

    def G(d):
        return d
        return gaussian(x=d, mu=0, sigma=SIGMA)

    gc = np.vectorize(G)(c[:support])
    return gc


def get_context_arclength(art, index, support):
    beziers = art.get_beziers()
    assert (index <= len(beziers))
    # left
    cl = []
    last = 0
    for i in range(index - 1, -1, -1):
        last = last + beziers[i].length()
        cl.insert(0, last)
    # right
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

    gc = np.vectorize(G)(c[:support])
    return gc


# arts are piecewise-bezier curves
def find_match(art1, art2, distance_vs_arclength):
    context1, context2 = np.zeros((art1.size() + 1, SUPPORT)), np.zeros((art2.size() + 1, SUPPORT))
    for i in range(context1.shape[0]):
        context1[i] = get_context_distance(art1, i, support=SUPPORT) if distance_vs_arclength else get_context_arclength(art1, i, support=SUPPORT)
    for j in range(context2.shape[0]):
        context2[j] = get_context_distance(art2, j, support=SUPPORT) if distance_vs_arclength else get_context_arclength(art2, j, support=SUPPORT)

    # Greedy
    matching = np.zeros((context1.shape[0]))
    for i in np.ndindex(context1.shape[0]):
        c1 = context1[i]

        def f(c2):
            return np.linalg.norm(c1 - c2)

        diff = np.apply_along_axis(f, 1, context2)
        matching[i] = np.argmin(diff)
    return matching


def transform(art1, art2, matching, threshold, draw, distance_vs_arclength):
    for i in range(len(matching)):
        j = int(matching[i])
        p = art1.get_vertex(i)
        q = art2.get_vertex(j)
        c1 = get_context_distance(art1, i, support=SUPPORT) if distance_vs_arclength else get_context_arclength(art1, i, support=SUPPORT)
        c2 = get_context_distance(art2, j, support=SUPPORT) if distance_vs_arclength else get_context_arclength(art2, i, support=SUPPORT)
        similarity = np.linalg.norm(c1 - c2)
        if similarity <= threshold:
            print("Matching similarity: {}".format(np.linalg.norm(c1 - c2)))
            point1 = Art.Circle(p, 0.3)
            point1.set_color((255, 0, 0))
            point2 = Art.Circle(q, 0.3)
            point2.set_color((0, 0, 255))
            l = Art.Line(p, q)
            l.set_color((0, 255, 0))
            draw.add_art(point1)
            draw.add_art(point2)
            draw.add_art(l)

            b1 = art1.get_bezier(i)
            b11 = art1.get_bezier(i-1)
            b2 = art2.get_bezier(j)
            b21 = art2.get_bezier(j-1)
            b1.set_color((255, 0, 0))
            b11.set_color((255, 0, 0))
            b2.set_color((0, 0, 255))
            b21.set_color((0, 0, 255))

            draw.add_art(b1)
            draw.add_art(b11)
            draw.add_art(b2)
            draw.add_art(b21)


def trees():
    t1 = ArtCollection.tree1
    t1.apply(Art.Translate([-3.5, 1]))
    t2 = ArtCollection.tree2
    t2.apply(Art.Translate([0, 3]))
    t1.apply(Art.Scale(5))
    t2.apply(Art.Scale(5))
    return t1, t2


def lion_gorilla():
    t1 = ArtCollection.lion
    t1.apply(Art.Scale(5))
    t1.apply(Art.Translate([-15, -11]))
    t2 = ArtCollection.gorilla
    t2.apply(Art.Scale(5))
    t2.apply(Art.Translate([2, -1]))
    return t1, t2


def main():
    d = Art.Draw()
    t1, t2 = lion_gorilla()

    dis_vs_length= False
    matching = find_match(art1=t1, art2=t2, distance_vs_arclength=dis_vs_length)
    transform(art1=t1, art2=t2, matching=matching, threshold=THRESHOLD, draw=d, distance_vs_arclength=dis_vs_length)

    d.add_art(t1)
    d.add_art(t2)
    d.draw()


main()
