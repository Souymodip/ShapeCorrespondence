import numpy as np
import ArtCollection
import Art
import matplotlib.pyplot as plt
from enum import Enum
import copy
import itertools


SIGMA = 2
SUPPORT = 2*8
THRESHOLD = 0.035
ALPHA = 0
MATCH_SUPPORT = 20
SIMILARITY_COUNT = 3


class Mode(Enum):
    Distance = 0
    ArcLength = 1
    Distance_ArcLength = 2

# def normalize(v):
#     norm = np.linalg.norm(v)
#     return v if norm == 0 else v / norm

def cartesian_product(l, r):
    ret = []
    if len(l) == 0:
        return [[i] for i in r]
    elif len(r) == 0:
        return [[i] for i in l]
    for i in l:
        for j in r:
            if type(i) == list:
                k = copy.copy(i)
                k.append(j)
            else:
                k = [i, j]
            ret.append(k)
    return ret


def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.))) / (np.sqrt(2 * np.pi) * sigma)


def get_context(art, index, support, mode):
    def get_context_distance(art, index, support):
        points = art.get_vertices()
        point = art.get_vertex(index)
        n_points = len(points)
        c = np.array([np.linalg.norm(p-point) for p in points])

        # right
        if n_points < index + int(support/2) and art.is_closed:
            cr = np.append(c[index:], c[:index + int(support/2) - n_points], axis=0)
        else:
            cr = c[index : index + int(support/2)]
        # left
        if index < int(support/2) and art.is_closed:
            if index == 0:
                cl = c[index - int(support/2):]
            else:
                cl = np.append(c[index - int(support/2):], c[:index] ,axis=0)
        else:
            cl = c[index - int(support/2) : index]

        if cl.shape[0] != int(support/2) or cr.shape[0] != int(support/2):
            print("Index {} points {} Support/2 {} \ncr {}\ncl {}".format(index, n_points, int(support / 2), cr, cl))
            assert (0)

        c = np.append(cl, cr, axis=0)
        return c

        # def G(d): # Gaussian distance drop-off
        #     return gaussian(x=d, mu=0, sigma=SIGMA)
        # gc = np.vectorize(G)(c)
        # return gc

    def get_context_arclength(art, index, support):
        beziers = art.get_beziers()
        n_beziers = len(beziers)
        assert (index <= len(beziers) and 2 * n_beziers > support)
        # left
        last_left = index - int(support/2)-1 if art.is_closed else max(0, index - int(support))
        cl = []
        last = 0
        for i in range(index - 1, last_left, -1):
            last = last + beziers[i].length()
            cl.insert(0, last)

        #  right
        last_right = index + int(support/2) if art.is_closed else min(n_beziers, index + int(support/2))
        last = 0
        cr = []
        for i in range(index, last_right):
            last = last + beziers[i % n_beziers].length()
            cr.append(last)

        if len(cl) != int(support/2) or len(cr)!= int(support/2):
            print("Index: {}, Beziers: {}, Support/2: {}, \ncr: {}\ncl: {}".format(index, n_beziers, int(support / 2), cr, cl))
            assert (0)

        c = np.array(cl + cr)
        return c

        # def G(d): # Gaussian distance drop-off
        #     return gaussian(d, 0, SIGMA)
        # gc = np.vectorize(G)(c[:support])
        # return gc

    if mode == Mode.Distance:
        return get_context_distance(art, index, support)
    elif mode == Mode.ArcLength:
        return get_context_arclength(art, index, support)
    elif mode == Mode.Distance_ArcLength:
        return ALPHA * get_context_distance(art, index, support) + (1-ALPHA) * get_context_arclength(art, index, support)


def least(context, contexts, match_support_count):
    def f(c2):
        return np.linalg.norm(context - c2)

    diff = np.apply_along_axis(f, 1, contexts)
    l = np.zeros((match_support_count), dtype=int)
    for i in np.ndindex(l.shape[0]):
        l[i] = np.argmin(diff)
        diff[l[i]] = np.inf
    return l


def get_similar_points(contexts1, contexts2, matching_support, similarity_count):
    matching = np.zeros((contexts1.shape[0], matching_support), dtype=int)
    min_diff = np.zeros((contexts1.shape[0]))
    for i in np.ndindex(contexts1.shape[0]):
        matching[i] = least(contexts1[i], contexts2, matching_support)
        min_diff[i] = np.linalg.norm(contexts1[i] - contexts2[matching[i][0]])

    similar_points = []
    for i in range(similarity_count):
        j = np.argmin(min_diff)
        similar_points.append([j, matching[j]])
        min_diff[j] = np.inf
    return similar_points


def cost(matching, art1, art2, contexts1, contexts2):
    difference = sum([np.linalg.norm(contexts1[p] - contexts2[q]) for p,q in zip(matching[0], matching[1])])
    distortion = 0
    for i in range(len(matching[0])):
        for j in range(i+1, len(matching[0])):
            p, p1 = art1.get_vertex(matching[0][i]), art1.get_vertex(matching[0][j])
            q, q1 = art2.get_vertex(matching[1][i]), art2.get_vertex(matching[1][j])
            distortion = distortion +  np.linalg.norm(p - p1 - (q - q1))
    return difference + distortion


def find_opt_match(art1, art2, mode, matching_support, similarity_count):
    context1, context2 = np.zeros((art1.size() + 1, SUPPORT)), np.zeros((art2.size() + 1, SUPPORT))
    for i in range(context1.shape[0]):
        context1[i] = get_context(art1, i, support=SUPPORT, mode=mode)
    for j in range(context2.shape[0]):
        context2[j] = get_context(art2, j, support=SUPPORT, mode=mode)

    similarity_points = get_similar_points(context1, context2, matching_support, similarity_count)
    search_space = []
    similar1 = []
    for s in similarity_points:
        search_space = cartesian_product(search_space, s[1])
        similar1.append(s[0])

    print("search space : {} ".format(len(search_space)))
    curr_cost = np.inf
    min_tau = []
    for tau in search_space:
        if len(similar1) != len(tau):
            print("{}, {}".format(similar1, tau))
            assert (len(similar1) == len(tau))
        if curr_cost > cost((similar1, tau), art1, art2, context1, context2):
            min_tau = tau
    return similar1, min_tau


# arts are piecewise-bezier curves
def find_match(art1, art2, mode):
    context1, context2 = np.zeros((art1.size() + 1, SUPPORT)), np.zeros((art2.size() + 1, SUPPORT))
    for i in range(context1.shape[0]):
        context1[i] = get_context(art1, i, support=SUPPORT, mode=mode)
    for j in range(context2.shape[0]):
        context2[j] = get_context(art2, j, support=SUPPORT, mode=mode)

    # Greedy
    matching = np.zeros((context1.shape[0]))
    for i in np.ndindex(context1.shape[0]):
        c1 = context1[i]

        def f(c2):
            return np.linalg.norm(c1 - c2)

        diff = np.apply_along_axis(f, 1, context2)
        matching[i] = np.argmin(diff)
    return matching


def plot_match(art1, art2, matching, draw):
    for i, j in zip(matching[0], matching[1]):
        p = art1.get_vertex(i)
        q = art2.get_vertex(j)
        print("Matching similarity: {}<->{}".format(i, j))
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
        b11 = art1.get_bezier(i - 1)
        b2 = art2.get_bezier(j)
        b21 = art2.get_bezier(j - 1)
        b1.set_color((255, 0, 0))
        b11.set_color((255, 0, 0))
        b2.set_color((0, 0, 255))
        b21.set_color((0, 0, 255))

        draw.add_art(b1)
        draw.add_art(b11)
        draw.add_art(b2)
        draw.add_art(b21)


def transform(art1, art2, matching, threshold, draw, mode):
    for i in range(len(matching)):
        j = int(matching[i])
        p = art1.get_vertex(i)
        q = art2.get_vertex(j)
        c1 = get_context(art1, i, support=SUPPORT, mode=mode)
        c2 = get_context(art2, j, support=SUPPORT, mode=mode)
        similarity = np.linalg.norm(c1 - c2)
        if similarity <= threshold:
            # print("Matching similarity: {}\n\t c1:{},\n\t c2:{}".format(similarity,c1, c2))
            print("Matching similarity: {}".format(similarity))
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
    t1.apply(Art.Translate([-15, -12]))
    t2 = ArtCollection.gorilla
    t2.apply(Art.Scale(5))
    t2.apply(Art.Translate([2, -1]))
    return t1, t2


def elephant_giraffe():
    t1 = ArtCollection.elephant
    t1.apply(Art.Scale(2))
    t1.apply(Art.Translate([-6, 3]))

    t2 = copy.copy(ArtCollection.elephant)
    # t2.apply(Art.Scale(3))
    # t2.apply(Art.Translate([2, -1]))
    return t1, t2


def main():
    d = Art.Draw()

    t1, t2 = lion_gorilla()

    mode = Mode.ArcLength
    matching = find_opt_match(art1=t1, art2=t2, mode=mode, matching_support=MATCH_SUPPORT, similarity_count=SIMILARITY_COUNT)
    plot_match(art1=t1, art2=t2, matching=matching, draw=d)
    # transform(art1=t1, art2=t2, matching=matching, threshold=THRESHOLD, draw=d, mode=mode)

    d.add_art(t1)
    d.add_art(t2)
    d.draw()

main()
