import numpy as np
import Art
import testLevel4 as ts
import MatchMaker as MM
import D3Plot as d3


def get_mid(p1, p2, frac):
    return (1-frac)*p1 + frac*p2


def equal_spacing(poly1, poly2, debug=False):
    dist1, dist2 = 0, 0

    N, M = len(poly1), len(poly2)
    new_poly1, new_poly2 = [poly1[0]], [poly2[0]]
    ind1, ind2 = 1, 1
    while ind1 < len(poly1) and ind2 < len(poly2):
        last1, last2 = new_poly1[-1], new_poly2[-1]
        next1 = dist1 + np.linalg.norm(poly1[ind1] - last1)
        next2 = dist2 + np.linalg.norm(poly2[ind2] - last2)
        if next1 < next2:
            new_poly1.append(poly1[ind1])
            ind1 = ind1 + 1
            dist1 = next1

            frac = np.linalg.norm(last1 - new_poly1[-1]) / np.linalg.norm(new_poly2[-1] - poly2[ind2])
            mid = get_mid(new_poly2[-1], poly2[ind2], frac)
            new_poly2.append(mid)
            dist2 = dist2 + np.linalg.norm(new_poly2[-2] - mid)

        elif next2 < next1:
            new_poly2.append(poly2[ind2])
            ind2 = ind2 + 1
            dist2 = next2

            frac = np.linalg.norm(last2 - new_poly2[-1]) / np.linalg.norm(new_poly1[-1] - poly1[ind1])
            mid = get_mid(new_poly1[-1], poly1[ind1], frac)
            new_poly1.append(mid)
            dist1 = dist1 + np.linalg.norm(new_poly1[-2] - mid)

        else:
            new_poly1.append(poly1[ind1])
            new_poly2.append(poly2[ind2])
            ind1 = ind1 + 1
            ind2 = ind2 + 1
        if debug:
            print("Dist1 : {:.4f}, Dist2: {:.4f}".format(dist1, dist2))
    if debug:
        print("{} == {}".format(len(new_poly1), len(new_poly2)))
    return np.array(new_poly1), np.array(new_poly2)


def remove_translation(poly):
    art = Art.Polygon(poly)
    art.apply(Art.Translate(-art.get_centroid()))
    return art.get_vertices()


def remove_scale(poly):
    s = np.sqrt(np.mean(np.linalg.norm(poly, axis=1)))
    assert s > 0
    return poly / s


def align(poly1, poly2):
    p1, p2 = equal_spacing(poly1, poly2)
    p1, p2 = remove_scale(remove_translation(p1)), remove_scale(remove_translation(p2))

    x1, y1 = np.array([p[0] for p in p1]), np.array([p[1] for p in p1])
    x2, y2 = np.array([p[0] for p in p2]), np.array([p[1] for p in p2])
    theta = np.arctan(np.sum(y1 * x2 - y2 * x1) / np.sum(y1 * x2 + y2 * x1))

    x = x2 * np.cos(theta) - y2 * np.sin(theta)
    y = x2 * np.sin(theta) + y2 * np.cos(theta)

    p3 = np.array([[x[i], y[i]] for i in range(len(x))])
    return p1, p3


def distance(q1, q2):
    p1, p2 = equal_spacing(q1, q2)
    p1, p2 = remove_scale(remove_translation(p1)), remove_scale(remove_translation(p2))

    x1, y1 = np.array([p[0] for p in p1]), np.array([p[1] for p in p1])
    x2, y2 = np.array([p[0] for p in p2]), np.array([p[1] for p in p2])
    theta = np.arctan(np.sum(y1*x2 - y2*x1) / np.sum(y1*x2 + y2*x1))

    x = x2 * np.cos(theta) - y2 * np.sin(theta)
    y = x2 * np.sin(theta) + y2 * np.cos(theta)

    return np.sqrt( np.sum ((x1 - x)**2 + (y1 - y)**2) )


if __name__ == '__main__':
    arts = ts.get_test(0)
    a0, a1 = arts[0], arts[1]

    mm = MM.MatchMaker(100)
    id1, id2 = mm.add_art(a0), mm.add_art(a1)

    p1, p2 = mm.get_poly(id1), mm.get_poly(id2)
    q1, q2 = equal_spacing(p1, p2)
    k1, k2 = align(p1, p2)

    d3.draw_polys([Art.Polygon(p1), Art.Polygon(k1), Art.Polygon(p2), Art.Polygon(k2)], [])
    print (distance(q1, q2))
