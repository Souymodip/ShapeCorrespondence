import numpy as np
import FunctionTransform as ft
import FunctionSimilarity as fs
import testLevel4 as testsuite
import Art
import D3Plot as d3


ERROR = 1.e-6
def is_diff(p1, p2):
    return np.linalg.norm(p1 - p2) > ERROR


def roll(f, index):
    r, l = f
    if index == 0:
        return r, l
    assert index < r.shape[0] and r.shape == l.shape
    n_r, n_l = np.zeros(r.shape), np.zeros(l.shape)
    for i in range(r.shape[0]):
        if i + index < r.shape[0]:
            n_l[i] = l[i + index] - l[index - 1]
            n_r[i] = r[i + index]
        else:
            n_l[i] = n_l[l.shape[0] - 1 - index] + l[i + index - l.shape[0]]
            n_r[i] = r[i + index - l.shape[0]]
    return n_r, n_l


def length(poly):
    sum = 0
    for i in range(1, len(poly)):
        sum = sum + np.linalg.norm(poly[i-1] - poly[i])
    return sum


def importance_corner(p0, p1, p2):
    a, b = p1 - p0, p2 - p1

    def angle(x, y):
        if x == 0:
            return np.pi / 2
        else:
            theta = np.arctan(np.abs(y) / np.abs(x))
            if y >= 0 and x >= 0:
                return theta
            if y >= 0 and x < 0:
                return np.pi - theta
            if y < 0 and x < 0:
                return np.pi + theta
            else:
                return 2 * np.pi - theta

    if np.linalg.norm(a) * np.linalg.norm(b) == 0:
        return 0.0
    else:
        a_mod, b_mod = np.linalg.norm(a), np.linalg.norm(b)
        a, b = a / a_mod, b / b_mod
        atheta, btheta = angle(a[0], a[1]), angle(b[0], b[1])
        t = btheta - atheta
        t = t if t >= 0 else 2 * np.pi + t
        return t * (a_mod + b_mod)


def importance_sample(poly, importance_percentile):
    assert poly is not None
    size = len(poly)
    sample = []
    for i in range(size):
        p0 = poly[i-1]
        p1 = poly[i]
        p2 = poly[(i+1)%size]
        sample.append(np.abs(importance_corner(p0, p1, p2)))

    p_val = np.percentile(sample, importance_percentile)
    new_poly = [poly[i] for i in range(size) if sample[i] <= p_val]
    return np.array(new_poly)


def poly_to_func(poly):
    assert (len(poly) > 2)
    l = len(poly) + 1
    radian, length = np.zeros((l)), np.zeros((l))
    radian[0] = ft.turn_angle(poly[-1], poly[0], poly[1])
    length[0] = 0.0
    for i in range(1, l - 1):
        p0 = poly[i - 1]
        p1 = poly[i]
        p2 = poly[(i + 1) % len(poly)]
        radian[i] = ft.turn_angle(p0, p1, p2)
        length[i] = length[i - 1] + np.linalg.norm(p0 - p1)

    length[-1] = length[l - 2] + np.linalg.norm(poly[0] - poly[-1])
    radian[-1] = radian[0]
    return radian, length



class ArtBox:
    def __init__(self, art, importance_percentile):
        self.art = art
        self.importance_percentile = importance_percentile
        self.poly = None
        self.fun = None
        self.distance = None
        self.art_poly_map = []
        self.poly_squint_map = []

    def piecewise_bezier_to_polygon(self):
        polygon = []
        beziers = self.art.get_beziers()

        for i in range(len(beziers)):
            b = beziers[i]
            if i == 0 or ft.is_diff(polygon[-1], b.controls[0]):
                polygon.append(b.controls[0])
                self.art_poly_map.append((i, 0))
            ex = b.get_extremes()
            for j in range(len(ex)):
                e = ex[j]
                if is_diff(polygon[-1], e):
                    polygon.append(e)
                    self.art_poly_map.append((i, j+1))
        return np.array(polygon)

    def importance_sample(self):
        assert self.poly is not None
        size = len(self.poly)
        sample = []
        for i in range(size):
            p0 = self.poly[i-1]
            p1 = self.poly[i]
            p2 = self.poly[(i+1)%size]
            sample.append(np.abs(importance_corner(p0, p1, p2)))

        p_val = np.percentile(sample, self.importance_percentile)
        new_poly = [self.poly[i] for i in range(size) if sample[i] <= p_val]
        return np.array(new_poly)

    def to_func_normal(self):
        if self.poly is None:
            self.poly = self.piecewise_bezier_to_polygon()
            self.poly = self.importance_sample()

        poly = self.poly if is_diff(self.poly[0], self.poly[-1]) else self.poly[:-1]
        assert (len(poly) > 2)
        l = len(poly) + 1
        radian, length = np.zeros((l)), np.zeros((l))
        radian[0] = ft.turn_angle(poly[-1], poly[0], poly[1])
        length[0] = 0.0
        for i in range(1, l-1):
            p0 = poly[i - 1]
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            radian[i] = ft.turn_angle(p0, p1, p2)
            length[i] = length[i - 1] + np.linalg.norm(p0 - p1)

        length[-1] = length[l-2] + np.linalg.norm(poly[0] - poly[-1])
        radian[-1] = radian[0]
        self.distance = length

        return radian, length/ length[-1]

    def get_function(self):
        assert self.art is not None
        if self.poly is None:
            a1, d1 = self.to_func_normal()
            self.fun = a1, d1
        return self.fun

    def get_poly(self):
        if self.poly is None:
            self.clear()
            self.get_function()
        return self.poly

    def get_polygon(self):
        return Art.Polygon(self.get_poly())

    def get_distance(self):
        if self.distance is None:
            self.clear()
            self.get_function()
        return self.distance

    def get_art(self):
        return self.art

    def dist(self, i, j):
        assert self.distance is not None
        if i == j :
            return 0.0
        if i == 0:
            return self.distance[j-1]
        if i <= j :
            return self.distance[j-1] - self.distance[i-1]
        else:
            if j == 0:
                return self.distance[-1] - self.distance[i-1]
            return self.distance[-1] - (self.distance[i-1] - self.distance[j-1])

    def clear(self):
        self.poly = None
        self.fun = None
        self.distance = None
        self.art_poly_map.clear()
        self.poly_squint_map.clear()


class MatchMaker:
    def __init__(self, importance_percentile, normalize=True):
        self.art_boxes = []
        self.importance_percentile = importance_percentile
        self.normalize = normalize

    def add_art(self, art):
        self.art_boxes.append(ArtBox(art, self.importance_percentile))
        return len(self.art_boxes) - 1

    def get_function(self, id, normalize=True):
        assert id < len(self.art_boxes)
        if normalize:
            return self.art_boxes[id].get_function()
        else:
            y, x = self.art_boxes[id].get_function()
            return y, x * self.art_boxes[id].distance

    def get_poly(self, id):
        assert id < len(self.art_boxes)
        return self.art_boxes[id].get_poly()

    def get_polygon(self, id):
        assert id < len(self.art_boxes)
        return self.art_boxes[id].get_polygon()

    def get_distance(self, id):
        assert id < len(self.art_boxes)
        return self.art_boxes[id].get_distance()

    def get_match(self, id1, id2, partial_match):
        assert id1 <= len(self.art_boxes) and id2 <= len(self.art_boxes)

        a, b = self.get_distance(id1), self.get_distance(id2)
        partial_match.sort(key=lambda x: x[0])
        partial_match.append(partial_match[0])

        first_a, first_b = partial_match[0][0], partial_match[0][1]
        match = [[(first_a, 0), (first_b, 0)]]
        for i in range(1, len(partial_match)):
            def in_between(low_a, high_a, low_b, high_b):
                # assert a[high_a] > a[low_a]

                def dist_a(i):
                    return self.art_boxes[id1].dist(low_a, i) * scale_a

                def dist_b(i):
                    return self.art_boxes[id2].dist(low_b, i) * scale_b

                scale_a = 1/self.art_boxes[id1].dist(low_a, high_a)
                scale_b = 1/self.art_boxes[id2].dist(low_b, high_b)

                ind1, ind2 = low_a, low_b
                mat = []
                print ("L:{}, {}  H:{}, {}".format(low_a, low_b, high_a, high_b))
                while ind1 != high_a and ind2 != high_b:
                    if (ind1==high_a and ind2==high_b) or (dist_a(ind1)==dist_b(ind2)):
                        if ind1 != low_a and ind2 != low_b:
                            # print (" = {} ~ {}".format(ind1, ind2))
                            mat.append([(ind1, 0), (ind2, 0)])
                        ind1 = (ind1 + 1) % len(a)
                        ind2 = (ind2 + 1) % len(b)
                    else:
                        if dist_a(ind1) < dist_b(ind2):
                            ind2_1 = (ind2 - 1) % len(b)
                            frac = (dist_a(ind1) - dist_b(ind2_1)) / (dist_b(ind2) - dist_b(ind2_1))
                            mat.append([(ind1, 0), (ind2_1, frac)])
                            # print(" < {},{:.2f} ~ {},{:.2f}".format(ind1, 0, ind2_1, frac))
                            ind1 = (ind1 + 1) % len(a)
                        else:
                            ind1_1 = (ind1 - 1) % len(a)
                            frac = (dist_b(ind2) - dist_a(ind1_1)) / (dist_a(ind1) - dist_a(ind1_1))
                            mat.append([(ind1_1, frac), (ind2, 0)])
                            # print(" > {},{:.2f} ~ {},{:.2f}".format(ind1_1, frac, ind2, 0))
                            ind2 = (ind2 + 1) % len(b)
                return mat
            match = match + in_between(partial_match[i-1][0], partial_match[i][0], partial_match[i-1][1], partial_match[i][1])
        return match

    def get_art(self, id):
        assert id < len(self.art_boxes)
        return self.art_boxes[id].get_art()

    def get_similarity_matrix(self, id1, id2, diff, debug = False):
        assert id1 <= len(self.art_boxes) and id2 <= len(self.art_boxes)
        f1, f2 = self.art_boxes[id1].get_function(), self.art_boxes[id2].get_function()
        mat = np.zeros((len(f1[0]), len(f2[0])))

        count = 0
        total = mat.shape[0] * mat.shape[1]
        if debug:
            print ("Iteration : {}".format(total))
        for ij in np.ndindex(mat.shape):
            i, j = ij
            f1_i, f2_j = roll(f1, i), roll(f2, j)
            mat[ij] = diff(f1_i, f2_j)

            if debug and count % int(total / 100) == 0:
                print("Complete {:.0f}% ...".format(count * 100 / total))
            count = count + 1

        return mat

    def get_least_diff_index_lit(self, id1, id2, diff, debug=False):
        assert id1 <= len(self.art_boxes) and id2 <= len(self.art_boxes)
        f1, f2 = self.art_boxes[id1].get_function(), self.art_boxes[id2].get_function()
        mat = np.zeros((len(f2[0])))
        total = mat.shape[0]
        count = 0
        f1 = roll(f1, 0)
        if debug:
            print ("Iteration : {}".format(total))
        for j in range(total):
            f2_j = roll(f2, j)
            mat[j] = diff(f1, f2_j)
            if debug and count % int(total / 100) == 0:
                print("Complete {:.0f}% ...".format(count * 100 / total))
            count = count + 1

        least = np.argmin(mat)
        if debug:
            ft.draw_graph([f1, roll(f2, least)])
        return [(0, least)]

    def get_least_diff_index(self, id1, id2, diff, best, debug = False):
        mat = self.get_similarity_matrix(id1, id2, diff, True)
        matched = set()

        def get_least(mat):
            least = np.inf, -1, -1
            for ij in np.ndindex(mat.shape):
                if ij not in matched and least[0] >= mat[ij]:
                    least = mat[ij], ij[0], ij[1]
            return least[1], least[2]
        for i in range(best):
            matched.add(get_least(mat))
        return list(matched)

    def index_to_point(self, id, index):
        assert id < len(self.art_boxes)
        return self.art_boxes[id].index_to_point(index)

def main():
    arts = testsuite.get_test(0)
    a1, a2 = arts[0], arts[2]
    mm = MatchMaker(100)
    print ("Adding art1.")
    id1 = mm.add_art(a1)
    print("Adding art2.")
    id2 = mm.add_art(a2)
    matched = mm.get_least_diff_index(id1, id2, fs.diff, 2)
    matched_points = [[mm.index_to_point(id1, (p,0)), mm.index_to_point(id2, (q, 0))] for p,q in matched]

    poly1 = Art.Polygon(mm.get_poly(id1))
    poly2 = Art.Polygon(mm.get_poly(id2))

    # all_matched = mm.get_match(id1, id2, partial_match=matched)
    # all_matched_points = [[mm.index_to_point(id1, m[0]), mm.index_to_point(id2, m[1])] for m in all_matched]

    d3.draw_polys(poly1, poly2, matched_points)

if __name__ == '__main__':
    main()
