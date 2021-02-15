import numpy as np
import FunctionTransform as ft
import FunctionSimilarity as fs
import testsLevel1 as testsuite
import Art


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


class ArtBox:
    def __init__(self, art, importance_angle):
        self.art = art
        self.importance_angle = np.deg2rad(importance_angle)
        self.poly = None
        self.fun = None
        self.distance = None
        self.art_poly_map = []
        self.poly_squint_map = []

    def index_to_point(self, index):
        ind, ind_t = index
        assert self.fun is not None
        ind_next = (ind+1)%len(self.poly_squint_map)

        poly_ind = self.poly_squint_map[ind]
        poly_ind_next = self.poly_squint_map[ind_next]

        def get_point(polynomial_index, t):
            art_ind, ex = self.art_poly_map[polynomial_index]
            bezier = self.art.get_bezier(art_ind)
            if ex == 0:
                point = bezier.point_at(t)
            else:
                ex_ts = bezier.get_extremes_t()
                t1 = ex_ts[ex - 1]
                t2 = ex_ts[ex] if ex < len(ex_ts) else 1
                n_t = (1 - t) * t1 + t * t2
                point = bezier.point_at(n_t)
            return point

        if poly_ind_next == poly_ind + 1: # consecutive
            b_point = get_point(poly_ind, ind_t)
        else:
            points = []
            count = poly_ind
            while count < poly_ind_next:
                points.append(get_point(count, 0))
                count = count + 1
            l = [0.0]
            for i in range(1, len(points)):
                l.append(np.linalg.norm(points[i] - points[i-1]))

            l = [a/l[-1] for a in l]
            index = 0
            while index < len(l):
                if l[index] >= ind_t:
                    break
                index = index + 1
            


        return b_point

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

    def squint(self):
        if self.importance_angle == 0.0:
            self.poly_squint_map = list(range(len(self.poly)))
            return self.poly

        new_polygon = []
        n = len(self.poly)
        for i in range(n):
            p0 = self.poly[(i + 1) % n]
            p1 = self.poly[i]
            p2 = self.poly[i - 1]

            if np.abs(ft.turn_angle(p0, p1, p2)) >= self.importance_angle:
                new_polygon.append(p1)
                self.poly_squint_map.append(i)

        return np.array(new_polygon)

    def to_function(self):
        assert (len(self.poly) > 2)
        radian, length = np.zeros((len(self.poly))), np.zeros((len(self.poly)))
        radian[0] = ft.turn_angle(self.poly[-1], self.poly[0], self.poly[1])
        # length[0] = 0.0

        for i in range(len(self.poly)):
            p0 = self.poly[i - 1]
            p1 = self.poly[i % len(self.poly)]
            p2 = self.poly[(i + 1) % len(self.poly)]
            radian[i] = ft.turn_angle(p0, p1, p2)
            length[i] = length[i-1] + np.linalg.norm(p1 - p2)

        return radian, length

    def get_function(self):
        assert self.art is not None
        if self.poly is None:
            self.poly = self.piecewise_bezier_to_polygon()
            self.poly = self.squint()
            a1, self.distance = self.to_function()
            d1 = self.distance / self.distance[-1]
            self.fun = a1, d1
        return self.fun

    def get_poly(self):
        if self.poly is None:
            self.clear()
            self.get_function()
            print(self.art_poly_map)
        return self.poly

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
    def __init__(self, importace_angle):
        self.art_boxes = []
        self.importance_angle = importace_angle

    def add_art(self, art):
        self.art_boxes.append(ArtBox(art, self.importance_angle))
        return len(self.art_boxes) - 1

    def get_function(self, id):
        assert id < len(self.art_boxes)
        return self.art_boxes[id].get_function()

    def get_poly(self, id):
        assert id < len(self.art_boxes)
        return self.art_boxes[id].get_poly()

    def get_distance(self, id):
        assert id < len(self.art_boxes)
        return self.art_boxes[id].get_distance()

    def get_match(self, id1, id2, partial_match):
        assert id1 <= len(self.art_boxes) and id2 <= len(self.art_boxes)

        a, b = self.get_distance(id1), self.get_distance(id2)
        partial_match.sort(key=lambda x: x[0])

        first_a, first_b = partial_match[0][0], partial_match[0][1]
        match = [[(first_a, 0), (first_b, 0)]]
        for i in range(1, len(partial_match)):
            def in_between(low_a, high_a, low_b, high_b):
                assert a[high_a] > a[low_a]

                def dist_a(i):
                    return self.art_boxes[id1].dist(low_a, i) * scale_a

                def dist_b(i):
                    return self.art_boxes[id2].dist(low_b, i) * scale_b

                scale_a = 1/self.art_boxes[id1].dist(low_a, high_a)
                scale_b = 1/self.art_boxes[id2].dist(low_b, high_b)

                ind1, ind2 = low_a, low_b
                mat = []
                print ("L:{}, {}  H:{}, {}".format(low_a, low_b, high_a, high_b))
                while ind1 <= high_a:
                    if (ind1==high_a and ind2==high_b) or (dist_a(ind1)==dist_b(ind2)):
                        if ind1 != low_a and ind2 != low_b:
                            print (" = {} ~ {}".format(ind1, ind2))
                            mat.append([(ind1, 0), (ind2, 0)])
                        ind1 = ind1 + 1
                        ind2 = (ind2 + 1) % len(b)
                    else:
                        if dist_a(ind1) < dist_b(ind2):
                            frac = (dist_a(ind1) - dist_b(ind2-1)) / (dist_b(ind2) - dist_b(ind2-1))
                            mat.append([(ind1, 0), (ind2-1, frac)])
                            print(" < {},{:.2f} ~ {},{:.2f}".format(ind1, 0, ind2-1, frac))
                            ind1 = ind1 + 1
                        else:
                            frac = (dist_b(ind2) - dist_a(ind1-1)) / (dist_a(ind1) - dist_a(ind1-1))
                            mat.append([(ind1-1, frac), (ind2, 0)])
                            print(" > {},{:.2f} ~ {},{:.2f}".format(ind1-1, frac, ind2, 0))
                            ind2 = (ind2 + 1) % len(b)
                return mat
            match = match + in_between(partial_match[i-1][0], partial_match[i][0], partial_match[i-1][1], partial_match[i][1])
        return match

    def get_art(self, id):
        assert id < len(self.art_boxes)
        return self.art_boxes[id].get_art()

    def get_similarity_matrix(self, id1, id2, diff):
        assert id1 <= len(self.art_boxes) and id2 <= len(self.art_boxes)
        f1, f2 = self.art_boxes[id1].get_function(), self.art_boxes[id2].get_function()
        mat = np.zeros((len(f1[0]), len(f2[0])))
        for ij in np.ndindex(mat.shape):
            i, j = ij
            f1_i, f2_j = roll(f1, i), roll(f2, j)
            mat[ij] = diff(f1_i, f2_j)
        return mat

    def get_least_diff(self, id1, id2, diff, best):
        mat = self.get_similarity_matrix(id1, id2, diff)
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
    a1, a2 = testsuite.get_test(6)
    mm = MatchMaker(importace_angle=10)
    id1 = mm.add_art(a1)
    id2 = mm.add_art(a2)
    matched = mm.get_least_diff(id1, id2, fs.diff, 3)

    d = Art.Draw()
    poly1 = Art.Polygon(mm.get_poly(id1))
    poly2 = Art.Polygon(mm.get_poly(id2))
    d.add_art(poly1), d.add_art(poly2)

    mask = set()
    for p,q in matched:
        print (p,q)
        c1, c2 = mm.index_to_point(id1, (p,0)), mm.index_to_point(id2, (q,0))
        ids = Art.draw_curve(c1, c2, d)
        d.draw(mask)
        mask = mask.union(ids)

    all_matched = mm.get_match(id1, id2, partial_match=matched)


if __name__ == '__main__':
    main()