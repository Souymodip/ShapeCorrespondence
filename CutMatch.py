import numpy as np
import copy as cp
import FunctionTransform as ft
import FunctionSimilarity as fs
import testLevel4 as ts
import MatchMaker as MM
import Art
import D3Plot as D3
import DFT as dft
import Procrutes
from Helpers import timer
from Helpers import bcolors
import Helpers
import Autoencoder as ae


STRIDE=5
CUT_LENGTH=30
INITIAL_RAND_SEARCH=10


def double(f):
    y, x = f
    return np.append(y, y[:-1]), np.append(x, np.array(x[1:-1]) + x[-1])


def get_cut(poly, x_val, length, poly_2=None):
    assert x_val >= 0
    index = 0
    curr_x = 0
    poly_2 = np.append(poly, poly, axis=0) if poly_2 is None else poly_2

    while index < len(poly) and curr_x < x_val:
        curr_x = curr_x + np.linalg.norm(poly[index - 1] - poly[index])
        index = index + 1

    if index == len(poly): index = 0

    if np.linalg.norm(poly[index - 1] - poly[index]) < 1e-6:
        ratio = 0
    else:
        ratio = (curr_x - x_val) / np.linalg.norm(poly[index - 1] - poly[index])
    first = poly[index] * (1 - ratio) + poly[index - 1] * (ratio)
    new_poly = [first]

    total = 0
    add_index = index
    if not ft.is_diff(new_poly[0], poly[add_index]):
        add_index = add_index + 1

    while total <= length:
        dis = np.linalg.norm(new_poly[-1] - poly_2[add_index])
        total = total + dis
        if total < length:
            new_poly.append(poly_2[add_index])
        else:
            ratio = (total - length) / dis
            p = poly_2[add_index] * (1 - ratio) + poly_2[add_index - 1] * ratio
            new_poly.append(p)

        add_index = add_index + 1
    return Cut(np.array(new_poly), index, length, x_val)


class Cut:
    def __init__(self, poly, original_index, length, x_val):
        self.poly = poly
        self.index = original_index
        self.length = length
        self.x_val = x_val

    def polygon(self):
        p = Art.Polygon(self.poly)
        p.isClosed = False
        return p

    def get_bounds(self):
        xs = [p[0] for p in self.poly]
        ys = [p[1] for p in self.poly]
        return np.array([[np.min(xs), np.min(ys)], [np.max(xs), np.max(ys)]])

    def __str__(self):
        s = "Cut [\n" \
            "\tindex:  {}\n" \
            "\tlength: {}\n" \
            "]".format(self.index, self.length)
        return s


class Cuts:
    def __init__(self, poly):
        self.poly = np.array(poly)
        self.poly_2 = np.append(self.poly, self.poly, axis=0)
        self.total_length = MM.length(self.poly)

    def get_cut_at(self, x_val, length):
        return get_cut(self.poly, x_val, length, self.poly_2)


def get_neighbourhood (cuts2, x_val2, cut_length):
    overlap_fraction = 0.5
    righ_overlap = overlap_fraction * cut_length

    if x_val2 > (1 + overlap_fraction) * cut_length:
        l = x_val2 - (1 + overlap_fraction) * cut_length
        intervals = [(l, x_val2 - righ_overlap)]
    else:
        if x_val2 - righ_overlap >= 0 :
            intervals = [(0, x_val2 - righ_overlap), \
                         (cuts2.total_length - ((1 + overlap_fraction) * cut_length - x_val2), cuts2.total_length)]
        else:
            intervals = [(0, x_val2 - righ_overlap), \
                         (cuts2.total_length - ((1 + overlap_fraction) * cut_length - x_val2), \
                          cuts2.total_length - (righ_overlap - x_val2))]
    return intervals


def match_internal(cut1, cut2):
    assert cut1.length == cut2.length
    match = [(cut1.poly[0], cut2.poly[0])]
    distance1, distance2 = [0], [0]
    for i in range(1, len(cut1.poly)):
        distance1.append(distance1[-1] + np.linalg.norm(cut1.poly[i-1]- cut1.poly[i]))
    for i in range(1, len(cut2.poly)):
        distance2.append(distance2[-1] + np.linalg.norm(cut2.poly[i - 1] - cut2.poly[i]))

    def find_closet(i1, i2):
        if distance2[i2] <= distance1[i1]:
            diff = distance2[i2] - distance1[i1]
            curr_i2 = i2
            while diff < 0 and curr_i2 < len(distance2):
                diff = distance2[curr_i2] - distance1[i1]
                curr_i2 = curr_i2 + 1

            if curr_i2 >= len(distance2):
                return curr_i2 - 1
            if diff == 0:
                return curr_i2
            else:
                return curr_i2 if np.abs(distance2[curr_i2] - distance1[i1]) < np.abs(distance2[curr_i2-1] - distance1[i1]) else curr_i2 -1
        else:
            return i2
    j = 0
    for i in range(1, len(distance1)):
        j = find_closet(i, j)
        match.append((cut1.poly[i], cut2.poly[j]))

    match.append((cut1.poly[-1], cut2.poly[-1]))

    return match


def get_relative(distance, stride, cut_length):
    assert stride > 0 and cut_length > 0
    return np.ceil(stride * distance / 155), cut_length * distance / 155


class Cut_Match:
    def __init__(self, poly1, poly2, diff, stride, cut_length):
        assert stride > 0 and cut_length > 0
        count = min(len(poly1), len(poly2))
        poly1, poly2 = dft.shrink(poly1, count), dft.shrink(poly2, count)
        # poly1, poly2 = Procrutes.equal_spacing(poly1, poly2)
        self.cuts1, self.cuts2 = Cuts(poly1), Cuts(poly2)

        self.stride = stride
        self.cut_length = cut_length
        self.diff = diff

    def exact_match(self, cut):
        return self.neighbour_match(cut,[(0, MM.length(self.cuts2.poly))])

    def neighbour_match(self, cut, intervals):
        print("\tNeighbourhood : {}".format(intervals))
        p1 = cut.poly

        min_x, min_val = 0, np.inf
        for l, h in intervals:
            diff = []
            for x in np.arange(l, h, self.stride):
                p2 = self.cuts2.get_cut_at(x, self.cut_length).poly
                d = self.diff(p1, p2) # np.linalg.norm(dft.diff_poly(p1, p2, frac=1.0)) #
                diff.append(d)
            if len(diff) > 0:
                min_index = np.argmin(diff)
                if min_val > diff[min_index]:
                    min_x, min_val = l + self.stride * min_index, diff[min_index]

        print("\t\tCut1 : {:.2f}, Cut2: {:.2f}, Diff := {:.3f}".format(cut.x_val, min_x, min_val))
        return self.cuts2.get_cut_at(min_x, self.cut_length), min_val

    def match_next(self, cut1, cut2):
        cut_length = cut1.length
        x_left = cut1.x_val - cut_length if cut1.x_val > cut_length else self.cuts1.total_length - (
                cut_length - cut1.x_val)
        intervals = get_neighbourhood(self.cuts2, cut2.x_val, cut_length)
        left1 = self.cuts1.get_cut_at(x_left, cut_length)
        left2, val = self.neighbour_match(left1, intervals)
        return left1, left2, val

    @timer
    def rand_initial(self, times):
        min_cut1, min_cut2, min_val = None, None, np.inf
        step = 2
        rs = np.random.choice(int(self.cuts1.total_length/step), times, replace=True)
        for r in rs:
            r_cut1 = self.cuts1.get_cut_at(r*step, length=self.cut_length)
            ex_cut2, diff = self.exact_match(r_cut1)
            if min_val > diff:
                min_cut1, min_cut2, min_val = r_cut1, ex_cut2, diff
        print("\tInitializing from Cut1:{}, Cut2:{}".format(min_cut1.x_val, min_cut2.x_val))
        return min_cut1, min_cut2

    def cut_match(self):
        # Random initialization
        print("Inital Random Match ...")
        r_cut1, ex_cut2 = self.rand_initial(INITIAL_RAND_SEARCH)

        # match adjacent
        print("Neighbourhood Match ...")
        nexts1, nexts2 = [r_cut1], [ex_cut2]
        total_val = 0
        for i in range (int(MM.length(self.cuts1.poly)/self.cut_length)):
            next1, next2, val = self.match_next(nexts1[-1], nexts2[-1])
            nexts1.append(next1), nexts2.append(next2)
            total_val = total_val + val



        polys = []
        for i in range(len(nexts1)):
            polys.append(nexts1[i].polygon()), polys.append(nexts2[i].polygon())

        D3.draw_poly_index(polys)

        for i in range(int(len(polys)/2)):
            match0 = match_internal(nexts1[i], nexts2[i])
            D3.draw_polys([polys[2*i], polys[2*i + 1]], match0)

    def test(self):
        x1, x2 = 66, 88.00

        cut1 = self.cuts1.get_cut_at(x1, self.cut_length)
        # cut2 = cuts2.get_cut_at(x2, cut_length)
        cut2, _ = self.exact_match(cut1)

        p1, p2 = cut1.poly, cut2.poly

        d = self.diff(p1, p2)
        print("Cut1 : {}, Cut2: {}, Diff := {}".format(x1, x2, d))
        match0 = match_internal(cut1, cut2)

        D3.draw_polys([cut1.polygon(), cut2.polygon()], match0)


def draw_cut_match(cuts1, cuts2, cut_pairs):
    d = Art.Draw()
    d.add_art(Art.Polygon(cuts1.poly)), d.add_art(Art.Polygon(cuts2.poly))
    for l, r in cut_pairs:
        box1,box2 = l.get_bounds(), r.get_bounds()
        Art.draw_bbox(d, box1, (100, 100, 250))
        Art.draw_bbox(d, box2, (100, 100, 250))
        c_i, c_j = np.mean(box1, axis=0), np.mean(box2, axis=0)
        mid1, mid2 = (c_i / 3 + c_j * 2 / 3), (c_i * 2 / 3 + c_j / 3)
        mid1[1], mid2[1] = mid1[1] + 5, mid2[1] + 5
        b = Art.Bezier([c_i, mid2, mid1, c_j])
        b.set_color((255, 0, 255))
        d.add_art(b)
    d.draw()


def get_diff_method(index):
    def func_procrustes(p1, p2):
        return Procrutes.distance(p1, p2)

    def func_dft(p1, p2):
        def abs(p):
            return np.sqrt(p[0] ** 2 + p[1] ** 2)
        # p1, p2 = Procrutes.align(p1, p2)
        return abs(dft.diff_poly(p1, p2, frac=1.0))

    diff_func = None
    func_name = ""
    if index == 0:
        diff_func = func_dft
        func_name = "Dft low pass diff"
    elif index == 1:
        diff_func = func_procrustes
        func_name = "Procrustes Diff"
    elif index == 2:
        diff_func = ae.get_Neural_encoder_diff()
        if diff_func is not None:
            func_name = "Neural Network Encoder"
        else:
            print(bcolors.WARNING + "No encoder found. Falling to option 0." + bcolors.ENDC)
            diff_func = func_dft
            func_name = "Dft low pass diff"

    print(bcolors.UNDERLINE + "Matching Option {} is chosen : {}".format(index, func_name) + bcolors.ENDC)
    return diff_func


def execute(art1, art2, option):
    mm = MM.MatchMaker(importance_percentile=100)
    id1, id2 = mm.add_art(art1), mm.add_art(art2)
    p1, p2 = mm.get_poly(id1), mm.get_poly(id2)
    stride, cut_length = get_relative(mm.get_length(id1), STRIDE, CUT_LENGTH)

    cm = Cut_Match(p1, p2, get_diff_method(option), stride=stride, cut_length=cut_length)
    cm.cut_match()


def main():
    mm = MM.MatchMaker(importance_percentile=100)
    arts = ts.get_test(0)
    id1, id2 = mm.add_art(arts[0]), mm.add_art(arts[5])
    p1, p2 = mm.get_poly(id1), mm.get_poly(id2)
    stride, cut_length = get_relative(mm.get_length(id1), STRIDE, CUT_LENGTH)

    cm = Cut_Match(p1, p2, get_diff_method(1), stride=stride, cut_length=cut_length)
    cm.cut_match()


if __name__ == '__main__':
    main()
