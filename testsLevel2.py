import Art
import numpy as np
import ArtCollection


'''
.---------------------------------------------------------------.
|   Level 2 Test cases                                          |
|   Contains set of pair of closed piece-wise Bezier curves     |
|   The two arts have the different number of anchor points     | 
'---------------------------------------------------------------'
'''

def get_test(index):
    if index == 0:
        test0 = Art.PieceWiseBezier(np.array([
                [[0.206395, -5.23171], [0.206395, -5.85327], [0.206395, -5.23171]],
                [[0.206395, -1.27258], [0.206395, -1.27258], [0.206395, -0.651023]],
                [[1.33183, -0.147147], [0.71027, -0.147147], [1.33183, -0.147147]],
                [[4.95096, -0.147147], [4.95096, -0.147147], [5.57252, -0.147147]],
                [[6.07639, -1.27258], [6.07639, -0.651023], [6.07639, -1.27258]],
                [[6.07639, -5.23171], [6.07639, -5.23171], [6.07639, -5.85327]],
                [[4.95096, -6.35715], [5.57252, -6.35715], [4.95096, -6.35715]],
                [[1.33183, -6.35715], [1.33183, -6.35715], [0.71027, -6.35715]]
            ]), is_closed=True, show_control=False)
        test1 = Art.PieceWiseBezier(np.array([
                [[0.206395, -5.23171], [0.206395, -5.85327], [0.206395, -5.23171]],
                [[0.206395, -3.29828], [0.206395, -3.29828], [0.206395, -3.29828]],
                [[0.206395, -2.45414], [0.206395, -2.45414], [0.206395, -2.45414]],
                [[0.206395, -1.27258], [0.206395, -1.27258], [0.206395, -0.651023]],
                [[1.33183, -0.147147], [0.71027, -0.147147], [1.33183, -0.147147]],
                [[3.30956, -0.147147], [3.30956, -0.147147], [3.30956, -0.147147]],
                [[4.95096, -0.147147], [4.95096, -0.147147], [5.57252, -0.147147]],
                [[6.07639, -1.27258], [6.07639, -0.651023], [6.07639, -1.27258]],
                [[6.07639, -5.23171], [6.07639, -5.23171], [6.07639, -5.85327]],
                [[4.95096, -6.35715], [5.57252, -6.35715], [4.95096, -6.35715]],
                [[1.33183, -6.35715], [1.33183, -6.35715], [0.71027, -6.35715]]
            ]), is_closed=True, show_control=False)

        test0.apply(Art.Translate([-15, 0]))
        test1.apply(Art.Translate([5, 0]))
        return test0, test1

    if index == 1:
        t1 = ArtCollection.lion
        t1.apply(Art.Scale(5))
        t1.apply(Art.Translate([-15, -12]))
        t2 = ArtCollection.gorilla
        t2.apply(Art.Scale(5))
        t2.apply(Art.Translate([2, -1]))
        return t1, t2

    if index == 2:
        t1 = ArtCollection.elephant
        t1.apply(Art.Scale(6))
        t1.apply(Art.Translate([-36, 13]))

        t2 = ArtCollection.giraffe
        t2.apply(Art.Scale(6))
        t2.apply(Art.Translate([2, -1]))
        return t1, t2

    if index == 3:
        t1 = ArtCollection.maple_leaf
        t2 = ArtCollection.plane
        t1.apply(Art.Scale(10))
        t2.apply(Art.Scale(10))
        t1.apply(Art.Translate([-40, 40]))
        t2.apply(Art.Translate([-20, 40]))
        return t1, t2

    if index == 4:
        t1 = ArtCollection.dog1
        t2 = ArtCollection.dog2
        t1.apply(Art.Scale(5))
        t2.apply(Art.Scale(5))
        t1.apply(Art.Translate([-25, 10]))
        t2.apply(Art.Translate([0, 10]))
        return t1, t2

    if index == 5:
        v0 = ArtCollection.snowMan
        v1 = ArtCollection.HalfMan
        v0.apply(Art.Translate([-10, 0]))
        v1.apply(Art.Translate([10, 0]))
        return v0, v1


def main_test(test):
    d = Art.Draw()
    t1, t2 = test
    d.add_art(t1)
    d.add_art(t2)
    d.draw()
