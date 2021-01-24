import numpy as np
import Art
import ShapeSimilarity
import testsLevel3 as testsuite
from scipy.optimize import linprog


DIFF_THRESHOLD = 1.e-7
SIMILARITY_THRESHOLD = 0.1

def draw_bbox(draw, bbox, color):
    bot_left = bbox[0]
    bot_right = [bbox[1][0], bbox[0][1]]
    top_right = bbox[1]
    top_left = [bbox[0][0], bbox[1][1]]
    l1 = Art.Line(bot_left, bot_right)
    l1.set_color(color)
    l2 = Art.Line(bot_right, top_right)
    l2.set_color(color)
    l3 = Art.Line(top_right, top_left)
    l3.set_color(color)
    l4 = Art.Line(top_left, bot_left)
    l4.set_color(color)
    draw.add_art(l1)
    draw.add_art(l2)
    draw.add_art(l3)
    draw.add_art(l4)
    c = Art.Circle(np.mean(bbox, axis=0), 0.2)
    c.set_fill_color((255, 0, 0))
    draw.add_art(c)


def match(mat):
    def get_lhs(start, width, total):
        assert (total % width == 0)
        ret = []
        for i in range(int(total/width)):
            if i == start:
                ret = ret + width * [1.0]
            else:
                ret = ret + width * [0.0]
        return ret

    obj = mat.flatten()
    lhs_eq = [np.ones(shape=(obj.shape[0]))]
    rhs_eq = [mat.shape[0]]
    lhs_ineq = [ get_lhs(i, mat.shape[1], obj.shape[0]) for i in range(mat.shape[0])]
    rhs_ineq = np.ones(shape=mat.shape[0])
    print ("Obj : {}".format(obj))
    print ("LHS: {} <= {}".format(lhs_eq, rhs_eq))
    bnd = [(0, 1.0) for i in range(obj.shape[0])]
    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd, method="revised simplex")
    x = np.zeros(mat.shape)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            x[i,j] = opt.x[i * mat.shape[1] + j]
    print(x)

def draw_matching(artGrp1, artGrp2, mat, draw):
    match(artGrp1, artGrp2, mat, draw)
    high_match = set()
    for i in range(mat.shape[0]):
        j = np.argmin(mat[i])
        if mat[i,j] <= SIMILARITY_THRESHOLD:
            high_match.add(j)
            bbox_i = artGrp1.get_art(i).get_bounding_box()
            bbox_j = artGrp2.get_art(j).get_bounding_box()
            draw_bbox(draw, bbox_i, (0, 0, 255))
            draw_bbox(draw, bbox_j, (0, 0, 255))
            c_i, c_j = np.mean(bbox_i, axis=0), np.mean(bbox_j, axis=0)
            mid1, mid2 = (c_i/3 + c_j *2 /3), (c_i *2/3 + c_j/3)
            mid1[1], mid2[1] = mid1[1] + 5, mid2[1] + 5
            b = Art.Bezier([c_i, mid2, mid1, c_j])
            b.set_color((255, 0, 255))
            draw.add_art(b)



def marching(artGrp1, artGrp2, draw):
    mat = np.zeros(shape=(artGrp1.no_of_arts(), artGrp2.no_of_arts()))
    print (mat.shape)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            mat_ij = ShapeSimilarity.measure(art1=artGrp1.get_art(i), art2=artGrp2.get_art(j), d=None)
            if len(mat_ij) > 0:
                print("Diff{} := Match:{}, Val:{}".format((i,j), mat_ij[0][0], mat_ij[0][1]))
                mat[i,j] = mat_ij[0][-1]
                if mat[i,j] < DIFF_THRESHOLD:
                    break
            else:
                mat[i,j] = np.inf

    draw_matching(artGrp1, artGrp2, mat, draw)


def main():
    d = Art.Draw()
    ag1, ag2 = testsuite.get_test(5)
    # ag1.keep(0)
    # ag2.keep(0)
    d.add_art(ag1)
    d.add_art(ag2)
    marching(ag1, ag2, d)
    d.draw()


if __name__ == '__main__':
    main()