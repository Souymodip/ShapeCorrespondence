import FunctionTransform as ft
import Discriminator as dis
import testsLevel3 as testsuite
import Art
import numpy as np


importance_angle = 1
test = 17 # strange brew

def match(from_match, to_match):
    fs1, discriminators = from_match
    assert (len(fs1) == len(discriminators))
    mat = np.zeros((len(discriminators), to_match.no_of_arts()))
    fs2 = [ft.art_to_function(art, importance_angle=importance_angle) for art in to_match.get_arts()]

    for ij in np.ndindex(mat.shape):
        i,j = ij
        fi_x = fs1[i][1]
        out = dis.test(discriminators[i], ft.change_suport(f=fs2[j], new_x=fi_x))
        mat[ij] = out[0]
    return mat


def main():
    ags = testsuite.get_test(6)
    a1, a2 = ags[0], ags[1]

    fs1 = [ft.art_to_function(art, importance_angle=importance_angle) for art in a2.get_arts()]
    discriminators = [dis.train_for(f) for f in fs1]
    mat = match(from_match=(fs1, discriminators), to_match=a2)
    d = Art.Draw()
    d.add_art(a1)
    d.add_art(a2)
    Art.draw_match_grp(a1, a2, mat, d)

if __name__ == '__main__':
    main()