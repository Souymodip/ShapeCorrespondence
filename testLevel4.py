import numpy as np
import Art
import ArtCollection


'''
.---------------------------------------------------------------.
|   Level 4 Test cases : Realistic examples                     |
|   Contains set of sets of closed piece-wise Bezier curves     |
|   Each of the arts have the different number of anchor points |
|   Each art group may have different number of arts            |
'---------------------------------------------------------------'
'''


def get_test(index):
    ags = [ArtCollection.girl0, ArtCollection.girl1, ArtCollection.girl2, ArtCollection.girl3, ArtCollection.girl4,
           ArtCollection.girl5, ArtCollection.girl6, ArtCollection.girl7, ArtCollection.girl8, ArtCollection.girl9,
           ArtCollection.girl10, ArtCollection.girl11]
    for ag in ags:
        ag.apply(Art.Translate(-ag.get_centroid()))
        ag.apply(Art.Scale(55))

    if index == 0:
        return [ags[0], ags[2], ags[4], ags[5], ags[6], ags[8]]
    if index == 1:
        return [ags[1], ags[3], ags[7], ags[9], ags[10], ags[11]]

    kids = [ArtCollection.kid0, ArtCollection.kid1]
    for ag in kids:
        ag.apply(Art.Translate(-ag.get_centroid()))
        ag.apply(Art.Scale(40))
    kids[0].apply(Art.Translate([-10, 0])), kids[1].apply(Art.Translate([10, 0]))
    if index == 2:
        return kids

    kids_jump = [ArtCollection.child0, ArtCollection.child1, ArtCollection.child2, ArtCollection.child3, ArtCollection.child4, ArtCollection.child5, ArtCollection.child6, ArtCollection.child7, ArtCollection.child8, ArtCollection.child9]
    for ag in kids_jump:
        c = ag.get_centroid()
        ag.apply(Art.Translate(-c))
        ag.apply(Art.Scale(40))
        ag.apply(Art.Translate(c))
    if index == 3:
        return kids_jump
    else:
        return []


if __name__ == '__main__':
    kids = get_test(3)
    d = Art.Draw()
    for k in kids:
        d.add_art(k)
    d.draw()