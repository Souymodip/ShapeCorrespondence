import numpy as np
import Art
import ArtCollection
import copy as cp

'''
.---------------------------------------------------------------.
|   Level 4 Test cases : Realistic examples                     |
|   Contains set of sets of closed piece-wise Bezier curves     |
|   Each of the arts have the different number of anchor points |
|   Each art group may have different number of arts            |
'---------------------------------------------------------------'
'''


def get_test(index):
    ags = cp.deepcopy([ArtCollection.girl0, ArtCollection.girl1, ArtCollection.girl2, ArtCollection.girl3, ArtCollection.girl4,
           ArtCollection.girl5, ArtCollection.girl6, ArtCollection.girl7, ArtCollection.girl8, ArtCollection.girl9,
           ArtCollection.girl10, ArtCollection.girl11])
    for ag in ags:
        ag.apply(Art.Translate(-ag.get_centroid()))
        ag.apply(Art.Scale(55))

    if index == 0:
        return [ags[0], ags[2], ags[4], ags[5], ags[6], ags[8]]
    if index == 1:
        return [ags[1], ags[3], ags[7], ags[9], ags[10], ags[11]]

    kids = cp.deepcopy([ArtCollection.kid0, ArtCollection.kid1])
    for ag in kids:
        ag.apply(Art.Translate(-ag.get_centroid()))
        ag.apply(Art.Scale(40))
    kids[0].apply(Art.Translate([-10, 0])), kids[1].apply(Art.Translate([10, 0]))
    if index == 2:
        return kids

    kids_jump = cp.deepcopy([ArtCollection.child0, ArtCollection.child1, ArtCollection.child2, ArtCollection.child3, ArtCollection.child4, ArtCollection.child5, ArtCollection.child6, ArtCollection.child7, ArtCollection.child8, ArtCollection.child9])
    for ag in kids_jump:
        c = ag.get_centroid()
        ag.apply(Art.Translate(-c))
        ag.apply(Art.Scale(40))
    if index == 3:
        return kids_jump

    ins = cp.deepcopy([ArtCollection.insects0, ArtCollection.insects1, ArtCollection.insects2, ArtCollection.insects3,
                       ArtCollection.insects4, ArtCollection.insects5, ArtCollection.insects6, ArtCollection.insects7,
                       ArtCollection.insects8, ArtCollection.insects9, ArtCollection.insects10, ArtCollection.insects11,
                       ArtCollection.insects12, ArtCollection.insects13, ArtCollection.insects14])
    if index ==4:
        for i in ins:
            c = i.get_centroid()
            i.apply(Art.Translate(-c))
            i.apply(Art.Scale(0.1))
        return ins
    else:
        return []


if __name__ == '__main__':
    kids = get_test(4)
    d = Art.Draw()
    d.add_art(kids[3])
    # for k in kids:
    #     d.add_art(k)
    d.draw()