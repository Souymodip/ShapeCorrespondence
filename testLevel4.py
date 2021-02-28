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
    else:
        return []
