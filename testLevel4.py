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
    if index ==0:
        ags = [ArtCollection.girl0, ArtCollection.girl1, ArtCollection.girl2, ArtCollection.girl3, ArtCollection.girl4,
               ArtCollection.girl5, ArtCollection.girl6, ArtCollection.girl7, ArtCollection.girl8, ArtCollection.girl9,
               ArtCollection.girl10, ArtCollection.girl11]

        for ag in ags:
            ag.apply(Art.Translate(-ag.get_centroid()))
            ag.apply(Art.Scale(55))
        return ags
    else:
        return []
