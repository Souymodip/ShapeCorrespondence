import Art
import numpy as np

'''
.---------------------------------------------------------------.
|   Level 2 Test cases                                          |
|   Contains set of pair of closed piece-wise Bezier curves     |
|   The two arts have the different number of anchor points     | 
'---------------------------------------------------------------'
'''

def get_test(index):
    if index == 0:
        test1 = [
            Art.PieceWiseBezier(np.array([

            ]), is_closed=True),
            Art.PieceWiseBezier(np.array([

            ]), is_closed=True)
        ]