
import Art
import numpy as np

'''
.-----------------------------------------------------------.
|   Level 0 Test cases                                      |
|   Contains set of pair of arts and their rigid transform  |
'-----------------------------------------------------------'
'''


def get_test(index):
    if index == 0:
        '''Rotation Translation'''
        test0 = Art.PieceWiseBezier(np.array([
        [[2.00737, -2.61834], [2.00737, -2.61834], [2.00737, -2.61834]] , [[2.16317, -3.09962], [2.16317, -3.09962], [2.16317, -3.09962]] , [[2.66737, -3.09962], [2.66737, -3.09962], [2.66737, -3.09962]] , [[2.25946, -3.39706], [2.25946, -3.39706], [2.25946, -3.39706]] , [[2.41527, -3.87834], [2.41527, -3.87834], [2.41527, -3.87834]] , [[2.00737, -3.58089], [2.00737, -3.58089], [2.00737, -3.58089]] , [[1.59946, -3.87834], [1.59946, -3.87834], [1.59946, -3.87834]] , [[1.75527, -3.39706], [1.75527, -3.39706], [1.75527, -3.39706]] , [[1.34737, -3.09962], [1.34737, -3.09962], [1.34737, -3.09962]] , [[1.85156, -3.09962], [1.85156, -3.09962], [1.85156, -3.09962]]
            ]), is_closed=True)
        test1 = Art.PieceWiseBezier(np.array([
        [[5.5555, -5.87009], [5.5555, -5.87009], [5.5555, -5.87009]] , [[5.3721, -6.34154], [5.3721, -6.34154], [5.3721, -6.34154]] , [[5.76285, -6.66017], [5.76285, -6.66017], [5.76285, -6.66017]] , [[5.25876, -6.63291], [5.25876, -6.63291], [5.25876, -6.63291]] , [[5.07535, -7.10437], [5.07535, -7.10437], [5.07535, -7.10437]] , [[4.94721, -6.61607], [4.94721, -6.61607], [4.94721, -6.61607]] , [[4.44311, -6.58881], [4.44311, -6.58881], [4.44311, -6.58881]] , [[4.868, -6.31428], [4.868, -6.31428], [4.868, -6.31428]] , [[4.73985, -5.82598], [4.73985, -5.82598], [4.73985, -5.82598]] , [[5.13061, -6.14462], [5.13061, -6.14462], [5.13061, -6.14462]]
            ]), is_closed=True)

        test0.apply(Art.Scale(5))
        test0.apply(Art.Translate([-20, 20]))
        test1.apply(Art.Scale(5))
        test1.apply(Art.Translate([-15, 35]))
        return test0, test1

    if index == 1:
        ''' Rotation Scale Translation'''
        test0 = Art.PieceWiseBezier(np.array([
        [[2.00737, -2.61834], [2.00737, -2.61834], [2.00737, -2.61834]] , [[2.16317, -3.09962], [2.16317, -3.09962], [2.16317, -3.09962]] , [[2.66737, -3.09962], [2.66737, -3.09962], [2.66737, -3.09962]] , [[2.25946, -3.39706], [2.25946, -3.39706], [2.25946, -3.39706]] , [[2.41527, -3.87834], [2.41527, -3.87834], [2.41527, -3.87834]] , [[2.00737, -3.58089], [2.00737, -3.58089], [2.00737, -3.58089]] , [[1.59946, -3.87834], [1.59946, -3.87834], [1.59946, -3.87834]] , [[1.75527, -3.39706], [1.75527, -3.39706], [1.75527, -3.39706]] , [[1.34737, -3.09962], [1.34737, -3.09962], [1.34737, -3.09962]] , [[1.85156, -3.09962], [1.85156, -3.09962], [1.85156, -3.09962]]
            ]), is_closed=True)
        test1 = Art.PieceWiseBezier(np.array([
        [[5.5555, -5.87009], [5.5555, -5.87009], [5.5555, -5.87009]] , [[5.3721, -6.34154], [5.3721, -6.34154], [5.3721, -6.34154]] , [[5.76285, -6.66017], [5.76285, -6.66017], [5.76285, -6.66017]] , [[5.25876, -6.63291], [5.25876, -6.63291], [5.25876, -6.63291]] , [[5.07535, -7.10437], [5.07535, -7.10437], [5.07535, -7.10437]] , [[4.94721, -6.61607], [4.94721, -6.61607], [4.94721, -6.61607]] , [[4.44311, -6.58881], [4.44311, -6.58881], [4.44311, -6.58881]] , [[4.868, -6.31428], [4.868, -6.31428], [4.868, -6.31428]] , [[4.73985, -5.82598], [4.73985, -5.82598], [4.73985, -5.82598]] , [[5.13061, -6.14462], [5.13061, -6.14462], [5.13061, -6.14462]]
            ]), is_closed=True)
        return test0, test1

    if index == 2:
        test0 =  Art.PieceWiseBezier(np.array([
        [[4.96309, -5.04897], [4.89511, -5.27358], [5.03153, -4.82285]] , [[5.52361, -4.79566], [5.52361, -4.79566], [5.52361, -4.79566]] , [[5.09006, -4.31634], [5.09006, -4.31634], [5.09006, -4.31634]] , [[4.70219, -4.84135], [5.11092, -4.79967], [4.26837, -4.88559]] , [[4.06491, -4.39388], [4.06491, -4.39388], [4.06491, -4.39388]] , [[3.76142, -4.80039], [3.76142, -4.80039], [3.76142, -4.80039]] , [[3.9509, -5.3137], [3.9509, -5.3137], [3.9509, -5.3137]] , [[4.62544, -4.99096], [4.21105, -4.7303], [5.02307, -5.24107]] , [[4.62232, -5.753], [4.62232, -5.753], [4.62232, -5.753]] , [[5.1996, -5.70548], [5.1996, -5.70548], [5.1996, -5.70548]]
            ]), is_closed=True)
        test1 = Art.PieceWiseBezier(np.array([
        [[4.73054, -5.33396], [4.49646, -5.31728], [4.96619, -5.35074]] , [[5.10143, -5.82466], [5.10143, -5.82466], [5.10143, -5.82466]] , [[5.47311, -5.29591], [5.47311, -5.29591], [5.47311, -5.29591]] , [[4.87538, -5.03363], [5.00634, -5.42305], [4.73638, -4.6203]] , [[5.17099, -4.31323], [5.17099, -4.31323], [5.17099, -4.31323]] , [[4.70747, -4.10706], [4.70747, -4.10706], [4.70747, -4.10706]] , [[4.24871, -4.40527], [4.24871, -4.40527], [4.24871, -4.40527]] , [[4.71251, -4.99183], [4.87517, -4.53009], [4.55643, -5.43489]] , [[3.96861, -5.15716], [3.96861, -5.15716], [3.96861, -5.15716]] , [[4.14251, -5.70967], [4.14251, -5.70967], [4.14251, -5.70967]]
            ]), is_closed=True)

        test0.apply(Art.Scale(5))
        test0.apply(Art.Translate([-30, 30]))
        test1.apply(Art.Scale(3))
        test1.apply(Art.Translate([0, 20]))
        return test0, test1
    ''' Symmetric reflection about x = 0 '''
    if index == 3:
        shark1 = Art.PieceWiseBezier(np.array([
            [[3.18653, -7.7979], [2.51663, -8.14829], [3.55991, -7.60261]],
            [[4.97359, -7.26631], [4.20085, -7.3226], [5.34176, -7.2395]],
            [[6.14438, -7.02566], [6.0331, -7.22105], [6.55277, -6.3086]],
            [[7.86285, -5.88481], [7.86285, -5.88481], [7.86285, -5.88481]],
            [[7.68371, -7.22039], [7.30413, -6.7215], [7.98068, -7.6107]],
            [[9.67019, -7.8402], [8.84569, -7.6699], [10.144, -7.93806]],
            [[11.3241, -7.94505], [11.1132, -8.22488], [11.902, -7.17828]],
            [[12.6761, -7.04223], [12.6761, -7.04223], [12.6761, -7.04223]],
            [[12.0003, -8.16643], [12.0003, -8.16643], [12.0003, -8.16643]],
            [[12.6919, -9.35143], [12.4774, -8.80204], [12.8302, -9.70582]],
            [[11.4742, -8.6448], [11.4742, -8.6448], [11.4742, -8.6448]],
            [[9.91189, -8.93885], [9.91189, -8.93885], [9.91189, -8.93885]],
            [[10.3736, -9.75698], [10.3736, -9.75698], [10.3736, -9.75698]],
            [[9.05765, -8.93606], [9.42766, -9.39996], [8.87717, -8.70979]],
            [[6.16403, -9.25355], [6.01899, -8.55394], [6.23488, -9.59529]],
            [[6.96231, -10.4106], [6.96231, -10.4106], [6.96231, -10.4106]],
            [[4.73407, -9.2391], [5.29222, -9.96773], [4.73407, -9.2391]],
            [[3.89514, -9.23964], [4.68024, -9.28627], [3.5161, -9.21713]],
            [[2.6493, -9.08317], [3.14582, -9.16858], [2.31766, -9.02613]],
            [[1.92412, -8.69059], [1.88786, -8.96997], [1.9408, -8.56206]]
        ]), is_closed=True, show_control=False)
        shark1.apply(Art.Translate([0, 10]))
        shark2 = Art.PieceWiseBezier(np.array([
            [[3.18653, -7.7979], [2.51663, -8.14829], [3.55991, -7.60261]],
            [[4.97359, -7.26631], [4.20085, -7.3226], [5.34176, -7.2395]],
            [[6.14438, -7.02566], [6.0331, -7.22105], [6.55277, -6.3086]],
            [[7.86285, -5.88481], [7.86285, -5.88481], [7.86285, -5.88481]],
            [[7.68371, -7.22039], [7.30413, -6.7215], [7.98068, -7.6107]],
            [[9.67019, -7.8402], [8.84569, -7.6699], [10.144, -7.93806]],
            [[11.3241, -7.94505], [11.1132, -8.22488], [11.902, -7.17828]],
            [[12.6761, -7.04223], [12.6761, -7.04223], [12.6761, -7.04223]],
            [[12.0003, -8.16643], [12.0003, -8.16643], [12.0003, -8.16643]],
            [[12.6919, -9.35143], [12.4774, -8.80204], [12.8302, -9.70582]],
            [[11.4742, -8.6448], [11.4742, -8.6448], [11.4742, -8.6448]],
            [[9.91189, -8.93885], [9.91189, -8.93885], [9.91189, -8.93885]],
            [[10.3736, -9.75698], [10.3736, -9.75698], [10.3736, -9.75698]],
            [[9.05765, -8.93606], [9.42766, -9.39996], [8.87717, -8.70979]],
            [[6.16403, -9.25355], [6.01899, -8.55394], [6.23488, -9.59529]],
            [[6.96231, -10.4106], [6.96231, -10.4106], [6.96231, -10.4106]],
            [[4.73407, -9.2391], [5.29222, -9.96773], [4.73407, -9.2391]],
            [[3.89514, -9.23964], [4.68024, -9.28627], [3.5161, -9.21713]],
            [[2.6493, -9.08317], [3.14582, -9.16858], [2.31766, -9.02613]],
            [[1.92412, -8.69059], [1.88786, -8.96997], [1.9408, -8.56206]]
        ]), is_closed=True, show_control=False)
        shark2.apply(Art.FlipX())
        return shark2, shark1
    else:
        print ("Level 0 test case of index {} is not available. Choose from : {}".format(index, ', '.join([str(i) for i in range(4)])))
        return None


def main_test(test):
    d = Art.Draw()
    d.add_art(test[0])
    d.add_art(test[1])
    d.draw()

# main_test(get_test(3))