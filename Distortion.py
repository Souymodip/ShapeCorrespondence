import numpy as np
import Art
import ArtCollection

def distance(p1, point_list):
    n = [np.linalg.norm(p1 - p) for p in point_list]
    # print("||{} - {}|| := {}".format(p1, point_list, n))
    return np.sum(n)


def context(pl):
    return np.array([distance(p, pl) for p in pl])


def match(source, target):
    v2 = target.get_vertices()
    v1 = source.get_vertices()
    target_context = context(v2)
    source_context = context(v1)

    distortions = np.array([np.linalg.norm(np.roll(source_context, i) - target_context) for i in range(len(v1))])
    min = np.min(distortions)

    target_context_flip = np.flip(target_context, axis=0)
    distortions_flip = np.array([np.linalg.norm(np.roll(source_context, i) - target_context_flip) for i in range(len(source_context))])
    min_flip = np.min(distortions_flip)

    if min < min_flip:
        index = np.argmin(distortions)
        return np.roll(v1, index, axis=0), v2
    else:
        print(v2[-1], v2[0])
        v2_flip = np.flip(v2, axis=0)
        print(v2_flip[0], v2_flip[-1])
        index = np.argmin(distortions_flip)
        return np.roll(v1, index, axis=0), v2


def bloated_match(source, target):
    n1, n2 = len(source.beziers), len(target.beziers)
    lcm = np.lcm(n1, n2)

    def bloat(pl, k):
        for i in range(len(pl.beziers)):
            pl.split_bezier_in_parts(i, k)

    if lcm > n1:
        bloat(source, lcm / n1)
    if lcm > n2:
        bloat(target, lcm / n2)

    return match(source, target)
# def skip_match(pl1, pl2):


def draw_matching(d, m):
    interval = 0
    for p, q in zip(m[0], m[1]):
        if interval % 4 == 0:
            line = Art.Line(p, q)
            line.set_color((255, 0, 0))
            d.add_art(line)
        interval = interval + 1
    p, q = m[0][0], m[1][0]
    line = Art.Line([p[0], p[1]], [q[0], q[1]])
    line.set_color((0, 255, 0))
    d.add_art(line)


def elephant_giraffe():
    d = Art.Draw()

    '''Creating two arts at different location by defining the list of [anchor, in, out]'''
    elephant = ArtCollection.elephant
    elephant.apply(Art.Scale([5, 5]))
    elephant.apply(Art.Translate([-30, 10]))
    d.add_art(elephant)

    giraffe = ArtCollection.giraffe
    giraffe.apply(Art.Scale([5, 5]))
    # giraffe.apply(Art.Rotate(np.deg2rad(90), giraffe.get_centroid()))
    giraffe.apply(Art.Translate([5, -5]))
    giraffe.apply(Art.FlipX(giraffe.get_centroid()[0]))
    d.add_art(giraffe)

    ''' We are choosing the list of vertices for distortion calculation.
    Dynamic programming based distortion minimization can be used to remove this restriction. Furthermore, we are using
    only the anchors of the curve. Once can envisage distortion calculation involving the control points as well'''
    m = bloated_match(giraffe, elephant)
    draw_matching(d,  m)

    d.draw()


def lion_gorilla():
    d = Art.Draw()

    '''Creating two arts at different location by defining the list of [anchor, in, out]'''
    lion = ArtCollection.lion
    lion.apply(Art.Scale([5, 5]))
    lion.apply(Art.Translate([-15, -8]))
    d.add_art(lion)

    gorilla = ArtCollection.gorilla
    gorilla.apply(Art.Scale([5, 5]))
    # giraffe.apply(Art.Rotate(np.deg2rad(90), giraffe.get_centroid()))
    gorilla.apply(Art.Translate([0, -5]))
    gorilla.apply(Art.FlipX(gorilla.get_centroid()[0]))
    d.add_art(gorilla)

    ''' We are choosing the list of vertices for distortion calculation.
    Dynamic programming based distortion minimization can be used to remove this restriction. Furthermore, we are using
    only the anchors of the curve. Once can envisage distortion calculation involving the control points as well'''
    m = bloated_match(gorilla, lion)
    print(m)
    draw_matching(d, gorilla, lion, m)

    d.draw()


def cat_gorilla():
    d = Art.Draw()

    '''Creating two arts at different location by defining the list of [anchor, in, out]'''
    cat = ArtCollection.cat
    cat.apply(Art.Scale([5, 5]))
    cat.apply(Art.Translate([-15, -8]))
    d.add_art(cat)

    gorilla = ArtCollection.gorilla
    gorilla.apply(Art.Scale([5, 5]))
    gorilla.apply(Art.Translate([0, -5]))
    gorilla.apply(Art.FlipX(gorilla.get_centroid()[0]))
    d.add_art(gorilla)

    ''' We are choosing the list of vertices for distortion calculation.
    Dynamic programming based distortion minimization can be used to remove this restriction. Furthermore, we are using
    only the anchors of the curve. Once can envisage distortion calculation involving the control points as well'''
    m = bloated_match(gorilla, cat)
    draw_matching(d, m)

    d.draw()


def lion_bear():
    d = Art.Draw()

    '''Creating two arts at different location by defining the list of [anchor, in, out]'''
    lion = ArtCollection.lion
    lion.apply(Art.Scale([5, 5]))
    lion.apply(Art.Translate([-15, -8]))
    d.add_art(lion)

    bear = ArtCollection.bear
    bear.apply(Art.Scale([5, 5]))
    # giraffe.apply(Art.Rotate(np.deg2rad(90), giraffe.get_centroid()))
    bear.apply(Art.Translate([0, -5]))
    bear.apply(Art.FlipX(bear.get_centroid()[0]))
    d.add_art(bear)

    ''' We are choosing the list of vertices for distortion calculation.
    Dynamic programming based distortion minimization can be used to remove this restriction. Furthermore, we are using
    only the anchors of the curve. Once can envisage distortion calculation involving the control points as well'''
    m = bloated_match(lion, bear)
    draw_matching(d, m)

    d.draw()


def elephant_horse():
    d = Art.Draw()

    '''Creating two arts at different location by defining the list of [anchor, in, out]'''
    horse = ArtCollection.horse
    horse.apply(Art.Scale([5, 5]))
    horse.apply(Art.Translate([-15, -8]))
    d.add_art(horse)

    elephant = ArtCollection.elephant
    elephant.apply(Art.Scale([5, 5]))
    # giraffe.apply(Art.Rotate(np.deg2rad(90), giraffe.get_centroid()))
    elephant.apply(Art.Translate([-10, 8]))
    elephant.apply(Art.FlipX(elephant.get_centroid()[0]))
    d.add_art(elephant)

    ''' We are choosing the list of vertices for distortion calculation.
    Dynamic programming based distortion minimization can be used to remove this restriction. Furthermore, we are using
    only the anchors of the curve. Once can envisage distortion calculation involving the control points as well'''
    m = bloated_match(horse, elephant)
    draw_matching(d, m)

    d.draw()

def main():
    elephant_horse()


main()
