from xml.dom import minidom
import Art
import numpy as np

from svg.path import parse_path
from svg.path.path import CubicBezier
from svg.path.path import Close
from svg.path.path import Line


def to_point(coord):
    return np.array([coord.real, - coord.imag])


def curve_anchor(e, last):
    p = to_point(e.start)
    in1 = p if last is None else last
    out1 = to_point(e.control1)
    last = to_point(e.control2)
    return np.array([p, in1, out1]), last


def line_anchor(e, last):
    p = to_point(e.start)
    in1 = p if last is None else last
    out1 = p
    last = to_point(e.end)
    return np.array([p, in1, out1]), last


def to_PiecewiseBezier(path_strings):
    arts = []
    for path_string in path_strings:
        bzs = []
        path = parse_path(path_string)
        last = None
        is_closed = False
        print(path)
        for e in path:
            if isinstance(e, CubicBezier):
                b, last = curve_anchor(e, last)
                print ("B:{} :-> {}".format(e, b))
                bzs.append(b)
                last_curve = e
            elif isinstance(e, Line) :
                b, last = line_anchor(e, last)
                print("L:{} :-> {}".format(e, b))
                bzs.append(b)
            elif isinstance(e, Close):
                b, last = line_anchor(e, last)
                bzs.append(b)
                print("C:{} :-> {}".format(e, b))
                is_closed = True
                break
        if not is_closed:
            p = to_point(last_curve.end)
            in1 = last
            out1 = p
            print("p:{}, in:{}, out:{}".format(p, in1, out1))
            bzs.append(np.array([p, in1, out1]))

        print (is_closed)
        print (bzs)
        arts.append(Art.PieceWiseBezier(np.array(bzs), is_closed=is_closed))
    return arts


def read(svg_file, scale =None):
    print("Reading file :{}".format(svg_file))
    doc = minidom.parse(svg_file)  # parseString also exists
    path_strings = [path.getAttribute('d') for path
                    in doc.getElementsByTagName('path')]
    doc.unlink()
    arts = to_PiecewiseBezier(path_strings)
    if len(arts) == 0 :
        print("Error: Could not parse svg {}".format(svg_file))
        exit(0)
    elif len(arts) > 1:
        print("Warning: Multiple Piecewise Bezier curve is not handled in this version. Choosing the first.")

    art = arts[0]
    art.apply(Art.Translate(-art.get_centroid()))

    if scale is None:
        bbox = art.get_bounding_box()
        width = np.abs(bbox[0][0] - bbox[1][0])
        height = np.abs(bbox[1][0] - bbox[1][1])

        if width + height <= 1.e-6:
            print("Error: Art is too bloody small for me to see")
            exit(0)
        scale = 15 / max(width, height)
    art.apply(Art.Scale(scale))


    return art, scale


def get_arts(file1, file2):
    # svg_file = "/Users/souchakr/Research/svg/cubic_bezier2.svg"
    art1, scale = read(file1)
    art2, _ = read(file2, scale)
    return art1, art2


if __name__ == '__main__':
    svg_file1 = "/Users/souchakr/Research/svg/cubic_bezier2.svg"
    svg_file2 = "/Users/souchakr/Research/svg/cubic_bezier.svg"
    art1, art2 = get_arts(svg_file1, svg_file2)
    d = Art.Draw()
    d.add_art(art1)
    d.add_art(art2)
    d.draw()