import CutMatch as cm
import Svg2Art
import Art


def execute(file1, file2):
    a1, a2 = Svg2Art.get_arts(file1, file2)
    cm.execute(a1, a2)
    # d = Art.Draw()
    # d.add_art(a1)
    # d.add_art(a2)
    # d.draw()