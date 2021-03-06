import CutMatch as cm
import Svg2Art
import os
from Helpers import bcolors

def execute(file1, file2, option=0):
    if not os.path.exists(file1):
        print(bcolors.FAIL + f"File \'{file1}\' does not exists!" + bcolors.ENDC)
        return
    if not os.path.exists(file2):
        print(bcolors.FAIL + f"File \'{file2}\' does not exists!" + bcolors.ENDC)
        return

    a1, a2 = Svg2Art.get_arts(file1, file2)
    cm.execute(a1, a2, option)
    # d = Art.Draw()
    # d.add_art(a1)
    # d.add_art(a2)
    # d.draw()