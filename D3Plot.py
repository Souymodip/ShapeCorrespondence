import matplotlib.pyplot as plt


def draw_poly_index(polys, matched_index=[]):
    matched_point = []
    for m in matched_index:
        points = []
        print (m)
        for k in range(len(m)):
            points.append(polys[k].get_vertex(m[k]))
        matched_point.append(points)
    draw_polys(polys, matched_point)


def draw_polys(polys, match=[]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def draw_poly(poly, z, ax):
        xs, ys, zs = [], [], []
        for p in poly.get_vertices():
            xs.append(p[0])
            ys.append(p[1])
            zs.append(z)
        if poly.isClosed:
            xs.append(xs[0])
            ys.append(ys[0])
            zs.append(zs[0])
        ax.plot(xs=xs, ys=ys, zs=zs)

    i = 0
    for poly in polys:
        draw_poly(poly, i, ax)
        i = i + 1

    if len(polys) == 2:
        for mp in match:
            xs, ys, zs = [], [], []
            xs.append(mp[0][0]), xs.append(mp[1][0])
            ys.append(mp[0][1]), ys.append(mp[1][1])
            zs.append(0), zs.append(1)
            ax.plot(xs=xs, ys=ys, zs=zs, linestyle='--', color='aqua')

    plt.show()
