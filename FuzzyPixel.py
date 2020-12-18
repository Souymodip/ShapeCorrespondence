import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import Analytic
import Art
import multiprocessing


N = 10000
sigma = 0.1
sigma2 = sigma * sigma
alpha = 10

def create_random_samples(n):
    d = int(n/100)
    subsets = np.arange(0, n + 1, n / d)
    u = np.zeros(n)
    for i in range(d):
        start = int(subsets[i])
        end = int(subsets[i + 1])
        u[start:end] = np.random.uniform(low=i / d, high=(i + 1) / d, size=end - start)
    np.random.shuffle(u)
    return u


def get_length(L):
    ax, ay = L[0]
    bx, by = L[1]
    return np.sqrt((bx - ax)*(bx - ax) + (by - ay)*(by - ay))


def dist(L, q, length):
    qx, qy = q
    ax, ay = L[0]
    bx, by = L[1]
    return ((bx - ax)*(by - qy) - (bx - qx)*(by - ay)) / length


def H(x):
    return 1 / (1 + np.exp(-alpha * x))
    # return 0.5 * (x / np.sqrt(l + np.power(x, 2)) + 1) // algebraic

def dH(x):
    return np.power(H(x), 2.0) * alpha

def D(x, sigma = 0.1):
    return np.exp(-np.power(x, 2) / (2 * np.power(sigma, 2))) / (np.sqrt(2 * np.pi) * sigma)


def kernel(q, L):
    qx, qy = q
    ax, ay = L[0]
    bx, by = L[1]

    def x(t):
        return ax + (bx - ax) * t

    def y(t):
        return ay + (by - ay) * t

    def dy_dt(t):
        return by - ay

    def f(t):
        # return  D(y(t) - qy) * dy_dt(t)
        return H(x(t) - qx) * D(y(t) - qy) * dy_dt(t)

    return np.vectorize(f)


def dkernel(q, L):
    qx, qy = q
    ax, ay = L[0]
    bx, by = L[1]

    def x(t):
        return ax + (bx - ax) * t

    def y(t):
        return ay + (by - ay) * t

    def dy_dt(t):
        return by - ay

    def dx_dt(t):
        return bx - ax

    def D1(t):
        return dH(x(t) - qx) * -y(t) * D(y(t) - qy) * dy_dt(t)

    def D2(t):
        return - H(x(t) - qx) * D(y(t) - qy) * (y(t) - qy) * x(t) / sigma2

    def D3(t):
        return H(x(t) - qx) * D(y(t) - qy) * dx_dt(t)

    def f(t):
        return D1(t) + D2(t) + D3(t)

    return np.vectorize(f)


def I(q, L, lim, n, u):
    fv = kernel(q, L)
    a = lim[0]
    b = lim[1]
    ufv = fv(a + (b - a) * u)
    return ((b - a) / n) * ufv.sum()


def dI(q, L, lim, n, u):
    fv = dkernel(q, L)
    a = lim[0]
    b = lim[1]
    ufv = fv(a + (b - a) * u)
    return ((b - a) / n) * ufv.sum()


def area(polygon, raster, offset, scale):
    now = datetime.now()
    Ls = Analytic.get_line_list(polygon=polygon)

    assert(len(raster.shape) == 3)
    lengths = np.array([get_length(L) for L in Ls])

    u = create_random_samples(N)

    for ij in np.ndindex(raster.shape[:2]):
        qx, qy = ij[0] / scale + offset[0], ij[1] / scale + offset[1]
        intensity = 0

        for L, length in zip(Ls, lengths):
            if dist(L, (qx, qy), length) * scale > 1 :
                intensity = intensity + Analytic.line_integ_HD(L, qx, qy)
            else:
                intensity = intensity + I((qx, qy), L, (0, 1), N, u)

        raster[ij] = intensity

    return datetime.now() - now


def mp_overlap_poly_raster(args):
    polygon, raster, overlap_raster, offset, scale, angle = args
    t = overlap_poly_raster(polygon, raster, overlap_raster, offset, scale)
    print("Delta Time :={}, Anlge: {:.3f}{}, Overlap Intensity {}".format(t, angle, u"\N{DEGREE SIGN}", np.sum(overlap_raster)))
    return polygon, overlap_raster


def overlap_poly_raster(polygon, raster, overlap_raster, offset, scale):
    now = datetime.now()
    Ls = Analytic.get_line_list(polygon=polygon)

    assert (len(raster.shape) == 3 and raster.shape == overlap_raster.shape)
    lengths = np.array([get_length(L) for L in Ls])

    u = create_random_samples(N)

    for ij in np.ndindex(raster.shape[:2]):
        qx, qy = ij[0] / scale + offset[0], ij[1] / scale + offset[1]
        intensity = 0
        if raster[ij] !=0 :
            for L, length in zip(Ls, lengths):
                if np.abs(dist(L, (qx, qy), length)) * scale > 1:
                    intensity = intensity + Analytic.line_integ_HD(L, qx, qy)
                else:
                    intensity = intensity + I((qx, qy), L, (0, 1), N, u)

        overlap_raster[ij] = intensity * raster[ij]
    return datetime.now() - now


def overlap_polys(polygon1, polygon2, raster, offset, scale):
    now = datetime.now()
    Ls1 = Analytic.get_line_list(polygon=polygon1)
    Ls2 = Analytic.get_line_list(polygon=polygon2)

    assert (len(raster.shape) == 3)
    lengths1 = np.array([get_length(L) for L in Ls1])
    lengths2 = np.array([get_length(L) for L in Ls2])

    u = create_random_samples(N)

    def get_intensity(qx, qy, Ls, lengths):
        intensity = 0
        for L, length in zip(Ls, lengths):
            if dist(L, (qx, qy), length) * scale > 1:
                intensity = intensity + Analytic.line_integ_HD(L, qx, qy)
            else:
                intensity = intensity + I((qx, qy), L, (0, 1), N, u)
        return intensity

    for ij in np.ndindex(raster.shape[:2]):
        qx, qy = ij[0] / scale + offset[0], ij[1] / scale + offset[1]
        i1 = get_intensity(qx, qy, Ls=Ls1, lengths=lengths1)
        if i1 != 0:
            i2 = get_intensity(qx, qy, Ls=Ls2, lengths=lengths2)
        else:
            i2 = 0
        raster[ij] = i1 * i2

    return datetime.now() - now


def m_grad_overlap(arg):
    polygon, raster, overlap_raster, offset, scale = arg
    t = grad_overlap(polygon, raster, overlap_raster, offset, scale)
    print("Delta Time for overlap :={},  Overlap Intensity {}".format(t, np.sum(overlap_raster)))
    return polygon, overlap_raster


def grad_overlap(polygon, raster, overlap_raster, offset, scale):
    now = datetime.now()
    Ls = Analytic.get_line_list(polygon=polygon)

    assert (len(raster.shape) == 3 and raster.shape == overlap_raster.shape)
    lengths = np.array([get_length(L) for L in Ls])

    u = create_random_samples(N)

    for ij in np.ndindex(raster.shape[:2]):
        qx, qy = ij[0] / scale + offset[0], ij[1] / scale + offset[1]
        intensity = 0
        if raster[ij] != 0:
            for L, length in zip(Ls, lengths):
                if dist(L, (qx, qy), length) * scale > 2:
                    intensity = intensity + 0
                else:
                    intensity = intensity + dI((qx, qy), L, (0, 1), N, u)
                # intensity = intensity + dI((qx, qy), L, (0, 1), N, u)
        overlap_raster[ij] = intensity * raster[ij]
    return datetime.now() - now


def dOverlap_dTheta(polygon, raster, overlap_raster, offset, scale):
    now = datetime.now()
    Ls = Analytic.get_line_list(polygon=polygon)

    assert (len(raster.shape) == 3 and raster.shape == overlap_raster.shape)
    lengths = np.array([get_length(L) for L in Ls])

    u = create_random_samples(N)
    lim = (0, 1)
    for ij in np.ndindex(raster.shape[:2]):
        q = ij[0] / scale + offset[0], ij[1] / scale + offset[1]
        intensity = 0
        if raster[ij] != 0:
            for L, length in zip(Ls, lengths):
                intensity = intensity + dKernel1_dTheta(q, L, length, lim, u, N) + \
                            dKernel2_dTheta(q, L, length, lim, u, N) + \
                            dKernel3_dTheta(q, L, length, lim, u, N)

        overlap_raster[ij] = intensity * raster[ij]
    return datetime.now() - now


def dKernel1_dTheta(q, L, length, lim, u, n):
    qx, qy = q
    ax, ay = L[0]
    bx, by = L[1]

    if np.abs(dist(L, q, length)) > sigma:
        if by != ay:
            return Analytic.line_integ_HD(L, qx, qy) * (bx - ax) / (by - ay)
        else:
            return 0
    else:
        def x(t):
            return ax + (bx - ax) * t

        def y(t):
            return ay + (by - ay) * t

        def dy(t):
            return by - ay

        def dx_dtheta(t):
            return -y(t)

        def f(t):
            return D(x(t) - qx) * D(y(t) - qy) * dx_dtheta(t) * dy(t)

        fv = np.vectorize(f)
        a = lim[0]
        b = lim[1]
        ufv = fv(a + (b - a) * u)
        return ((b - a) / n) * ufv.sum()


def dKernel2_dTheta(q, L, length, lim, u, n):
    qx, qy = q
    ax, ay = L[0]
    bx, by = L[1]

    if np.abs(dist(L, q, length)) > sigma:
        return 0
    else:
        def x(t):
            return ax + (bx - ax) * t

        def y(t):
            return ay + (by - ay) * t

        def dy(t):
            return by - ay

        def dy_dtheta(t):
            return x(t)

        def f(t):
            return (-1/sigma2) * H(x(t) - qx) * D(y(t) - qy) * (y(t) - qy) * dy_dtheta(t) * dy(t)

        fv = np.vectorize(f)
        a = lim[0]
        b = lim[1]
        ufv = fv(a + (b - a) * u)
        return ((b - a) / n) * ufv.sum()


def dKernel3_dTheta(q, L, length, lim, u, n):
    qx, qy = q
    ax, ay = L[0]
    bx, by = L[1]

    if np.abs(dist(L, q, length)) > sigma:
        return
    else:
        def x(t):
            return ax + (bx - ax) * t

        def y(t):
            return ay + (by - ay) * t

        def ddy_dtheta(t):
            return by - ay

        def f(t):
            return H(x(t) - qx) * D(y(t) - qy) * ddy_dtheta(t)

        fv = np.vectorize(f)
        a = lim[0]
        b = lim[1]
        ufv = fv(a + (b - a) * u)
        return ((b - a) / n) * ufv.sum()


def Rotate(p, theta):
    R = Art.Rotate(np.deg2rad(theta))
    p.apply(R)


def main():
    '''Creating two Polygon. One is rotated by -1 degrees'''
    p1 = Art.Polygon([(1, 1), (1, -1), (-1, -1), (-1, 1)])
    angle = -1
    Rotate(p1, angle)
    p2 = Art.Polygon([(1, 1), (1, -1), (-1, -1), (-1, 1)])

    '''Defines the quarry area.'''
    qr = [[-2, -2], [2, 2]]
    ''' defines the density of pixel in each unit length. E.g. if scale is 1 then one unit contains one pixel.
    If the scale is 25 then one unit will contain 25 pixels. Need less to say, higher the scale more the computation.'''
    scale = 25

    raster, offset = Analytic.get_raster_map(qr=qr, scale=scale)

    time = area(p2, raster, offset, scale)
    print("Delta Time for Rasterizing poly2 :={}".format(time))


    '''Interval at which the overlap will be calculated.'''
    rate = 0.05
    args = []
    '''Pack the arguments to be mapped to different processes and run them in parallel'''
    for i in range(0, 40):
        '''Create an empty numpy matrix'''
        overlap_raster, _ = Analytic.get_raster_map(qr=qr, scale=scale)
        args.append((p1.copy(), raster, overlap_raster, offset, scale, angle))
        Rotate(p1, rate)
        angle = angle + rate

    '''Using 8 processes'''
    now = datetime.now()
    pool = multiprocessing.Pool(processes=8)
    results = pool.map(mp_overlap_poly_raster, args)
    print("Time := {}".format(datetime.now() - now))

    pool.close()
    pool.join()

    xs = [p[-1] for p in args]
    ys = []
    d = Art.Draw(height=3, width=4, scale=1)
    for p, r in results:
        ''' Uncomment the following line to see the change in overlap visually.'''
        # d.add_art(p2)
        # d.add_art(p)
        # d.add_raster(r, offset, scale)
        # d.draw()
        # d.flush()
        ys.append(np.sum(r))

    plt.plot(xs, ys)
    plt.show()


main()