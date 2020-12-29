import numpy as np
import datetime


def gaussian(x, sigma, mu):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.))) / (np.sqrt(2 * np.pi) * sigma)



def sigmoid(qx, gamma):
    def f(x):
        return 1/(1 + np.exp(-(x-qx)/gamma))
    return f


def Heavy(qx):
    def f(x):
        return 0 if x < qx else (1/2 if x==qx else 1)
    return f
    # return sigmoid(qx, 0.01)


def DDelta(qy, sigma):
    def f(y):
        return gaussian(y, sigma=sigma, mu=qy)

    return f


def create_random_samples(n):
    d = 1000
    subsets = np.arange(0, n + 1, n / d)
    u = np.zeros(n)
    for i in range(d):
        start = int(subsets[i])
        end = int(subsets[i + 1])
        u[start:end] = np.random.uniform(low=i / d, high=(i + 1) / d, size=end - start)
    np.random.shuffle(u)
    return u


def mc_unifrom_integration(f, lim, n, u=None):
    u = u if u.any() else create_random_samples(n)

    a = lim[0]
    b = lim[1]
    fv = np.vectorize(f)
    u_func = fv(a + (b - a) * u)
    s = ((b - a) / n) * u_func.sum()
    return s


def get_line_list(polygon):
    Ls = []
    l = len(polygon.points)
    if l > 0:
        p = polygon.points[0]
        for i in range(1, l):
            Ls.append([p, polygon.points[i]])
            p = polygon.points[i]
    if polygon.isClosed:
        Ls.append([polygon.points[l - 1], polygon.points[0]])
    return Ls


def area(polygon, qr):
    n = 10000
    u = create_random_samples(n)
    def I(q, L):
        sigma = 0.001
        ax, ay = L[0]
        bx, by = L[1]

        def h(t):
            return Heavy(q[0])(ax + (bx - ax) * t)

        def delta(t):
            return DDelta(q[1], sigma)(ay + (by - ay) * t)

        def f(t):
            v = h(t) * delta(t) * (by - ay)
            return v

        integ = mc_unifrom_integration(f, lim=(0, 1), n=10000, u=u)
        # print("I({}, {}) := {}".format(q, L, integ))
        return integ

    Ls1 = get_line_list(polygon)

    a = 0
    print("Calculating Area in the region : {}".format(qr))
    print("Polygon : {} ".format(Ls1))

    scale = 1
    anp = np.zeros(((qr[1][0] - qr[0][0])*scale, (qr[1][1] - qr[0][1])*scale, 1))
    offset = (qr[0][0], qr[0][1])
    print(anp.shape)
    total = anp.shape[0]*anp.shape[1]
    T_now = datetime.datetime.now()
    t = T_now

    for ij in np.ndindex(anp.shape[:2]):
        qx, qy = ij[0]/scale + offset[0], ij[1]/scale + offset[1]
        k1 = np.array([(I((qx, qy), L)) for L in Ls1])
        v = np.abs(sum(np.sign(k1)))
        anp[ij] = v
        a = a + v

        # Debug print
        if (ij[0]*anp.shape[1] + ij[1]) % anp.shape[1] ==0 :
            t = datetime.datetime.now() - t
            print("\tDelta T:{}, epoch : x.y := {} ".format(t, ij))
            t=datetime.datetime.now()

    print("Total Time := {}".format(datetime.datetime.now() - T_now))
    return a/scale, anp, offset, scale


def I(L, q, u, n, sigma):
    ax, ay = L[0]
    bx, by = L[1]
    qx, qy = q

    dyt = (by - ay)

    def x(t):
        return ax + (bx - ax) * t

    def y(t):
        return ay + (by - ay) * t

    def f(t):
        return np.heaviside(x(t) - qx, 0.5) * gaussian(y(t), sigma, qy) * dyt

    return mc_unifrom_integration(f, lim=(0, 1), n=n, u=u)

def overlap(polygon1, polygon2, qr):
    n = 10000
    u = create_random_samples(n)
    sigma = 0.001

    Ls1 = get_line_list(polygon1)
    Ls2 = get_line_list(polygon2)

    scale = 4
    anp = np.zeros(((qr[1][0] - qr[0][0])*scale, (qr[1][1] - qr[0][1])*scale, 1))
    offset = (qr[0][0], qr[0][1])
    print("\tCalculating overlap Area in the region : {}, Space {}".format(qr, anp.shape))

    T_now = datetime.datetime.now()
    t = T_now
    a = 0

    for ij in np.ndindex(anp.shape[:2]):
        qx, qy = ij[0]/scale + offset[0], ij[1]/scale + offset[1]
        k1 = [np.sign(I(L, (qx, qy), u, n, sigma)) for L in Ls1]
        if sum(k1) == 0:
            k2 = np.zeros(len(Ls2))
        else:
            k2 = [np.sign(I(L, (qx, qy), u, n, sigma)) for L in Ls2]

        v = np.abs(sum(k1)*sum(k2))
        anp[ij] = v
        a = a + v

        # Debug print
        if ij[0]*2 == anp.shape[0] and ij[1]*2 == anp.shape[1] :
            t = datetime.datetime.now() - t
            print("\t\tMid point Delta T:{}, epoch : x.y := {} ".format(t, ij))
            t=datetime.datetime.now()

    print("Total Time := {}".format(datetime.datetime.now() - T_now))
    return a/scale, anp, offset, scale

def Diff(L, q, u, n, sigma):
    ax, ay = L[0]
    bx, by = L[1]
    qx, qy = q

    dyt = (by - ay)
    dxt = (bx - ax)

    def x(t):
        return ax + (bx - ax) * t

    def y(t):
        return ay + (by - ay) * t

    def D1(t):
        return gaussian(x(t), sigma, qx) * (-y(t)) * gaussian(y(t), sigma, qy) * dyt, True

    def D2(t):
        yt = y(t)
        if yt == qy:
            return 0, False
        else:
            return -np.heaviside(x(t) - qx, 0.5) * gaussian(y(t), sigma, qy) * x(t) * dyt / (yt - qy) , True

    def D3(t):
        return np.heaviside(x(t) - qx, 0.5) * gaussian(y(t), sigma, qy) * dxt, True

    def f(t):
        d2 = D2(t)
        return D1(t)[0] + d2[0] + D3(t)[0] if d2[1] else 0

    return mc_unifrom_integration(f, lim=(0, 1), n=n, u=u)


def grad_overlap(polygon1, polygon2, qr, scale, cache, isSet):
    n = 10000
    sigma = 0.001

    Ls1 = get_line_list(polygon1)
    Ls2 = get_line_list(polygon2)
    a = 0

    anp = np.zeros(((qr[1][0] - qr[0][0]) * scale, (qr[1][1] - qr[0][1]) * scale, 1))
    offset = (qr[0][0], qr[0][1])

    T_now = datetime.datetime.now()
    t = T_now
    print("\tCalculating gradient of overlap Area in the region: {}, Raster Space: {}".format(qr, anp.shape))

    u = create_random_samples(n)
    for ij in np.ndindex(anp.shape[:2]):
        qx, qy = ij[0]/scale + offset[0], ij[1]/scale + offset[1]
        if isSet:
            sK1 = cache[ij][0]
        else:
            sK1 = sum([np.sign(I(L, (qx, qy), u, n, sigma)) for L in Ls1])
            cache[ij] = sK1

        if sK1 == 0:
            K4 = np.zeros((len(Ls2)))
        else:
            K4 = np.array([Diff(L, (qx, qy), u, n, sigma) for L in Ls2])

        anp[ij] = np.abs(sK1)

        a = a + np.abs(sK1) * sum(K4)


        # Debug print
        if ij[0]*2 == anp.shape[0] and ij[1]*2 == anp.shape[1]:
            t = datetime.datetime.now() - t
            print("\t\tMid point Delta T:{}, epoch : x.y := {} ".format(t, ij))
            t = datetime.datetime.now()
            u = create_random_samples(n)


    print("\tTotal Time := {}".format(datetime.datetime.now() - T_now))
    return a/scale, anp, offset, scale


