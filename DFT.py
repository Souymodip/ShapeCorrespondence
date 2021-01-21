import numpy as np


def super_sample(y, x, samples=-1):
    assert (y.shape == x.shape)
    if samples <= 0:
        samples = 10 * x.shape[0]

    L = x[-1]
    ys, xs = np.zeros((samples)), np.zeros((samples))
    j = 0
    for i in range(samples):
        xs[i] = L * (i / (samples - 1))
        while j < x.shape[0] - 1:
            if x[j] < xs[i] and x[j + 1] <= xs[i]:
                j = j + 1
            else:
                break
        ys[i] = y[j]

    # fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle('Vertically stacked subplots')
    # ax1.plot(x, y)
    # ax2.plot(xs, ys)
    # plt.show()

    return ys, xs


def DFT_SIG(signal, xs, dimensions):
    y, x = super_sample(signal, xs)

    A = np.zeros(shape=(dimensions))
    l = np.zeros(x.shape)

    for i in range(1, x.shape[0]):
        l[i] = l[i - 1] + x[i]

    L = l[-1]
    if L==0:
        print (np.max(xs), xs[-1])
        assert(l != 0)
    for i in range(dimensions):
        n = i + 1
        k = (2 * np.pi * n / L)
        an = - np.sum(y * np.sin(k * l)) / (np.pi * n)
        bn = np.sum(y * np.cos(k * l)) / (np.pi * n)
        A[i] = np.linalg.norm(np.array([an, bn]))
    return A


