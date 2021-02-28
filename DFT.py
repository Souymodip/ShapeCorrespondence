import numpy as np
import FunctionTransform as ft
import testsLevel1 as testsuite
import FunctionSimilarity as fs
import Art


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


def sample(y, x, interval):
    f_x = 0.0
    f_y = [y[0]]

    index = 1
    while index < len(x):
        f_x = f_x + interval
        if x[index - 1] < f_x < x[index]:
            a_, a = x[index-1], x[index]
            b_, b = y[index-1], y[index]
            f_y.append(ft.y_at(b_, b, a_, a, f_x))
        elif x[index-1] == f_x:
            f_y.append(y[index-1])
        elif x[index] == f_x:
            f_y.append(y[index])
        elif f_x > x[index]:
            def next(curr, x_val):
                while (curr < len(x)) and (x[curr] < x_val):
                    curr = curr + 1
                return curr
            index = next(index, f_x)
            if index < len(x):
                a_, a = x[index - 1], x[index]
                b_, b = y[index - 1], y[index]
                f_y.append(ft.y_at(b_, b, a_, a, f_x))
    return np.array(f_y)


def DFT(f, n, debug=False):
    y, x = f
    assert (n > 0 and x[0] == 0.0)
    interval = 1 / n
    f_y = sample(y, x, interval)
    if debug:
        ft.draw_graph([(f_y, np.arange(0.0, 1.0 + interval, interval)), f])
    return np.fft.fft(f_y)


def low_pass_filter(f, sample, filter):
    df = DFT(f, sample)
    assert(len(df) % 2 == 1)
    l = int(len(df)/2)
    for i in range(l):
        if i > filter:
            df[i] = 0.0
            df[l + i] = 0.0
    idf = np.fft.ifft(df)
    t = np.arange(0, 1, 1 / len(idf))
    return np.absolute(idf), t


def DFT_diff(art1, art2, sampling_rate=2, debug=False):
    f1, f2 = ft.art_to_function(art1, importance_angle=0.0), ft.art_to_function(art2, importance_angle=0.0)

    ys, x = fs.merge(f1[0], f1[1], f2[0], f2[1])
    m_y1 = np.array([y[0] for y in ys])
    m_y2 = np.array([y[1] for y in ys])
    m_f1 = m_y1, x
    m_f2 = m_y2, x

    dft1 = DFT(m_f1, sampling_rate * len(f1[0]), debug=False)
    dft2 = DFT(m_f2, sampling_rate * len(f1[0]), debug=False)

    comps = int(len(dft1)/2)
    imp_dft_comp1 = np.absolute(dft1)[:comps]
    imp_dft_comp2 = np.absolute(dft2)[:comps]
    if debug:
        ft.draw_graph([(np.absolute(dft1), np.array(range(len(dft1)))),
                       (np.absolute(dft2), np.array(range(len(dft2))))])

    return np.linalg.norm(imp_dft_comp1 - imp_dft_comp2)


def test(art1, r=30):
    poly1 = ft.piecewise_bezier_to_polygon(art1)
    poly2 = ft.roll(poly1, r)

    f1 = ft.poly_to_turn_v_length(poly1, closed=True)
    f2 = ft.poly_to_turn_v_length(poly2, closed=True)
    print(len(f1[0]), len(f2[0]))
    ys, x = fs.merge(f1[0], f1[1], f2[0], f2[1])

    m_y1 = np.array([y[0] for y in ys])
    m_y2 = np.array([y[1] for y in ys])

    m_f1 = m_y1, x
    m_f2 = m_y2, x
    print(len(x))
    ft.draw_graph([m_f1, m_f2])

    dft1 = DFT(m_f1, 2 * len(f1[0]), debug=False)
    dft2 = DFT(m_f2, 2 * len(f1[0]), debug=False)
    print(len(dft1), len(dft2))
    ft.draw_graph([(np.absolute(dft1), np.array(range(len(dft1)))),
                   (np.absolute(dft2), np.array(range(len(dft2))))])
    comp = int(len(dft1) / 2)
    imp1, imp2 = np.absolute(dft1)[:comp], np.absolute(dft2)[:comp]
    print(np.linalg.norm(imp1 - imp2))


def low_pass(fft, frac):
    data_count = len(fft)
    freq = np.fft.fftfreq(data_count)
    high_index = int(data_count/2)-1 if data_count % 2 == 0 else int(data_count/2)
    high_freq = np.abs(freq[high_index])
    for i in range(data_count):
        if np.abs(freq[i]) > high_freq * frac:
            fft[i] = 0.0 + 0.0j
    return fft


def poly_to_dft(poly, times):
    def sample(poly, times):
        assert times >= 1
        new_y, new_x = [], []
        for i in range(1, len(poly)):
            for j in np.arange(0, 1 + 1/times, 1/times):
                p = poly[i-1] *(1-j) + poly[i] * j
                new_y.append(p[1]), new_x.append(p[0])
        new_y.append(poly[-1][1]), new_x.append(poly[-1][0])
        return np.array(new_y), np.array(new_x)

    y, x = sample(poly, times)
    complex_series = x + 1j * y
    fft = np.fft.fft(complex_series)
    # data_count = len(complex_series)
    # freq = np.fft.fftfreq(data_count)
    # ft.draw_graph([(np.real(fft), freq), (np.imag(fft), freq)])
    return fft


def pad(poly, length):
    if len(poly) == length:
        return poly
    else:
        add = length - len(poly)
        add_inds = np.random.choice(len(poly), add, replace=True)
        for i in add_inds: # not as intended
            p_ = poly[i-1] if i > 0 else np.zeros((2))
            p = poly[i]
            new_p = 0.5 * p_ + 0.5 * p
            poly = np.insert(poly, i, new_p, axis=0)
        return poly


def shrink(poly, length):
    if len(poly) <= length:
        return poly
    else:
        remove = len(poly) - length
        diff = [(np.linalg.norm(poly[i-1] -  poly[i]), i) for i in range(1, len(poly))]
        s_diff = sorted(diff, key=lambda x: x[0])[:remove]
        remove_indexes = set([s[1] for s in s_diff])
        new_poly = [poly[i] for i in range(len(poly)) if i not in remove_indexes]
        return np.array(new_poly)


def diff_poly(poly1, poly2, frac=0.5):
    data_count = min(len(poly1), len(poly2))
    poly1, poly2 = shrink(poly1, data_count), shrink(poly2, data_count)

    fft1 = poly_to_dft(poly1, 2)
    fft2 = poly_to_dft(poly2, 2)

    l_fft1 = low_pass(fft1, frac)
    l_fft2 = low_pass(fft2, frac)
    diff = l_fft1 - l_fft2
    return np.linalg.norm(np.real(diff), ord=1), np.linalg.norm(np.imag(diff), ord=1)








if __name__ == '__main__':
    art1, art2 = testsuite.get_test(5)
    d = Art.Draw()
    d.add_art(art1), d.add_art(art2)
    d.draw()
    print (DFT_diff(art1, art2, 2, False))










