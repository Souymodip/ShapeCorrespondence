import torch as th
from torch import nn
import Generator as gen
import numpy as np
import copy as cp
import testsLevel1 as testsuite
import ShapeSimilarity as ss
import FunctionSimilarity as fs
import Art


th.manual_seed(111)
np.random.seed(111)


def perturb(f, p=0.8):
    p_f = cp.deepcopy(f)
    if np.random.rand() < p:
        p_f = gen.perturb_x(p_f, 0.9, 0.5)

    if np.random.rand() < p:
        p_f = gen.perturb_y(p_f, 0.9, 1)

    if np.random.rand() < 0.9:
        p_f = gen.cycle(p_f)

    return p_f


def flatten(f):
    yf, xf = f
    data = []
    for i in range(xf.size):
        data.append(xf[i])
        data.append(yf[i])
    return data


def get_train_set(f, N=2048, debug=False):
    y, x = f
    assert (x.size == y.size)
    f_dim = x.size
    train_data = th.zeros((N, 2*f_dim)) # N rows of sample. Each row contains pair of y_i, x_i of variation of f

    for i in range(N):
        f_ = perturb(f, p=0.9)
        train_data[i] = th.Tensor(flatten(f_))
        if debug:
            k = int(N/3)
            if i % k == 0:
                fs.draw_graph([f, f_])

    train_labels = th.ones(N)
    train_set = [
        (train_data[i], train_labels[i]) for i in range(N)
    ]
    return train_set


def train(net, train_set, num_epoch, batch_size, generator):
    train_loader = th.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    def build_samples(real_samples, batch_size, generator):
        real_samples_labels = th.ones((batch_size, 1))
        rand_samples = generator(batch_size)
        rand_samples_labels = th.zeros((batch_size, 1))
        return th.cat((real_samples, rand_samples)), th.cat((real_samples_labels, rand_samples_labels))

    def train_net(net, samples, labels):
        net.zero_grad()
        out_net = net(samples)
        loss_net = net.loss_function(out_net, labels)
        loss_net.backward()
        net.optimize()
        return loss_net

    for epoch in range(num_epoch):
        for n, (real_samples, _) in enumerate(train_loader):
            samples, labels = build_samples(real_samples, batch_size, generator)
            loss_discriminator = train_net(net, samples, labels)
            # Show loss
            if epoch % 10 == 0 and n == batch_size - 1:
                print("Epoch: {} Loss D.: {}".format(epoch, loss_discriminator))

    return net


class Discriminator(nn.Module):
    def __init__(self, f_dim, lr):
        super().__init__()
        self.dim = 2*f_dim
        self.lr = lr
        self.loss_function = nn.BCELoss()

        self.model = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

    def optimize(self):
        th.optim.Adam(self.parameters(), lr=self.lr).step()


def train_for(f):
    y, x = f
    assert (x.size == y.size)
    f_dim = x.size
    print("Creating Discriminator")
    discriminator = Discriminator(f_dim=f_dim, lr=0.001)

    print("Building Training Set")
    train_set = get_train_set((y, x), N=1024*32, debug=False)

    num_epochs = 20
    batch_size = 64

    def rand_generator(size):
        data = [flatten(gen.random(f_dim=f_dim, low_y=np.min(y), high_y=np.max(y))) for i in range(size)]
        return th.Tensor(data)

    print("Training ...")
    discriminator = train(net=discriminator, train_set=train_set, num_epoch=num_epochs, batch_size=batch_size,
                          generator=rand_generator)
    return discriminator


def test(discriminator, f):
    y, x = f
    print("Test ...")
    t = th.Tensor(flatten((y, x)))
    out = discriminator(t)
    print(out.detach())
    return out


def main():
    def get_f():
        d = Art.Draw()
        art1, art2 = testsuite.get_test(4)
        d.add_art(art1)
        # d.add_art(art2)
        _, art3 = testsuite.get_test(5)
        _, art4 = testsuite.get_test(6)
        _, art5 = testsuite.get_test(7)
        art7, art6 = testsuite.get_test(3)
        # d.add_art(art3)
        # d.add_art(art4)
        d.add_art(art5)
        # d.add_art(art6)
        # d.add_art(art7)
        d.draw()

        def get(art):
            importance_angle = 15
            polygon = ss.piecewise_bezier_to_polygon(art=art)
            n_p = ss.squint(polygon, True, np.deg2rad(importance_angle))
            a1, d1 = ss.poly_to_turn_v_length(n_p, closed=True)
            d1 = d1/d1[-1]
            return a1, d1

        # return [ get(art6), get(art7)]
        return [get(art1), get(art2), get(art3), get(art4), get(art5), get(art6), get(art7)]

    l = get_f()
    y1, x1 = l[0]
    y2, x2 = l[4]

    m_y, m_x = fs.merge(y1, x1, y2, x2)

    y1 = np.array([y[0] for y in m_y])
    y2 = np.array([y[1] for y in m_y])
    assert (y1.size == y2.size == m_x.size)

    discriminator = train_for((y1, m_x))

    print("With Self: expectating perfect match")
    test(discriminator, (y1, m_x))

    print("\nWith Random Noice: expectating perfect mis-match")
    test(discriminator, gen.random(m_x.size, np.min(y1), np.max(y1)))

    print("\nWith non-affine transform: Expectating high confidence of match")
    test(discriminator, (y2, m_x))
    test(discriminator, perturb((y1, m_x)))

    fs.draw_graph([(y1, m_x), (y2, m_x)])


if __name__ == '__main__':
    main()