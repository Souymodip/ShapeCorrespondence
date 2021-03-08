import torch as th
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import copy as cp
import random

import Art
import MatchMaker as MM
import testLevel4 as ts
import Procrutes
import D3Plot as d3
import CutMatch as cm
import Helpers


BITS=128
FINAL = 64
DISCRETE_STEP_DISTANCE=0.1
SIGMA=0.04
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-5

#
# def normalize(poly):
#     return Procrutes.remove_scale(Procrutes.remove_translation(poly))


def de_flatten(flatten_poly):
    poly = []
    if th.is_tensor(flatten_poly):
        flatten_poly = flatten_poly.detach().numpy()
    for i in range (int(len(flatten_poly)/2)):
        poly.append([flatten_poly[2*i], flatten_poly[2*i+1]])
    ret = np.array(poly)
    return ret


def flatten_poly(poly, disc_step=DISCRETE_STEP_DISTANCE, bits=BITS):
    new_poly = [poly[0][0], poly[0][1]]
    i=1
    count = 1
    while i < len(poly) and count < bits:
        p_ = poly[i-1]
        p = poly[i]
        d = np.linalg.norm(p - p_)

        if d < disc_step:
            i = i + 1
            continue

        fraction = disc_step/d
        while count < bits and fraction <= 1:
            q = (1-fraction)*p_ + fraction*p
            new_poly.append(q[0])
            new_poly.append(q[1])
            count = count + 1
            fraction = fraction + disc_step/d
        if count >= bits:
            break
        i = i + 1
    last2, last = new_poly[-2], new_poly[-1]
    while count < bits:
        new_poly.append(last2)
        new_poly.append(last)
        count = count + 1

    return np.array(new_poly)


def rotate_point(p, theta):
    x = p[0] * np.cos(theta) - p[1] * np.sin(theta)
    y = p[0] * np.sin(theta) + p[1] * np.cos(theta)
    return np.array([x, y])


def rotate_poly(poly):
    theta = np.random.rand() * 2 * np.pi
    return np.array([rotate_point(p, theta) for p in poly])


def noisefy(poly):
    N = len(poly)
    noise = np.random.normal(0, SIGMA, size=(N,2))
    return poly + noise


def flip(poly):
    return np.flip(poly, axis=0)


class autoencoder(nn.Module):
    def __init__(self, dimension=2*BITS, final=FINAL):
        super(autoencoder, self).__init__()
        self.dimension = dimension
        self.final = final
        self.encode = nn.Sequential(
            nn.Linear(self.dimension, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Linear(128, self.final)
        )
        self.decode = nn.Sequential(
            nn.Linear(self.final, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, self.dimension)
        )

    def forward(self, x):
        code = self.encode(x)
        reconstructed = self.decode(code)
        return reconstructed


def get_last_model():
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(Helpers.bcolors.HEADER + "DEVICE : {}".format(device) + Helpers.bcolors.ENDC)
    model = autoencoder(dimension=2*BITS).to(device)
    return Helpers.load_last_model(model.dimension, model.final, model)


def get_model():
    model, isLoaded = get_last_model()
    print(model)

    optimizer = th.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.L1Loss()
    return model, criterion, optimizer


@Helpers.timer
def train(train_set, batch_size, num_epochs, sample_rate):
    model, criterion, optimizer = get_model()

    print(Helpers.bcolors.HEADER + "Train set size:{}, batch_size: {}, Epochs:{}, "
                                   "Reconstruction samples:{}"
          .format(len(train_set), batch_size, num_epochs, int(num_epochs/sample_rate)) + Helpers.bcolors.ENDC)

    cache_reconstructions = []

    train_loader = th.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        epoch_loss = 0
        sampled = False
        for n, (samples, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(samples)
            train_loss = criterion(output, labels)
            train_loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + train_loss.item()
            if not sampled and (epoch + 1) % sample_rate == 0:
                cache_reconstructions.append((epoch, output[-1], labels[-1]))
                sampled = True

        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, epoch_loss/len(train_loader)))

    Helpers.save_model_state(model)
    return cache_reconstructions, model


def generate_test_case(poly, N):
    data = []
    poly = Procrutes.normalize(poly)
    p_f = th.Tensor(flatten_poly(poly))
    for i in range(int(N/4)):
        p = th.Tensor(flatten_poly(rotate_poly(poly)))
        data.append((p, p_f))

    for i in range(int(N /4)):
        p = th.Tensor(flatten_poly(noisefy(poly)))
        data.append((p, p_f))

    for i in range(int(N /4)):
        p = th.Tensor(flatten_poly(flip(poly)))
        data.append((p, p_f))

    for i in range(int(N/4)):
        data.append((p_f, p_f))
    return data


@Helpers.timer
def build_data_set(cut_length, stride, num_data_per_poly):
    '''Creating data set'''
    print(Helpers.bcolors.HEADER +  "Creating data set. Parameters cut_length :{}, Stride :{}, num of data per poly : "
                                    "{}".format(cut_length, stride, num_data_per_poly) + Helpers.bcolors.ENDC)

    def add_arts(arts):
        mm = MM.MatchMaker(100)
        ids = [mm.add_art(art) for art in arts]
        print("\t Adding a new art group of size {} to data set".format(len(ids)))
        count = 0
        ds = []
        for id in ids:
            length = mm.get_length(id)
            poly = mm.get_poly(id)
            st, cl = cm.get_relative(length, stride, cut_length)
            count_in = 0

            for x in np.arange(0, length, st):
                count_in = count_in + 1
                cut = cm.get_cut(poly, x, cl)
                ds = ds + generate_test_case(cut.poly, N=num_data_per_poly)
            count = count + 1
            print("\t\t Finished {}/{}: Added {} entries".format(count, len(ids), count_in))
        return ds

    as1 = ts.get_test(0)
    as2 = ts.get_test(2)
    as3 = ts.get_test(3)
    as4 = ts.get_test(4)

    # data_set = []
    data_set = add_arts(as1)
    data_set = data_set + add_arts(as2)
    data_set = data_set + add_arts(as3)
    data_set = data_set + add_arts(as4)

    return data_set


@Helpers.timer
def find_match(encode):
    def diff (p1, p2):
        p1 = Procrutes.normalize(p1)
        p2 = Procrutes.normalize(p2)
        f1 = th.Tensor(flatten_poly(p1))
        f2 = th.Tensor(flatten_poly(p2))
        return th.linalg.norm(encode(f1) - encode(f2))

    mm = MM.MatchMaker(100)
    arts = ts.get_test(0)
    id1, id2 = mm.add_art(arts[1]), mm.add_art(arts[5])
    p1, p2 = mm.get_poly(id1), mm.get_poly(id2)

    c = cm.Cut_Match(p1, p2, diff, stride=5, cut_length=30)
    c.cut_match()


def get_Neural_encoder_diff():
    model, isLoaded = get_last_model()
    if not isLoaded:
        return None
    else:
        def diff (p1, p2):
            p1 = Procrutes.normalize(p1)
            p2 = Procrutes.normalize(p2)
            f1 = th.Tensor(flatten_poly(p1))
            f2 = th.Tensor(flatten_poly(p2))
            return th.linalg.norm(model.encode(f1) - model.encode(f2))
        return diff


def show_reconstruction(rc):
    assert len(rc) > 0
    p = rc[-1][-1]
    print (p.shape)
    ps = [Art.Polygon(de_flatten(p), isClosed=False)]

    for e, p_, _ in rc:
        print ("Reconstruction at {}".format(e))
        ps.append(Art.Polygon(de_flatten(p_), isClosed=False))

    d3.draw_polys(ps)


if __name__ == '__main__':
    train_set = build_data_set(cut_length=30, stride=5, num_data_per_poly=16)
    N = len(train_set)
    batch_size = 64
    epochs = int(N / (4 * batch_size))
    record_rate = int(epochs / 4)
    rc, model = train(train_set=train_set, batch_size=batch_size, num_epochs=epochs, sample_rate=record_rate)
    show_reconstruction(rc)
    find_match(model.encode)


def test_minimal():
    arts = ts.get_test(0)
    mm = MM.MatchMaker(100)
    id = mm.add_art(arts[0])
    poly = mm.get_poly(id)
    cut = cm.get_cut(poly, 30, 30)
    poly = cut.poly

    train_set = generate_test_case(poly, N=512*1)
    rc = train(train_set, batch_size=256, num_epochs=16, sample_rate=4)
    show_reconstruction(rc)

def test_test_case_gen(poly):
    test_cases = generate_test_case([poly], N=1024 * 4)

    ls = np.random.choice(len(test_cases), size=2)
    p, p1, p2 = test_cases[ls[0]][1], test_cases[ls[0]][0], test_cases[ls[1]][0]
    p, p1, p2 = Art.Polygon(de_flatten(p), isClosed=False), Art.Polygon(de_flatten(p1), isClosed=False), Art.Polygon(
        de_flatten(p2), isClosed=False)
    d3.draw_polys([p, p1, p2])



