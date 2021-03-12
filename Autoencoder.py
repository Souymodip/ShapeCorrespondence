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


BITS=256
FINAL = 64
DISCRETE_STEP_DISTANCE=0.1
SIGMA=0.05
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-5


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


def rotate_point_at(p, pivot, theta):
    p = p - pivot
    x = p[0] * np.cos(theta)[0] - p[1] * np.sin(theta)[0]
    y = p[0] * np.sin(theta)[0] + p[1] * np.cos(theta)[0]
    p = np.array([x, y])
    return p + pivot


def rotate_poly(poly):
    theta = np.random.rand() * 2 * np.pi
    return np.array([rotate_point(p, theta) for p in poly])


def noisefy(poly):
    N = len(poly)
    noise = np.random.normal(0, SIGMA, size=(N,2))
    return poly + noise


def flip(poly):
    return np.flip(poly, axis=0)


class Deformer:
    def __init__(self, poly):
        self.poly = poly
        self.N = len(self.poly)
        self.length = 0
        for i in range(1, self.N):
            self.length = self.length + np.linalg.norm(self.poly[i-1] - self.poly[i])

    def get_window(self):
        l = np.random.randint(0, self.N, 1)[0]
        r = np.random.randint(l, min(self.N, l + int(self.N / 3)), 1)[0]
        if r < l + 1:
            l = np.random.randint(0, self.N, 1)[0]
            r = np.random.randint(l, min(self.N, l + int(self.N / 3)), 1)[0]
        return np.array([l, r])

    def rotate(self):
        theta = np.random.rand(1) * 2 * np.pi
        pivot = np.mean(self.poly, axis=0)
        return np.array([rotate_point_at(p, pivot, theta) for p in self.poly])

    def bend(self):
        window = self.get_window()
        if window[1] < window[0] + 2:
            return self.poly
        poly = cp.deepcopy(self.poly)
        sub = self.poly[window[0]:window[1]]
        pivot = np.mean(sub, axis=0)
        angle = np.random.rand(1) * np.pi / 6
        for i in range(window[0], window[1]):
            poly[i] = rotate_point_at(self.poly[i], pivot, angle)
        return poly

    def elongate(self):
        window = self.get_window()
        if window[1] < window[0] + 2:
            return self.poly

        def angle(p) : return np.arctan(p[1]/p[0]) if np.abs(p[0]) > 1e-5 else np.pi/2
        gr = np.array([angle(self.poly[i] - self.poly[i-1]) for i in range(window[0]+1, window[1])])
        gr_avg = np.mean(gr, axis=0) + np.pi/2
        distance = np.random.rand() * self.length * 0.01
        vec = distance * np.cos(gr_avg) + distance * np.sin(gr_avg)
        poly = cp.deepcopy(self.poly)
        for i in range(window[0], window[1]):
            poly[i] = self.poly[i] + vec
        return poly

    def remove_features(self):
        threshold = 0.003 * self.length
        poly = cp.deepcopy(self.poly)
        poly = np.array([poly[i] for i in range(self.N - 1) if np.linalg.norm(poly[i] -poly[i+1]) > threshold])
        return poly

    def add_noise(self):
        window = self.get_window()
        if window[1] < window[0]:
            return self.poly
        poly = cp.deepcopy(self.poly)
        poly[window[0]: window[1]] = noisefy(self.poly[window[0]: window[1]])
        return poly

    def flip(self):
        return np.flip(self.poly, axis=0)


class TestGenerator:
    def __init__(self, cut_length, stride, num_data_per_poly):
        self.cut_length = cut_length
        self.stride = stride
        self.num_data_per_poly = num_data_per_poly

        # as1 = ts.get_test(0)
        as2 = ts.get_test(2)
        # as3 = ts.get_test(3)
        as4 = ts.get_test(4)
        self.art_suits = [as2, as4]
        self.indexes = []
        self.data_set = self.get()

    def generate_test_case(self, poly):
        data = []
        poly = Procrutes.normalize(poly)
        p_f = th.Tensor(flatten_poly(poly))
        deformer = Deformer(poly)

        for i in range(int(np.ceil(self.num_data_per_poly / 5))):
            p = th.Tensor(flatten_poly(deformer.elongate()))
            data.append((p, p_f))

        for i in range(int(np.ceil(self.num_data_per_poly / 5))):
            p = th.Tensor(flatten_poly(deformer.bend()))
            data.append((p, p_f))

        for i in range(int(np.ceil(self.num_data_per_poly / 5))):
            p = th.Tensor(flatten_poly(deformer.flip()))
            data.append((p, p_f))

        # for i in range(int(np.ceil(self.num_data_per_poly /  5))):
        #     p = th.Tensor(flatten_poly(deformer.add_noise()))
        #     data.append((p, p_f))

        for i in range(int(np.ceil(self.num_data_per_poly /  5))):
            p = th.Tensor(flatten_poly(deformer.rotate()))
            data.append((p, p_f))

        for i in range(int(np.ceil(self.num_data_per_poly /  20))):
            data.append((p_f, p_f))

        return data

    @Helpers.timer
    def get(self):
        """Creating data set"""
        print(
            Helpers.bcolors.HEADER + "Creating data set. Parameters cut_length :{}, Stride :{}, num of data per poly : "
                                     "{}".format(self.cut_length, self.stride, self.num_data_per_poly) + Helpers.bcolors.ENDC)

        def add_arts(arts):
            mm = MM.MatchMaker(100)
            ids = [mm.add_art(art) for art in arts]
            print("\t Adding a new art group of size {} to data set".format(len(ids)))
            count = 0
            ds = []
            for id in ids:
                length = mm.get_length(id)
                poly = mm.get_poly(id)
                st, cl = cm.get_relative(length, self.stride, self.cut_length)
                count_in = 0

                for x in np.arange(0, length, st):
                    count_in = count_in + 1
                    cut = cm.get_cut(poly, x, cl)
                    test = self.generate_test_case(cut.poly)
                    ds = ds + test

                    if len(self.indexes) == 0:
                        self.indexes.append(len(test))
                    else:
                        self.indexes.append(self.indexes[-1] + len(test))
                count = count + 1
                print("\t\t Finished {}/{}: Added {} entries".format(count, len(ids), count_in))
            return ds

        self.data_set = []
        for aset in self.art_suits:
            self.data_set = self.data_set + add_arts(aset)

        return self.data_set

    def get_train_set_for_encoder(self):
        return self.data_set


    @Helpers.timer
    def get_train_set_for_discriminator(self, N, encoder):
        N = int(np.ceil(N/2))
        def grp_index(ind):
            return np.searchsorted(self.indexes, ind)
        l = np.sort(np.random.choice(len(self.data_set), N))
        r = np.sort(np.random.choice(len(self.data_set), N))
        train_set = []
        label_1_count = 0
        label_0_count = 0
        for i,j in zip(l,r):
            p1, p2 = self.data_set[i]
            q1, q2 = self.data_set[j]
            if encoder:
                p1, p2 = encoder(p1).detach(), encoder(p2).detach()
                q1, q2 = encoder(q1).detach(), encoder(q2).detach(),

            one = th.ones(1)
            zero = th.zeros(1)
            if grp_index(i) == grp_index(j):
                train_set.append((th.cat((p2, q2)), one))
                train_set.append((th.cat((p1, q1)), one))
                train_set.append((th.cat((p1, q2)), one))
                train_set.append((th.cat((p2, q1)), one))
                label_1_count += 4
            else:
                train_set.append((th.cat((p2, q2)), zero))
                train_set.append((th.cat((p1, q1)), zero))
                label_0_count += 2

        print ("\tAdding test case for discriminator:\n"
               "\t\t Label :0 Count :{} , Label :1 Count :{}, Total : {}".format(label_0_count, label_1_count, label_1_count+label_0_count))
        return train_set


class Discriminator(nn.Module):
    def __init__(self, dimension, lr):
        super().__init__()
        self.dimension = dimension * 2
        self.lr = lr
        self.final = 1
        self.model = nn.Sequential(
            nn.Linear(self.dimension, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Autoencoder(nn.Module):
    def __init__(self, dimension=2*BITS, final=FINAL):
        super(Autoencoder, self).__init__()
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


def get_last_autoencoder_model():
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(Helpers.bcolors.HEADER + "DEVICE : {}".format(device) + Helpers.bcolors.ENDC)
    model = Autoencoder(dimension=2 * BITS).to(device)
    return Helpers.load_last_model(model.dimension, model.final, model, pre='M')


def get_last_discriminator_model():
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(Helpers.bcolors.HEADER + "DEVICE : {}".format(device) + Helpers.bcolors.ENDC)
    model = Discriminator(dimension=FINAL, lr=LEARNING_RATE).to(device)
    return Helpers.load_last_model(model.dimension, model.final, model, pre='D')


def get_autoencoder_model():
    model, isLoaded = get_last_autoencoder_model()
    print("------------------------------------------------------------------")
    print(model)
    print("------------------------------------------------------------------")
    optimizer = th.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.L1Loss()
    return model, criterion, optimizer


def get_discriminator_model():
    model, isLoaded = get_last_discriminator_model()
    print("------------------------------------------------------------------")
    print(model)
    print("------------------------------------------------------------------")
    optimizer = th.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()
    return model, criterion, optimizer


@Helpers.timer
def train_autoencoder(train_set, batch_size, num_epochs, sample_rate):
    model, criterion, optimizer = get_autoencoder_model()
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

        print("Training Encoder: epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, epoch_loss/len(train_loader)))

    Helpers.save_model_state(model, pre='M')
    return cache_reconstructions, model


@Helpers.timer
def train_discriminator(train_set_discriminate, batch_size, num_epochs):
    model, criterion, optimizer = get_discriminator_model()
    print(Helpers.bcolors.HEADER + "Train set size:{}, batch_size: {}, Epochs:{}"
          .format(len(train_set_discriminate), batch_size, num_epochs) + Helpers.bcolors.ENDC)

    train_loader = th.utils.data.DataLoader(train_set_discriminate, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for n, (samples, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(samples)
            #print("{}: batch: {} Sample's shape {}, {}, Output shape {}".format(n, batch_size, samples.shape, labels.shape, output.shape))
            train_loss = criterion(output, labels)
            train_loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + train_loss.item()

        print("Training discriminator epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, epoch_loss / len(train_loader)))
    Helpers.save_model_state(model, pre='D')
    return model


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
    model, isLoaded = get_last_autoencoder_model()
    dis, d_isLoaded = get_last_discriminator_model()
    if not isLoaded or not d_isLoaded:
        return None
    else:
        def diff (p1, p2):
            p1 = Procrutes.normalize(p1)
            p2 = Procrutes.normalize(p2)
            f1 = th.Tensor(flatten_poly(p1))
            f2 = th.Tensor(flatten_poly(p2))
            # e = th.cat((model.encode(f1), model.encode(f2)))
            # print (" D : ",dis(e).detach())
            return th.linalg.norm(model.encode(f1) - model.encode(f2))
            return dis(e).detach()[0]
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


def test_test_case_gen():
    arts = ts.get_test(0)
    mm = MM.MatchMaker(100)
    id = mm.add_art(arts[0])
    poly = mm.get_poly(id)

    deformer = Deformer(poly)

    p1 = deformer.elongate()
    p2 = deformer.bend()
    p3 = deformer.remove_features()
    p4 = deformer.add_noise()
    p5 = deformer.flip()

    d3.draw_polys([Art.Polygon(poly, isClosed=False), Art.Polygon(p1, isClosed=False), Art.Polygon(p2, isClosed=False),
                   Art.Polygon(p3, isClosed=False), Art.Polygon(p4, isClosed=False), Art.Polygon(p5, isClosed=False)])


def test_encoder():
    model, isLoaded = get_last_autoencoder_model()
    if not isLoaded:
        print ("Model not loaded!!")
        exit(0)
    else:
        arts = ts.get_test(0)
        mm = MM.MatchMaker(100)
        id = mm.add_art(arts[0])
        poly = mm.get_poly(id)

        cut = cm.get_cut(poly, 100, 20)
        r_poly = Deformer(cut.poly).rotate()
        p1 = Procrutes.normalize(cut.poly)
        p2 = Procrutes.normalize(r_poly)
        f1 = th.Tensor(flatten_poly(p1))
        f2 = th.Tensor(flatten_poly(p2))
        print(th.linalg.norm(model.encode(f1) - model.encode(f2)))
        e1 = model.encode(f1)
        e2 = model.encode(f2)

        g1, g2 = model.decode(e1), model.decode(e2)

        d3.draw_polys([Art.Polygon(de_flatten(g1), isClosed=False), Art.Polygon(de_flatten(g2), isClosed=False),
                       Art.Polygon(de_flatten(f1), isClosed=False), Art.Polygon(de_flatten(f2), isClosed=False)])


def test_test_gen2():
    tg  = TestGenerator(cut_length=25, stride=5, num_data_per_poly=50)
    model, isLoaded = get_last_autoencoder_model()
    if not isLoaded:
        print("Model not loaded!!")
        exit(0)
    tg.get_encoded_train_set(3000, model.encode)


def create_encoder_decoder(train_set_for_encoder):
    batch_size = 32
    epochs = 20
    record_rate = int(epochs / 5)
    rc, model = train_autoencoder(train_set=train_set_for_encoder, batch_size=batch_size, num_epochs=epochs, sample_rate=record_rate)
    # show_reconstruction(rc)
    return model, rc


if __name__ == '__main__':
    tg = TestGenerator(cut_length=30, stride=5, num_data_per_poly=400)
    train_set_for_encoder = tg.get_train_set_for_encoder()
    model, _ = create_encoder_decoder(train_set_for_encoder)
    N = int(np.ceil(len(tg.data_set)/2))
    train_set_for_discriminator = tg.get_train_set_for_discriminator(N=N, encoder=model.encode)
    train_discriminator(train_set_for_discriminator, batch_size=32, num_epochs=20)
    # find_match(model.encode)
    #test_encoder()
    # test_test_gen2()







