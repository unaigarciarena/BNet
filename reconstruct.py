from evolution.genome import Genome
from evolution.discriminator import Discriminator
from evolution.generator import Generator
from evolution.gan_train import GanTrain
from metrics.generative_score import initialize_fid
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

seeds = range(30)


def load_data(force=True):
    data = []
    path = "coegan.npy"
    if os.path.isfile(path) and not force:
        data = np.load(path)
    else:
        for seed in seeds:
            subpath = "ResCOEGAN/res_" + str(seed) + "_30_5_MNIST_10_gan_10.npy"
            if os.path.isfile(subpath):
                data += [np.load(subpath)]
            else:
                print(subpath)
        data = np.array(data)
        print(data[0, 0, :10, :-4])
        np.save(path, data)
        np.save("Generators.npy", data[0, 0, :, :-4])
        np.save("Discriminators.npy", data[0, 1, :, :-4])



def reconstruct():
    gans = np.load("coegan.npy")

    gengen = Genome()
    gengen.decodify(gans[0, 0, -2, :-1], "gen")
    discgen = Genome()
    discgen.decodify(gans[0, 1, -1, :-1], "disc")
    train = GanTrain()
    initialize_fid(train.train_loader)
    discriminator = Discriminator(genome=discgen, output_size=1, input_shape=[1]+list(train.input_shape))
    discriminator.setup()
    generator = Generator(genome=gengen, output_size=train.input_shape)
    generator.setup()

    train.train_evaluate(generator, discriminator)
    generator.calc_fid()
    print(generator.fitness(), discriminator.fitness())


def data_for_R(n=40):
    gans = np.load("coegan.npy")
    generators = []
    discriminators = []
    for i in range(gans.shape[0]):
        gens = gans[i, 0]
        discs = gans[i, 1]
        fid_threshold = np.sort(gens[:, -1])[n]
        generators += [gens[gens[:, -1] < fid_threshold, :-1]]
        acc_threshold = np.sort(discs[:, -1])[n]
        discriminators += [discs[discs[:, -1] < acc_threshold, :-1]]
    generators = np.concatenate(generators)
    #print(generators[:, 24:28])
    discriminators = np.concatenate(discriminators)

    # count_gen = []
    # count_disc = []
    # print(discriminators.shape, generators.shape)
    # for i in range(generators.shape[0]):
    #     count_gen += [np.sum(generators[i, :6] == -1)]
    #     count_disc += [np.sum(discriminators[i, :6] == -1)]
    # print(count_gen)
    # print(count_disc)
    # sns.histplot(count_disc, label="Discriminator Depth")
    # sns.histplot(count_gen, label="Generator Depth", color="orange")
    # plt.legend()
    # plt.show()

    for depth in [1, 2, 3, 4, 5, 6]:
        aux = generators[generators[:, depth] == -1]
        generators = generators[generators[:, depth] > -1]
        indices = np.concatenate([np.arange(0, depth), np.arange(12, 12 + depth), np.arange(18, 18 + depth)])
        if aux.shape[0] > 0:
            print("Gens")
            print(aux[:3])
            print(aux[:, indices])
            np.save("generator" + str(depth) + "ForR.npy", aux[:, indices])
        aux = discriminators[discriminators[:, depth] == -1]
        discriminators = discriminators[discriminators[:, depth] > -1]
        if aux.shape[0] > 0:
            print("Discs")
            print(aux[:3])
            print(aux[:, indices])
            np.save("discriminator" + str(depth) + "ForR.npy", aux[:, indices])


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('integers', metavar='int', type=int, choices=range(3000), nargs=1)
    #
    # args = parser.parse_args()
    # seed = args.integers[0]
    load_data()
    #data_for_R()