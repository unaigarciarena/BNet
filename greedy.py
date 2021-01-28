import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from evolution.generator import Generator
from evolution.discriminator import Discriminator
from evolution.gan_train import GanTrain
from metrics.generative_score import initialize_fid
from torch import manual_seed
import random
from copy import deepcopy
import matplotlib.pyplot as plt

r = robjects.r
r['source']("BayesianNeswors.r")

gen_bnet = robjects.r("""load.nets("generator")""")

disc_bnet = robjects.r("""load.nets("discriminator")""")

read = robjects.r("""compute.lengths""")

compute_probas = robjects.r("""manual.probas""")


def search(limit=50, rnd=False, seed=0):

    manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    train = GanTrain()
    initialize_fid(train.train_loader)

    g = Generator(output_size=train.input_shape)
    d = Discriminator(input_shape=[1]+list(train.input_shape))
    g.setup()
    d.setup()
    train.train_evaluate(g, d, limit=100)
    g.calc_fid()
    current_fitness = g.fitness()
    fids = [current_fitness]
    steps = 0
    action = "Start"
    while steps < limit:
        print(steps, rnd)
        if rnd:
            cand_g, cand_d = create_random_neighbor(g, d)
        else:
            cand_g, cand_d = create_informed_neighbor(g, d)
        ims = g.forward(g.generate_noise(9).cpu()).detach().numpy()[:, 0]
        ims = np.concatenate([x for x in ims])
        plt.imshow(ims)
        plt.savefig(action + str(steps) + ".png")
        train.train_evaluate(cand_g, cand_d)
        cand_g.calc_fid()
        cand_fitness = cand_g.fitness()
        fids += [cand_fitness]
        if cand_fitness < current_fitness:
            action = "Accepted"
            g = cand_g
            d = cand_d
            current_fitness = cand_fitness
        else:
            action = "Rejected"
        steps += 1
    np.save("FIDs" + "_" + ("Random" if rnd else "Informed") + str(seed) + ".npy", fids)


def create_random_neighbor(generator, discriminator):
    g = deepcopy(generator)
    d = deepcopy(discriminator)

    if np.random.rand() > 0.5:
        g.genome.mut_for_greedy()
    else:
        d.genome.mut_for_greedy()

    return g, d


def create_informed_neighbor(generator, discriminator):
    aux = deepcopy(generator.genome)
    if len(aux.genes) < aux.max_layers:
        aux.add_random_gene()
        g1 = Generator(output_size=generator.output_size, genome=aux, input_shape=generator.input_shape)
        g1.setup()
    else:
        g1 = generator
    if len(aux.genes) > 1:
        aux.remove_random_gene()
        g2 = Generator(output_size=generator.output_size, genome=aux, input_shape=generator.input_shape)
        g2.setup()
    else:
        g2 = generator

    codes = np.array([g1.genome.codify()[:-3], g2.genome.codify()[:-3]])

    np.save("Generators.npy", codes.astype("float"))

    aux = deepcopy(discriminator.genome)
    if len(aux.genes) < aux.max_layers:

        aux.add_random_gene()
        d1 = Discriminator(output_size=discriminator.output_size, genome=aux, input_shape=discriminator.input_shape)
        d1.setup()
    else:
        d1 = discriminator

    if len(aux.genes) > 1:
        aux.remove_random_gene()
        d2 = Discriminator(output_size=discriminator.output_size, genome=aux, input_shape=discriminator.input_shape)
        d2.setup()
    else:
        d2 = generator

    codes = np.array([d1.genome.codify()[:-3], d2.genome.codify()[:-3]])

    np.save("Discriminators.npy", codes.astype("float"))

    reduced_gens = read("Generators.npy", "generator")
    reduced_discs = read("Discriminators.npy", "discriminator")

    gen_probas = np.array(compute_probas(reduced_gens, gen_bnet))
    disc_probas = np.array(compute_probas(reduced_discs, disc_bnet))

    print(gen_probas, disc_probas)

    if np.random.rand()<0.1:
        chrit = np.max
        arg_c = np.argmax
    else:
        chrit = np.min
        arg_c = np.argmin
    
    arg_gen = arg_c(gen_probas)
    arg_disc = arg_c(disc_probas)
    arg_gan = arg_c([gen_probas[arg_gen], disc_probas[arg_disc]])
    if arg_gan == 0:
        if arg_gen == 0:
            return g1, discriminator
        else:
            return g2, discriminator
    elif arg_disc == 0:
        return generator, d1
    else:
        return generator, d2



if __name__ == "__main__":

    search()
