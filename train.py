import torch
from evolution.gan_train import GanTrain
from evolution.config import config
# import better_exceptions; better_exceptions.hook()
import logging
import numpy as np
import argparse
import random
from memory_profiler import memory_usage

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.propagate = False
if torch.cuda.is_available():
    logger.info("CUDA device detected!")
else:
    logger.info("CUDA device not detected!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('integers', metavar='int', type=int, choices=range(3000), nargs=1)

    args = parser.parse_args()
    seed = args.integers[0]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    mem, data = memory_usage(GanTrain().start, retval=True)
    print(mem, max(mem))
    path = "res_" + str(seed) + "_" + str(config.evolution.max_generations) + "_" + str(config.evolution.max_layers) + "_" + config.gan.dataset + "_" + ("-".join(config.gan.dataset_classes) if config.gan.dataset_classes else "10") + "_" + config.gan.type + "_" + str(config.gan.discriminator.population_size)
    np.save(path, data)
