import random
import math

import numpy as np
# import torch
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

from stochasticsolver import OpenES, CMAES, PEPG, HillClimber, RND, AFPO, GeneticAlgorithm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)


def random_solution(config):
    return np.random.random(config.n_params)


def create_solver(config):
    name = config.solver
    n_params = config.n_params
    if name == "es":
        return OpenES(n_params, popsize=40, rank_fitness=False, forget_best=False)
    elif name == "ga":
        return GeneticAlgorithm(num_params=config.n_params, seed=config.s, pop_size=10,
                                init_range=(0, 150000))  # SimpleGA(n_params, init_range=(0, 500000))
    elif name == "afpo":
        return AFPO(num_params=config.n_params, seed=config.s, pop_size=100, init_range=(0, 1000000))
    elif name == "cmaes":
        pop_size = 4 + math.floor(3 * math.log(n_params))
        return CMAES(n_params, sigma_init=1, popsize=pop_size + (config.np - pop_size % config.np))
    elif name == "pepg":
        return PEPG(n_params, forget_best=False)
    elif name == "hill":
        return HillClimber(n_params)
    elif name == "rnd":
        return RND(n_params)
    raise ValueError("Invalid solver name: {}".format(name))


def create_pattern():
    target = np.load("targets/orig.npy")
    new_target = np.zeros_like(target)
    radii = []
    for i in range(0, 100, 50):
        radii.append(i)
    center = np.array([new_target.shape[0] / 2, new_target.shape[1] / 2])
    for i, radius in enumerate(radii):
        for x in range(target.shape[0]):
            for y in range(target.shape[1]):
                if np.linalg.norm(np.array([x, y]) - center) > radius:
                    continue
                elif x == 100 and y == 100:
                    continue
                new_target[x, y] = target.max() if i % 2 == 0 else target.min()
    plt.imshow(new_target)
    plt.savefig("test.png")
    # np.save("targets/one.npy", new_target)
