import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to

import pyro
import pyro.contrib.gp as gp

import pandas as pd

assert pyro.__version__.startswith('1.8.4')
pyro.set_rng_seed(1)


def f(x):
    data = pd.read_csv("xenobits.csv", sep=",")
    data = data[data["n_cells"] == x]
    return data.sample(1)["n_spikes"]


def main():
    pass


if __name__ == "__main__":
    main()
