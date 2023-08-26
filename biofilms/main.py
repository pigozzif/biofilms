import argparse
import os
from multiprocessing import Pool
import random
import time
import logging

from bacteria.lattice import ClockLattice

import numpy as np
import gym
from scipy.integrate import solve_ivp

# import matplotlib.pyplot as plt
from bacteria.bacterium import ClockBacterium
from evo.evolution.algorithms import StochasticSolver
from evo.evolution.objectives import ObjectiveDict
from evo.listeners.listener import FileListener


def parse_args():
    parser = argparse.ArgumentParser(prog="BiofilmSimulation", description="Simulate a B. subtilis biofilm")
    parser.add_argument("--s", type=int, default=0, help="seed")
    parser.add_argument("--dt", type=float, default=0.3, help="integration step")
    parser.add_argument("--policy", type=str, default="sin", help="problem")
    parser.add_argument("--np", type=int, default=7, help="parallel optimization processes")
    parser.add_argument("--solver", type=str, default="afpo", help="solver")
    parser.add_argument("--n_params", type=int, default=4, help="solution size")
    parser.add_argument("--evals", type=int, default=30000, help="fitness evaluations")
    parser.add_argument("--mode", type=str, default="test", help="modality")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)


def parallel_solve(solver, config, listener):
    best_result = None
    best_fitness = float("-inf")
    start_time = time.time()
    evaluated = 0
    j = 0
    while evaluated < config.evals:
        solutions = solver.ask()
        with Pool(config.np) as pool:
            results = pool.map(parallel_wrapper, [(config, solutions[i], i) for i in range(solver.pop_size)])
        fitness_list = [value for _, value in sorted(results, key=lambda x: x[0])]
        solver.tell(fitness_list)
        result = solver.result()  # first element is the best solution, second element is the best fitness
        if (j + 1) % 10 == 0:
            logging.warning("fitness at iteration {}: {}".format(j + 1, result[1]))
        listener.listen(**{"iteration": j, "elapsed.sec": time.time() - start_time,
                           "evaluations": evaluated, "best.fitness": result[1],
                           "best.solution": "/".join([str(x) for x in result[0]])})
        if result[1] >= best_fitness or best_result is None:
            best_result = result[0]
            best_fitness = result[1]
        evaluated += len(solutions)
        j += 1
    return best_result, best_fitness


def parallel_wrapper(arg):
    c, solution, i = arg
    fitness = simulation(config=c, solution=solution)
    return i, fitness


def set_params(params):
    ClockBacterium.alpha_e = params[0]
    ClockBacterium.alpha_o = params[1]


def integrate_lattice(num_actions, num_params, solution, dt, max_t):
    policies = []
    for i in range(num_actions):
        set_params(params=solution[i * num_params: (i + 1) * num_params])
        policies.append(solve_ivp(fun=ClockBacterium.NasA_oscIII_D,
                                  t_span=[0.0, max_t * dt],
                                  t_eval=[i * dt for i in range(max_t)],
                                  y0=ClockLattice.init_conditions))
    return policies


def sinusoidal_wave(num_actions, num_params, solution, max_t):
    policies = []
    for i in range(num_actions):
        params = solution[i * num_params: (i + 1) * num_params]
        policies.append(params[0] * np.sin(2 * np.pi * params[1] * np.arange(max_t) + params[2]) + params[3])
    return policies


def simulation(config, solution, render=False):
    env = gym.make("BipedalWalker-v3")
    env.seed(config.s)
    _ = env.reset()
    fitness = 0.0
    if config.policy == "sin":
        policies = sinusoidal_wave(num_actions=env.action_space.shape[0],
                                   num_params=config.n_params // env.action_space.shape[0],
                                   solution=solution,
                                   max_t=env.spec.max_episode_steps)
    else:
        policies = integrate_lattice(num_actions=env.action_space.shape[0],
                                     num_params=config.n_params // env.action_space.shape[0],
                                     solution=solution,
                                     dt=config.dt,
                                     max_t=env.spec.max_episode_steps)
    for t in range(env.spec.max_episode_steps):
        action = [policy[t] for policy in policies]
        observation, reward, done, _ = env.step(action)
        fitness += reward
        if done:
            _ = env.reset()
            break
        if render:
            env.render()
    env.close()
    # if video_name is not None:
    #     policy.render(video_name=video_name)
    return fitness


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.s)
    file_name = os.path.join("output", ".".join([args.solver, str(args.s), "txt"]))
    args.n_params *= 4
    if args.mode == "opt":
        objectives_dict = ObjectiveDict()
        objectives_dict.add_objective(name="fitness", maximize=True, best_value=300.0, worst_value=-100)
        listener = FileListener(file_name=file_name, header=["iteration", "elapsed.sec", "evaluations", "best.fitness",
                                                             "best.solution"])
        solver = StochasticSolver.create_solver(name=args.solver,
                                                seed=args.s,
                                                num_params=args.n_params,
                                                pop_size=100,
                                                genotype_factory="uniform_float",
                                                objectives_dict=objectives_dict,
                                                offspring_size=100,
                                                remap=False,
                                                genetic_operators={"gaussian_mut": 1.0},
                                                genotype_filter=None,
                                                tournament_size=5,
                                                mu=0.0,
                                                sigma=0.1,
                                                n=args.n_params,
                                                range=(-1, 1))
        best = parallel_solve(solver=solver, config=args, listener=listener)
        logging.warning("fitness score at this local optimum: {}".format(best[1]))
    else:
        best = [float(x) for x in open(file_name, "r").readlines()[-1].split(";")[-1].strip().split("/")]
    # orig: [20000, 100000], uneven: [90000, 100000], one: [100000, 800000]
    print(simulation(config=args, solution=best[0] if isinstance(best, tuple) else best, render=True))
