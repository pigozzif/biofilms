import argparse
import os
from multiprocessing import Pool
import random
import time
import logging

import numpy as np
import gym

from bacteria.bacterium import SignalingBacterium
from bacteria.lattice import Lattice
from evo.evolution.algorithms import StochasticSolver
from evo.evolution.objectives import ObjectiveDict
from evo.listeners.listener import FileListener


def parse_args():
    parser = argparse.ArgumentParser(prog="BiofilmSimulation", description="Simulate a B. subtilis biofilm")
    parser.add_argument("--s", type=int, default=0, help="seed")
    parser.add_argument("--w", type=int, default=3, help="width in cells of the biofilm")
    parser.add_argument("--h", type=int, default=6, help="height in cells of the biofilm")
    parser.add_argument("--dt", type=float, default=0.2, help="integration step")
    parser.add_argument("--np", type=int, default=1, help="parallel optimization processes")
    parser.add_argument("--solver", type=str, default="afpo", help="solver")
    parser.add_argument("--task", type=str, default="pend", help="solver")
    parser.add_argument("--n_params", type=int, default=2, help="solution size")
    parser.add_argument("--evals", type=int, default=5000, help="fitness evaluations")
    parser.add_argument("--mode", type=str, default="random", help="work modality")
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
                           "best.solution": result[0]})
        if result[1] >= best_fitness or best_result is None:
            best_result = result[0]
            best_fitness = result[1]
        evaluated += len(solutions)
        j += 1
    return best_result, best_fitness


def parallel_wrapper(arg):
    c, solution, i, video_name = arg
    fitness = simulation(config=c, solution=solution)
    print(i)
    return i, -fitness


def set_params(params):
    SignalingBacterium.u_0 = params[0]
    SignalingBacterium.tau = params[1]


def simulation(config, solution, render=False, video_name=None):
    env = gym.make("Pendulum-v0", g=9.81)
    if render:
        env = gym.wrappers.Monitor(env, "videos", force=True)
    env.seed(config.s)
    obs = env.reset()
    set_params(params=solution)
    world = Lattice.create_lattice(name="signaling",
                                   w=config.w,
                                   h=config.h,
                                   dt=config.dt,
                                   max_t=env.spec.max_episode_steps,
                                   obs=obs)
    world.solve(env=env, render=render)
    env.close()
    if video_name is not None:
        world.render(video_name=video_name)
    return world.fitness


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.s)
    file_name = os.path.join("output", ".".join([args.task, args.solver, str(args.s), "txt"]))
    if args.mode == "random":
        print(simulation(config=args, solution=[0.0, 300], render=True, video_name="random.mp4"))
    elif args.mode == "opt":
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
                                                sigma=0.2,
                                                n=args.n_params,
                                                range=(-1, 1))
        best = parallel_solve(solver=solver, config=args, listener=listener)
        logging.warning("fitness score at this local optimum: {}".format(best[1]))
        print(simulation(config=args, solution=best[0], render=True))
    else:
        best = [float(x) for x in open(file_name, "r").readlines()[-1].split(";")[-1].strip().split("/")]
        print(simulation(config=args, solution=best, render=True))
