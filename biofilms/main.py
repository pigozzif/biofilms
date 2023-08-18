import argparse
import pickle
from multiprocessing import Pool
import random
import time
import logging

from bacteria.lattice import Lattice

import numpy as np

from evo.utils.utilities import parse_largest_component, fill_morphology
from evo.evolution.algorithms import StochasticSolver
from evo.listeners.listener import FileListener
from evo.evolution.objectives import ObjectiveDict

SIZE = 15


def parse_args():
    parser = argparse.ArgumentParser(prog="BiofilmSimulation", description="Simulate a B. subtilis biofilm")
    parser.add_argument("--s", type=int, default=0, help="seed")
    parser.add_argument("--w", type=int, default=SIZE + 2, help="width in cells of the biofilm")
    parser.add_argument("--h", type=int, default=SIZE + 2, help="height in cells of the biofilm")
    parser.add_argument("--dt", type=float, default=0.3, help="integration step")
    parser.add_argument("--t", type=int, default=100, help="max simulation steps")
    parser.add_argument("--p", type=str, default="clock", help="problem")
    parser.add_argument("--np", type=int, default=7, help="parallel optimization processes")
    parser.add_argument("--solver", type=str, default="neat", help="solver")
    parser.add_argument("--n_params", type=int, default=SIZE * SIZE, help="solution size")
    parser.add_argument("--evals", type=int, default=5000, help="fitness evaluations")
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
            results = pool.map(parallel_wrapper, [(solutions[i], i, config) for i in range(len(solutions))])
        fitness_list = [value for _, value in sorted(results, key=lambda x: x[0])]
        solver.tell(fitness_list)
        result = solver.result()  # first element is the best solution, second element is the best fitness
        if result[1] >= best_fitness or best_result is None:
            best_result = result[0]
            best_fitness = result[1]
        if (j + 1) % 10 == 0:
            logging.warning("fitness at iteration {}: {}".format(j + 1, best_fitness))
        listener.listen(**{"iteration": j, "elapsed.sec": time.time() - start_time,
                           "evaluations": evaluated, "best.fitness": best_fitness,
                           "best.solution": parse_largest_component(fill_morphology(best_result, solver.config, config)).astype(int)})
        evaluated += solver.get_num_evaluated() - evaluated
        # print(evaluated)
        j += 1
    return best_result, best_fitness


def parallel_wrapper(arg):
    solution, i, c = arg
    fitness = simulation(config=c, solution=solution, video_name=None)
    # print(i)
    return i, -fitness


def simulation(config, solution, video_name):
    parsed_solution = parse_largest_component(solution=solution.reshape(SIZE, SIZE))
    if parsed_solution is None:
        return float("inf")
    world = Lattice.create_lattice(name=config.p, w=config.w, h=config.h, dt=config.dt, max_t=config.t, phi_c=0.43,
                                   solution=parsed_solution, video_name=video_name)
    err = []
    # for inputs, truth in zip([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [1]]):
    for inputs, truth in zip([[0, 0], [1, 0]], [[1], [0]]):
        outputs = world.solve(inputs=inputs)
        err.append((truth[0] - outputs[0]) ** 2)
    if video_name is not None:
        world.render(video_name=video_name)
    return sum(err) / len(err)  # fitness


def compute_fitness_landscape(config, file_name, num_workers=8):
    solutions = []
    # for x in range(0, 100000, 2500):
    #    for y in range(0, 300000, 2500):
    for x in range(0, 40000, 1000):
        for y in range(0, 200000, 1000):
            solutions.append([x, y])
    with Pool(num_workers) as pool:
        results = pool.map(parallel_wrapper, [(config, solutions[i], i) for i in range(len(solutions))])
    with open(file_name, "wb") as file:
        pickle.dump(results, file)


def sample_fitness_landscape(config, num_workers):
    solutions = []
    for x in range(0, 200000, 10000 * 2):
        for y in range(0, 500000, 15000 * 2):
            solutions.append([x, y])
    with Pool(num_workers) as pool:
        results = pool.map(parallel_wrapper, [(config, solutions[i], i, ".".join([str(i), str(solutions[i]), "mp4"]))
                                              for i in range(len(solutions))])


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.s)
    # sample_fitness_landscape(config=args, num_workers=args.np)
    file_name = ".".join([args.solver, str(args.s), "txt"])
    objectives_dict = ObjectiveDict()
    objectives_dict.add_objective(name="fitness", maximize=False, best_value=0.0, worst_value=float("inf"))
    listener = FileListener(file_name=file_name, header=["iteration", "elapsed.sec", "evaluations", "best.fitness",
                                                         "best.solution"])
    solver = StochasticSolver.create_solver(name=args.solver,
                                            seed=args.s,
                                            num_params=args.n_params,
                                            pop_size=50,
                                            # genotype_factory="uniform_float",
                                            objectives_dict=objectives_dict,
                                            # offspring_size=100,
                                            # remap=False,
                                            # genetic_operators={"gaussian_mut": 1.0},
                                            # genotype_filter="connected",
                                            # tournament_size=5,
                                            # mu=0.0,
                                            # sigma=0.35,
                                            # n=args.n_params,
                                            # range=(-1, 1))
                                            num_inputs=2,
                                            num_outputs=1,
                                            np=args.np,
                                            fitness_func=parallel_wrapper,
                                            config=args)
    best = parallel_solve(solver=solver, config=args, listener=listener)
    logging.warning("fitness score at this local optimum: {}".format(best[1]))
    # best = np.array([float(x) for x in open(file_name, "r").readlines()[-1].split(";")[-1].strip().strip("[]").
    #                 split(" ")[1:]])
    # orig: [20000, 100000], uneven: [90000, 100000], one: [100000, 800000]
    print(simulation(config=args, solution=parse_largest_component(fill_morphology(best[0], solver.config, args)),
                     video_name="best.mp4"))  # ".".join([file_name, "video", "mp4"])))
