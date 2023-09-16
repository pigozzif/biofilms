import argparse
import os
import pickle
from multiprocessing import Pool
import random
import logging

from bacteria.lattice import Lattice

import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser(prog="BiofilmSimulation", description="Simulate a B. subtilis biofilm")
    parser.add_argument("--s", type=int, default=0, help="seed")
    parser.add_argument("--w", type=int, default=201, help="width in cells of the biofilm")
    parser.add_argument("--h", type=int, default=201, help="height in cells of the biofilm")
    parser.add_argument("--dt", type=float, default=0.3, help="integration step")
    parser.add_argument("--t", type=int, default=100, help="max simulation steps")
    parser.add_argument("--p", type=str, default="clock", help="problem")
    parser.add_argument("--np", type=int, default=1, help="parallel optimization processes")
    parser.add_argument("--solver", type=str, default="afpo", help="solver")
    parser.add_argument("--n_params", type=int, default=2, help="solution size")
    parser.add_argument("--evals", type=int, default=2500, help="fitness evaluations")
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
    c, solution, i = arg
    fitness = simulation(config=c, solution=solution, video_name=None)
    print(i)
    return i, -fitness


def simulation(config, solution, video_name):
    world = Lattice.create_lattice(name=config.p,
                                   w=config.w,
                                   h=config.h,
                                   dt=config.dt,
                                   max_t=config.t,
                                   video_name=video_name)
    world.set_params(params=solution)
    world.solve()
    # fitness = world.get_fitness()
    return 0.0  # fitness


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


def make_arras(directory, num_x=10, num_y=17):
    size_x, size_y = 2004, 2004
    step_x, step_y = 10000 * 2, 15000 * 2
    image = np.zeros((size_x * num_x, size_y * num_y, 3))
    for n, file in enumerate(os.listdir(directory)):
        if not file.endswith("mp4"):
            continue
        name = os.path.join(directory, file)
        coords = file.split(".")[1].strip("[]").split(",")
        raw_i = int(coords[0])
        i = int((raw_i / step_x) * size_x)
        raw_j = int(coords[1].strip(" "))
        j = int((raw_j / step_y) * size_y)
        if raw_j % step_y != 0 or raw_i % step_x != 0:
            continue
        cap = cv2.VideoCapture(name)
        frame = np.zeros((size_x, size_y, 3))
        while cap.isOpened():
            ret, _ = cap.read()
            if not ret:
                break
            else:
                frame = _
        image[i: i + size_x, j: j + size_y] = frame
    min_val = np.min(image)
    max_val = np.max(image)
    print(min_val, max_val)
    cv2.imwrite("arras.png", image)


def create_pattern():
    import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.s)
    # sample_fitness_landscape(config=args, num_workers=args.np)
    # make_arras(".")
    # file_name = ".".join([args.solver, str(args.s), "txt"])
    # objectives_dict = ObjectiveDict()
    # objectives_dict.add_objective(name="fitness", maximize=False, best_value=0.0, worst_value=5.0)
    # listener = FileListener(file_name=file_name, header=["iteration", "elapsed.sec", "evaluations", "best.fitness",
    #                                                      "best.solution"])
    # solver = StochasticSolver.create_solver(name=args.solver,
    #                                         seed=args.s,
    #                                         num_params=args.n_params,
    #                                         pop_size=100,
    #                                         genotype_factory="uniform_float",
    #                                         objectives_dict=objectives_dict,
    #                                         offspring_size=100,
    #                                         remap=False,
    #                                         genetic_operators={"gaussian_mut": 1.0},
    #                                         genotype_filter=None,
    #                                         tournament_size=5,
    #                                         mu=0.0,
    #                                         sigma=5000,
    #                                         n=args.n_params,
    #                                         range=(0, 1000000),
    #                                         upper=1000000,
    #                                         lower=0)
    # best = parallel_solve(solver=solver, config=args, listener=listener)
    # logging.warning("fitness score at this local optimum: {}".format(best[1]))
    # best = [float(x) for x in open(FileListener.get_log_file_name(file_name), "r").readlines()[-1].split(";")[-1].strip().strip("[]").
    #         split(" ")[1:]]
    # 41.07383632659912 ± 0.3074971987242349
    # 29.66873347759247 ± 0.26987235562151174
    # 27.248690128326416 ± 0.2814969384013431 (probably lower)
    print(simulation(config=args, solution=[20000, 100000], video_name="random.mp4"))
    exit()
    import time
    times = []
    for i in range(10):
        set_seed(i)
        start = time.time()
        simulation(config=args, solution=[20000, 100000], video_name=None)
        times.append(time.time() - start)
    print(np.median(times), "±", np.std(times))
