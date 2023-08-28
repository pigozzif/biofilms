import argparse
from multiprocessing import Pool
import random
import time
import logging

from bacteria.lattice import Lattice

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(prog="BiofilmSimulation", description="Simulate a B. subtilis biofilm")
    parser.add_argument("--s", type=int, default=0, help="seed")
    parser.add_argument("--w", type=int, default=30, help="width in cells of the biofilm")
    parser.add_argument("--h", type=int, default=50, help="height in cells of the biofilm")
    parser.add_argument("--dt", type=float, default=0.02, help="integration step")
    parser.add_argument("--t", type=int, default=1000, help="max simulation steps")
    parser.add_argument("--p", type=str, default="signaling", help="problem")
    parser.add_argument("--np", type=int, default=1, help="parallel optimization processes")
    parser.add_argument("--solver", type=str, default="afpo", help="solver")
    parser.add_argument("--n_params", type=int, default=2, help="solution size")
    parser.add_argument("--evals", type=int, default=2500, help="fitness evaluations")
    parser.add_argument("--task", type=str, default="one", help="target pattern name")
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
    fitness = simulation(config=c, solution=solution, video_name=video_name)
    print(i)
    return i, -fitness


def simulation(config, solution, video_name):
    world = Lattice.create_lattice(name=config.p, w=config.w, h=config.h, dt=config.dt, max_t=config.t, phi_c=0.43)
    world.set_params(params=solution)
    world.solve(dt=config.dt)
    if video_name is not None:
        world.render(video_name=video_name)
    # fitness = world.get_fitness()
    return 0.0  # fitness


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
    # orig: [20000, 100000], uneven: [90000, 100000], one: [100000, 800000]
    print(simulation(config=args, solution=[20000, 100000], video_name="signaling.mp4"))  # ".".join([file_name, "video", "mp4"])))
    exit()
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    def fitzhugh_nagumo(t, y, tau):
        v, w = y
        dvdt = (v - (v ** 3) / 3) * (v - 0.01) - w
        dwdt = v / tau
        return [dvdt, dwdt]
    tau = 5

    # Initial conditions
    v0 = 1.0
    w0 = 0.0
    y0 = [v0, w0]

    # Time span for simulation
    t_span = (0, 50)

    # Perform the simulation using solve_ivp
    t_eval = [i * args.dt for i in range(args.t)]
    solution = solve_ivp(
        fitzhugh_nagumo,
        t_span=[0.0, args.t * args.dt],
        t_eval=t_eval,
        y0=y0,
        args=[tau]
    )

    # Plot the results
    plt.plot(t_eval, solution.y[0], label='v')
    plt.plot(t_eval, solution.y[1], label='w')
    plt.xlabel('Time')
    plt.ylabel('Variables')
    plt.title('FitzHugh-Nagumo Model')
    plt.legend()
    plt.show()
