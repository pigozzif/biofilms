import abc
import math
import os
import random

import numpy as np
import cv2
import networkx as nx
from scipy.integrate import solve_ivp

from bacteria.bacterium import SignallingBacterium, ClockBacterium


class Lattice(abc.ABC):
    cell_height = 1.0
    cell_width = 1.0
    magnify = 10.0

    def __init__(self, w, h, dt, max_t, is_init_cell, *args):
        self.w = w
        self.h = h
        self.t = 0.0
        self.dt = dt
        self.max_t = max_t
        self.is_init_cell = is_init_cell
        self._lattice = self.init_lattice(args)

    def get_lattice(self):
        return self._lattice

    @abc.abstractmethod
    def init_lattice(self, *args):
        pass

    @abc.abstractmethod
    def set_params(self, params):
        pass

    def get_center(self):
        return self.w // 2, self.h // 2

    def get_neighborhood(self, cell):
        return [self._lattice.nodes[neigh] for neigh in self._lattice.neighbors(cell["i"])]

    def should_step(self, dt):
        return self.t <= self.max_t * dt

    @abc.abstractmethod
    def solve(self, dt):
        pass

    def _fill_canvas(self):
        return np.full(
            shape=(int(max(self._lattice.nodes(data=True), key=lambda x: x[1]["cy"])[1]["cy"] * self.magnify),
                   int(max(self._lattice.nodes(data=True), key=lambda x: x[1]["cx"])[1]["cx"] * self.magnify),
                   3),
            fill_value=255, dtype=np.uint8)

    @abc.abstractmethod
    def render(self, video_name):
        pass

    @abc.abstractmethod
    def get_fitness(self):
        pass

    @classmethod
    def create_lattice(cls, name, **kwargs):
        if name == "signalling":
            return SignallingLattice(kwargs["w"], kwargs["h"], kwargs["dt"], kwargs["max_t"], kwargs["phi_c"])
        elif name == "clock":
            return ClockLattice(kwargs["w"], kwargs["h"], kwargs["dt"], kwargs["max_t"], kwargs["task"])
        raise ValueError("Invalid lattice name: {}".format(name))


class SignallingLattice(Lattice):

    def __init__(self, w, h, dt, max_t, phi_c):
        super().__init__(w, h, dt, max_t, lambda x, y: x % 2 == 0, phi_c)
        self._connect_triangular_lattice()
        self._init_conditions = [1.0 if d["col"] == 0 else 0.0 for _, d in self._lattice.nodes(data=True)
                                 if d["cell"] is not None]
        self.sol = None

    def init_lattice(self, phi_c):
        lattice = nx.Graph()
        i = 0
        for row in range(self.h):
            for col in range(self.w):
                cell = SignallingBacterium(idx=i, u_init=1.0 if col == 0 else 0.0, phi_c=phi_c)
                if self.is_init_cell(row, col):
                    lattice.add_node(i, i=i, cell=cell, row=row, col=col * 2,
                                     cx=col * self.cell_width + self.cell_width / 2.0,
                                     cy=math.ceil(row / 2) * self.cell_height + self.cell_height / 2.0)
                else:
                    lattice.add_node(i, i=i, cell=cell, row=row, col=1 + col * 2,
                                     cx=col * self.cell_width + self.cell_width,
                                     cy=math.ceil(row / 2) * self.cell_height)
                i += 1
        return lattice

    def _connect_triangular_lattice(self):
        for node, d in self._lattice.nodes(data=True):
            row = d["row"]
            neigh = node + 2 * self.w
            if self._lattice.has_node(neigh) and self._lattice.nodes[neigh]["row"] > d["row"]:
                self._lattice.add_edge(node, neigh)
            neigh = node - 2 * self.w
            if self._lattice.has_node(neigh) and self._lattice.nodes[neigh]["row"] < d["row"]:
                self._lattice.add_edge(node, neigh)
            neigh = node + self.w if row % 2 == 0 else node + self.w + 1
            if self._lattice.has_node(neigh) and self._lattice.nodes[neigh]["row"] > d["row"] \
                    and self._lattice.nodes[neigh]["col"] > d["col"]:
                self._lattice.add_edge(node, neigh)
            neigh = node + self.w - 1 if row % 2 == 0 else node + self.w
            if self._lattice.has_node(neigh) and self._lattice.nodes[neigh]["row"] > d["row"] \
                    and self._lattice.nodes[neigh]["col"] < d["col"]:
                self._lattice.add_edge(node, neigh)
            neigh = node - self.w - 1 if row % 2 == 0 else node - self.w
            if self._lattice.has_node(neigh) and self._lattice.nodes[neigh]["row"] < d["row"] \
                    and self._lattice.nodes[neigh]["col"] < d["col"]:
                self._lattice.add_edge(node, neigh)
            neigh = node - self.w if row % 2 == 0 else node - self.w + 1
            if self._lattice.has_node(neigh) and self._lattice.nodes[neigh]["row"] < d["row"] \
                    and self._lattice.nodes[neigh]["col"] > d["col"]:
                self._lattice.add_edge(node, neigh)
        return self._lattice

    def set_params(self, params):
        return

    def _get_coupling(self, i, j):
        return 0.5 if j == i + 2 * self.w or j == i - 2 * self.w else 0.25

    def propagate(self, t, y, dt):
        dy = []
        for node, d in self._lattice.nodes(data=True):
            if d["cell"] is not None:
                dy.append(d["cell"].FitzHughNagumo_percolate(t=t, y=y, lattice=self, dt=dt))
        return dy

    def solve(self, dt):
        self.sol = solve_ivp(fun=self.propagate,
                             t_span=[0.0, self.max_t * self.dt],
                             t_eval=[i * self.dt for i in range(self.max_t)],
                             y0=self._init_conditions,
                             args=[dt])
        if self.sol.y.shape[1] != self.max_t:
            raise RuntimeError("Integration failed: {}".format(self.sol.y.shape))

    def _draw_cell(self, image, d, c):
        cv2.ellipse(image, (int(d["cx"] * self.magnify), int(d["cy"] * self.magnify)),
                    axes=(int(self.cell_width * self.magnify - self.magnify),
                          int(self.cell_height * self.magnify - self.magnify / 2)),
                    angle=0.0, startAngle=0.0, endAngle=360.0, color=c, thickness=-1)

    def render(self, video_name):
        renderer = None
        for i in range(self.max_t):
            image = self._fill_canvas()
            for _, d in self._lattice.nodes(data=True):
                if d["cell"] is None:
                    continue
                self._draw_cell(image=image, d=d, c=self.sol.y[d["i"], i])
            if renderer is None:
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                renderer = cv2.VideoWriter(video_name, fourcc, 20, (image.shape[1], image.shape[0]))
            renderer.write(image)

    def get_fitness(self):
        raise NotImplementedError


class ClockLattice(Lattice):
    D = 0.5
    init_conditions = [0.6 * 1000.0, 0.7, 0.1, 2.0, 10.0, 90.0 * 1000.0, 1.0 * 1000.0, 10.0 * 1000.0, 0.1]

    def __init__(self, w, h, dt, max_t, task):
        super().__init__(w, h, dt, max_t, lambda x, y: (x == h // 2 and y == w // 2))
        self._connect_square_lattice()
        self.dt = dt
        self.task = task
        self.sols = []
        self.frontier = [d for _, d in self._lattice.nodes(data=True)
                         if d["cell"] is not None and d["cell"].is_frontier]
        self.seed = [f for f in self.frontier]  # TODO: BE CAREFUL WITH MORE COMPLEX SEEDS
        self._update_distances()

    def init_lattice(self, *args):
        lattice = nx.Graph()
        i = 0
        for row in range(self.h):
            for col in range(self.w):
                if self.is_init_cell(row, col):
                    cell = ClockBacterium(idx=i)
                else:
                    cell = None
                lattice.add_node(i,
                                 i=i,
                                 cell=cell,
                                 row=row,
                                 col=col,
                                 cx=col * self.cell_width + self.cell_width / 2.0,
                                 cy=row * self.cell_height + self.cell_height / 2.0,
                                 history=[],
                                 distance=float("inf"))
                i += 1
        return lattice

    def _connect_square_lattice(self):
        for node, d in self._lattice.nodes(data=True):
            for offset in [self.w, -self.w, 1, -1, self.w + 1, self.w - 1, -self.w - 1, -self.w + 1]:
                neigh = node + offset
                if self._lattice.has_node(neigh) and abs(self._lattice.nodes[neigh]["row"] - d["row"]) <= 1.0 \
                        and abs(self._lattice.nodes[neigh]["col"] - d["col"]) <= 1.0:
                    self._lattice.add_edge(node, neigh)

    def _update_distances(self):
        for _, d in self._lattice.nodes(data=True):
            nx.set_node_attributes(self._lattice, values={
                d["i"]: min([math.sqrt((d["cx"] - s["cx"]) ** 2 + (d["cy"] - s["cy"]) ** 2) for s in self.seed])},
                                   name="distance")

    def diffuse(self, i, cell, idx):  # TODO: IS DIFFUSION AMONG BACTERIA ONLY?
        return - self.D * sum([cell["cell"].y[i - 1, idx] - n["cell"].y[i - 1, idx]
                               for n in self.get_neighborhood(cell=cell) if n["cell"] is not None])

    def _select_parent(self, cell):
        return random.choice([d for d in self.get_neighborhood(cell=cell) if d["cell"] is not None])

    def _metabolize(self, i):
        for _, d in self._lattice.nodes(data=True):
            if d["cell"] is not None:
                d["cell"].propagate(lattice=self, t=i, dt=self.dt, d=d)

    def _grow(self, i):
        n_f = len(self.frontier)
        for node, d in self._lattice.nodes(data=True):
            if d["cell"] is None and (i - 1) * self.cell_width <= d["distance"] <= i * self.cell_width:
                nx.set_node_attributes(self._lattice,
                                       values={node: ClockBacterium(idx=node)},
                                       name="cell")
                self.frontier.append(d)
        for j in range(n_f):
            f = self.frontier.pop(0)
            f["cell"].is_frontier = False

    def _update_ages(self):
        for _, d in self._lattice.nodes(data=True):
            if d["cell"] is not None:
                d["cell"].age += 1

    def _correct_age(self):
        for _, d in self._lattice.nodes(data=True):
            if d["cell"] is None:
                continue
            elif d["cell"].age == 1:
                d["cell"].age = self.max_t

    def set_params(self, params):
        ClockBacterium.alpha_e = params[0]
        ClockBacterium.alpha_o = params[1]

    def _sols2cells(self):
        for _, d in self._lattice.nodes(data=True):
            if d["cell"] is None:
                continue
            d["cell"].y = self.sols[len(self.sols) - self.max_t + d["cell"].age - 1].y

    def solve(self, dt):
        for i in range(self.max_t):
            self._grow(i=i)
            self._update_ages()
        # self._correct_age()
        sol_frontier = solve_ivp(fun=ClockBacterium.NasA_oscIII_D,
                                 t_span=[0.0, self.max_t * self.dt],
                                 t_eval=[i * self.dt for i in range(self.max_t)],
                                 y0=self.init_conditions)
        self.sols.append(sol_frontier)
        for i in range(self.max_t - 1):
            sol = solve_ivp(fun=ClockBacterium.NasA_oscIII_eta,
                            t_span=[0.0, (self.max_t - i - 1) * self.dt],
                            t_eval=[j * self.dt for j in range(self.max_t - i - 1)],
                            y0=np.concatenate((sol_frontier.y[:, i + 1], np.zeros(1))))
            self.sols.append(sol)
            if sol.y.shape[1] != self.max_t - i - 1:
                raise RuntimeError("Integration failed at step {0}: {1}".format(i, sol.y.shape))
        # self._sols2cells()

    def _draw_cell(self, val, image, d, min_val, max_val):
        cv2.rectangle(image,
                      (int((d["cx"] - self.cell_width / 2) * self.magnify),
                       int((d["cy"] - self.cell_height / 2) * self.magnify)),
                      (int((d["cx"] + self.cell_width / 2) * self.magnify),
                       int((d["cy"] + self.cell_height / 2) * self.magnify)),
                      color=d["cell"].draw(val=val, min_val=min_val, max_val=max_val),
                      thickness=-1)

    def render(self, video_name):
        image = self._fill_canvas()
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        renderer = cv2.VideoWriter(video_name, fourcc, 20, (image.shape[1], image.shape[0]))
        min_val = min([np.min(sol.y[8, :]) for sol in self.sols])
        max_val = max([np.max(sol.y[8, :]) for sol in self.sols])
        for i in range(self.max_t):
            for _, d in self._lattice.nodes(data=True):
                if d["cell"] is None or d["cell"].age < self.max_t - i:
                    continue
                diff = self.max_t - d["cell"].age
                if diff == i:
                    val = self.sols[0].y[8, i]
                else:
                    sol = self.sols[diff]
                    val = sol.y[8, i - (self.max_t - sol.y.shape[1])]
                self._draw_cell(val=val, image=image, d=d, min_val=min_val, max_val=max_val)
            cv2.putText(image,
                        text="Min response: {}".format(round(min_val, 3)),
                        org=(int((self.w - 50) * self.magnify), int(10 * self.magnify)),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1,
                        color=(0, 0, 0),
                        thickness=2)
            cv2.putText(image,
                        text="Max response: {}".format(round(max_val, 3)),
                        org=(int((self.w - 50) * self.magnify), int(15 * self.magnify)),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1,
                        color=(0, 0, 0),
                        thickness=2)
            renderer.write(image)

    def get_fitness(self):
        target = np.load(os.path.join("targets", self.task + ".npy"))
        prediction = np.zeros_like(target)
        for node, d in self._lattice.nodes(data=True):
            if d["cell"] is None or self.max_t - 1 not in d["vars"]:
                continue
            prediction[d["row"], d["col"]] = d["vars"][self.max_t - 1][8]
        # np.save("targets/one.npy", prediction)
        return np.sqrt(np.concatenate(np.square(prediction - target)).sum())
