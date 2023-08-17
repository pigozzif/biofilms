import abc
import math
import os
import random

import numpy as np
import cv2
import networkx as nx
from scipy.integrate import solve_ivp
from shapely.geometry import Point, Polygon

from bacteria.bacterium import SignallingBacterium, ClockBacterium


class Lattice(abc.ABC):
    cell_height = 1.0
    cell_width = 1.0
    magnify = 100.0

    def __init__(self, w, h, dt, max_t, is_init_cell, *args):
        self.w = w
        self.h = h
        self.t = 0.0
        self.dt = dt
        self.max_t = max_t
        self.is_init_cell = is_init_cell
        self.input_cells = []
        self.output_cells = []
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
    init_conditions = [0.6 * 1000.0, 0.7, 0.1, 2.0, 10.0, 90.0 * 1000.0, 1.0 * 1000.0, 10.0 * 1000.0, 0.1,
                       0.0]

    def __init__(self, w, h, dt, max_t, task):
        super().__init__(w, h, dt, max_t, lambda x, y: 0 < y < w - 1 and 0 < x < h - 1)
        self._connect_square_lattice()
        self.dt = dt
        self.task = task
        self.frontier = []
        for _, d in self._lattice.nodes(data=True):
            if d["cell"] is None:
                continue
            elif any([n["cell"] is None for n in self.get_neighborhood(d)]):
                self.frontier.append(d)
            else:
                d["cell"].is_frontier = False
        self._update_distances()

    def init_lattice(self, *args):
        lattice = nx.Graph()
        i = 0
        for row in range(self.h):
            for col in range(self.w):
                if self.is_init_cell(row, col):
                    cell = ClockBacterium(idx=i,
                                          t=0,
                                          y=self.init_conditions,
                                          max_t=self.max_t)
                else:
                    cell = None
                lattice.add_node(i,
                                 i=i,
                                 cell=cell,
                                 row=row,
                                 col=col,
                                 cx=col * self.cell_width + self.cell_width / 2.0,
                                 cy=row * self.cell_height + self.cell_height / 2.0)
                if row == 1 and col == self.w // 2:
                    self.output_cells.append(cell)
                if row == self.h - 2 and (col == self.h * 0.3 or col == self.h * 0.6):
                    self.input_cells.append(cell)
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
            if d["cell"] is not None:
                d["cell"].distance = round(min([abs(d["cx"] - s["cx"]) + abs(d["cy"] - s["cy"])
                                                for s in self.frontier]))

    def diffuse(self, i, cell, idx):  # IS DIFFUSION AMONG BACTERIA ONLY?
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
                parent = self._select_parent(cell=d)
                nx.set_node_attributes(self._lattice,
                                       values={node: ClockBacterium(idx=node,
                                                                    t=i,
                                                                    y=parent["cell"].y[i],
                                                                    max_t=self.max_t)},
                                       name="cell")
                self.frontier.append(d)
        for j in range(n_f):
            f = self.frontier.pop(0)
            f["cell"].is_frontier = False

    def _update_ages(self):
        for _, d in self._lattice.nodes(data=True):
            if d["cell"] is not None:
                d["cell"].age += 1

    def set_params(self, params):
        ClockBacterium.alpha_e = params[0]
        ClockBacterium.alpha_o = params[1]

    def solve(self, inputs):
        for _, d in self._lattice.nodes(data=True):
            if d["cell"] is not None:
                d["cell"].init_y(max_t=self.max_t, t=0, y_0=self.init_conditions)
        for cell, inp in zip(self.input_cells, inputs):
            cell.y[0, 5] = inp * 10000
        for i in range(1, self.max_t):
            self._metabolize(i=i)
            for cell, inp in zip(self.input_cells, inputs):
                cell.y[i, 5] = inp * 10000
        return [cell.y[self.max_t - 1, 8] for cell in self.output_cells]

    def _draw_cell(self, i, image, d, min_val, max_val):
        cv2.rectangle(image,
                      (int((d["cx"] - self.cell_width / 2) * self.magnify),
                       int((d["cy"] - self.cell_height / 2) * self.magnify)),
                      (int((d["cx"] + self.cell_width / 2) * self.magnify),
                       int((d["cy"] + self.cell_height / 2) * self.magnify)),
                      color=d["cell"].draw(t=i, min_val=min_val, max_val=max_val),
                      thickness=-1)

    def render(self, video_name):
        image = self._fill_canvas()
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        renderer = cv2.VideoWriter(video_name, fourcc, 20, (image.shape[1], image.shape[0]))
        min_val = min([np.min(d["cell"].y[:, 8]) for _, d in self._lattice.nodes(data=True) if d["cell"] is not None])
        max_val = max([np.max(d["cell"].y[:, 8]) for _, d in self._lattice.nodes(data=True) if d["cell"] is not None])
        for i in range(self.max_t):
            for _, d in self._lattice.nodes(data=True):
                if d["cell"] is None:
                    continue
                self._draw_cell(i=i, image=image, d=d, min_val=min_val, max_val=max_val)
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
