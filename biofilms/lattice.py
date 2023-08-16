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

    def __init__(self, w, h, dt, max_t, task):
        super().__init__(w, h, dt, max_t,
                         lambda x,
                                y: (x == h // 2 and y == w // 2))  # or (
        # x == h // 4 and y == w // 2))  # (x == h // 2 and w // 2 - 10 <= y <= w // 2 + 10) or (y == w // 2 and h // 2 - 10 <= x <= h // 2 + 10))
        self._connect_square_lattice()
        self._init_conditions = [0.6 * 1000.0, 0.7, 0.1, 2.0, 10.0, 90.0 * 1000.0, 1.0 * 1000.0, 10.0 * 1000.0, 0.1,
                                 0.0]
        self.dt = dt
        self.task = task
        # self.sols = []
        self.sol = np.zeros((self.max_t, self.w, self.h, len(self._init_conditions)))
        self.sol[0, self.h // 2, self.w // 2] = self._init_conditions
        self.frontier = [d for _, d in self._lattice.nodes(data=True)
                         if d["cell"] is not None and d["cell"].is_frontier]
        self.seed = [f for f in self.frontier]  # BE CAREFUL WITH MORE COMPLEX SEEDS
        self.fronts = {}
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
                                 # vars={},
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

    def _order_vertices_clockwise(self, vertices):
        centroid_x = sum(x for x, y in vertices) / len(vertices)
        centroid_y = sum(y for x, y in vertices) / len(vertices)
        sorted_vertices = sorted(vertices, key=lambda vertex: (
            -math.atan2(vertex[1] - centroid_y, vertex[0] - centroid_x),
            (vertex[0] - centroid_x) ** 2 + (vertex[1] - centroid_y) ** 2
        ))
        return sorted_vertices

    def _update_histories(self):
        if len(self.seed) > 1:
            center1 = [self.seed[0]["row"], self.seed[0]["col"]]
            center2 = [self.seed[1]["row"], self.seed[1]["col"]]
            radius1 = self.seed[0]["cell"].age + 1
            radius2 = self.seed[1]["cell"].age + 1
            distance = math.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2)
            if radius1 + radius2 >= distance > 0.0:
                angle1 = math.acos((radius1 ** 2 + distance ** 2 - radius2 ** 2) / (2 * radius1 * distance))
                x1 = center1[0] + radius1 * math.cos(
                    math.atan2(center2[1] - center1[1], center2[0] - center1[0]) + angle1)
                y1 = center1[1] + radius1 * math.sin(
                    math.atan2(center2[1] - center1[1], center2[0] - center1[0]) + angle1)
                x2 = center1[0] + radius1 * math.cos(
                    math.atan2(center2[1] - center1[1], center2[0] - center1[0]) - angle1)
                y2 = center1[1] + radius1 * math.sin(
                    math.atan2(center2[1] - center1[1], center2[0] - center1[0]) - angle1)
                cuspids = [[x1, y1], [x2, y2]]
                diamond = Polygon(self._order_vertices_clockwise([center1, center2, [x1, y1], [x2, y2]]))
            else:
                diamond = None
        else:
            diamond = None
        for _, d in self._lattice.nodes(data=True):
            if d["cell"] is not None:
                if diamond is not None and diamond.contains(Point(d["row"], d["col"])):
                    distances = [math.floor(math.sqrt((d["row"] - c[0]) ** 2 + (d["col"] - c[1]) ** 2))
                                 for c in cuspids]
                    d["history"].append(min(distances) // self.cell_width)
                else:
                    d["history"].append(d["cell"].age)

    def _diffuse(self, i, cell, idx):
        return - self.D * sum([self.sol[i - 1, cell["row"], cell["col"], idx] - self.sol[i - 1, neigh["row"], neigh["col"], idx]
                               for neigh in self.get_neighborhood(cell=cell)])

    def _select_parent(self, cell):
        return random.choice([d for d in self.get_neighborhood(cell=cell) if d["cell"] is not None])

    def _grow_frontier(self, i):
        # to_grow = [d for _, d in self._lattice.nodes(data=True) if d["cell"] is not None and d["cell"].is_frontier]
        # for d in to_grow:
        # for _, d in self._lattice.nodes(data=True):
        #    if d["cell"] is not None:
        #        d["cell"].is_frontier = False
        # for n in self.get_neighborhood(cell=d["cell"]):
        #    elif (i - 1) * self.cell_width <= np.linalg.norm(
        #                np.array([d["cx"], d["cy"]]) - np.array(self.get_center())) <= i * self.cell_width:
        #            nx.set_node_attributes(self._lattice, values={d["i"]: ClockBacterium(idx=d["i"])}, name="cell")
        # d["cell"].is_frontier = False
        # candidates = {}
        # for d in self.frontier:
        #    for n in self.get_neighborhood(cell=d["cell"]):
        #        if n["cell"] is None:
        #            nx.set_node_attributes(self._lattice, values={n["i"]: ClockBacterium(idx=n["i"])}, name="cell")
        #    d["cell"].is_frontier = False
        n_f = len(self.frontier)
        for _, d in self._lattice.nodes(data=True):
            if d["cell"] is not None:
                if d["cell"].is_frontier:
                    dy = ClockBacterium.NasA_oscIII_D(t=i, y=self.sol[i - 1, d["row"], d["col"]])
                else:
                    dy = ClockBacterium.NasA_oscIII_eta(t=i, y=self.sol[i - 1, d["row"], d["col"]])
                dy[5] += self._diffuse(i=i, cell=d, idx=5)
                dy[7] += self._diffuse(i=i, cell=d, idx=7)
                self.sol[i, d["row"], d["col"]] = self.sol[i - 1, d["row"], d["col"]] + self.dt * dy
        for node, d in self._lattice.nodes(data=True):
            if d["cell"] is None and (i - 1) * self.cell_width <= d["distance"] <= i * self.cell_width:
                nx.set_node_attributes(self._lattice, values={node: ClockBacterium(idx=node)}, name="cell")
                parent = self._select_parent(cell=d)
                self.sol[i, d["row"], d["col"]] = self.sol[i, parent["row"], parent["col"]]
                self.frontier.append(d)
        for j in range(n_f):
            f = self.frontier.pop(0)
            f["cell"].is_frontier = False
        # self._update_histories()

    def _update_ages(self):
        for _, d in self._lattice.nodes(data=True):
            if d["cell"] is not None:
                d["cell"].age += 1

    def set_params(self, params):
        ClockBacterium.alpha_e = params[0]
        ClockBacterium.alpha_o = params[1]

    def sol2cell(self, sol, f):
        for _, d in self._lattice.nodes(data=True):
            if d["history"] == list(f):
                nx.set_node_attributes(self._lattice, values={d["i"]: {i + (self.max_t - sol.y.shape[1]):
                                                                           sol.y[:, i] for i in range(sol.y.shape[1])}},
                                       name="vars")

    def solve(self, dt):
        for i in range(1, self.max_t):
            self._grow_frontier(i=i)
            # self._update_ages()
        return
        paths = list(set([tuple(d["history"]) for _, d in self._lattice.nodes(data=True) if d["cell"] is not None]))
        paths = sorted(paths, key=lambda x: len(x), reverse=True)
        sol_frontier = solve_ivp(fun=ClockBacterium.NasA_oscIII_D,
                                 t_span=[0.0, self.max_t * self.dt],
                                 t_eval=[i * self.dt for i in range(self.max_t)],
                                 y0=self._init_conditions)
        # self.sols.append(sol_frontier)
        # for i in range(self.max_t - 1):
        for t, f in enumerate(paths):
            # sol = solve_ivp(fun=ClockBacterium.NasA_oscIII_eta,
            #                t_span=[0.0, (self.max_t - t - 1) * self.dt],
            #                t_eval=[j * self.dt for j in range(self.max_t - t - 1)],
            #                y0=np.concatenate((sol_frontier.y[:, t + 1], np.zeros((1,)))),
            #                args=(f, self.dt))
            sol = solve_ivp(fun=ClockBacterium.NasA_oscIII_eta,
                            t_span=[0.0, (len(f)) * self.dt],
                            t_eval=[j * self.dt for j in range(len(f))],
                            y0=np.concatenate((sol_frontier.y[:, self.max_t - len(f)], np.zeros((1,)))),
                            args=(f, self.dt))
            self.sols.append(sol)
            self.sol2cell(sol=sol, f=f)
            if sol.y.shape[1] != len(f):
                raise RuntimeError("Integration failed at step {0}: {1}".format(t, sol.y.shape))

    def _draw_cell(self, image, d, min_val, max_val):
        cv2.rectangle(image,
                      (int((d["cx"] - self.cell_width / 2) * self.magnify),
                       int((d["cy"] - self.cell_height / 2) * self.magnify)),
                      (int((d["cx"] + self.cell_width / 2) * self.magnify),
                       int((d["cy"] + self.cell_height / 2) * self.magnify)),
                      color=d["cell"].draw(min_val=min_val, max_val=max_val),
                      thickness=-1)

    def render(self, video_name):
        image = self._fill_canvas()
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        renderer = cv2.VideoWriter(video_name, fourcc, 20, (image.shape[1], image.shape[0]))
        min_val = np.min(self.sol[:, :, :, 8])
        max_val = np.max(self.sol[:, :, :, 8])
        for i in range(self.max_t):
            for _, d in self._lattice.nodes(data=True):
                if d["cell"] is None:
                    continue
                d["cell"].set_vars(y=self.sol[i, d["row"], d["col"]])
                self._draw_cell(image=image, d=d, min_val=min_val, max_val=max_val)
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
