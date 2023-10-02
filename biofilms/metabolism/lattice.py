import abc
import os

import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.signal import convolve2d

from metabolism.bacterium import ClockBacterium, MatrixProducing


class Lattice(abc.ABC):
    cell_height = 1.0
    cell_width = 1.0
    magnify = 10.0

    def __init__(self, w, h, dt, max_t, video_name, *args):
        self.w = w
        self.h = h
        self.t = 0.0
        self.dt = dt
        self.max_t = max_t
        self.cells = []
        if video_name is not None:
            self.image = self._fill_canvas()
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.renderer = cv2.VideoWriter(video_name, fourcc, 20, (self.image.shape[1],
                                                                     self.image.shape[0]))
        else:
            self.renderer = None

    @abc.abstractmethod
    def set_params(self, params):
        pass

    def get_center(self):
        return self.w // 2, self.h // 2

    @abc.abstractmethod
    def solve(self):
        pass

    def _fill_canvas(self):
        return np.full(
            shape=(round(self.h * self.magnify),
                   round(self.w * self.magnify),
                   3),
            fill_value=255, dtype=np.uint8)

    @abc.abstractmethod
    def render(self):
        pass

    @abc.abstractmethod
    def get_fitness(self):
        pass

    @classmethod
    def create_lattice(cls, name, **kwargs):
        if name == "clock":
            return ClockLattice(kwargs["w"], kwargs["h"], kwargs["dt"], kwargs["max_t"], kwargs["video_name"])
        raise ValueError("Invalid lattice name: {}".format(name))


class ClockLattice(Lattice):
    D = 0.005
    friction = 1.0
    init_conditions = np.array([0.6 * 1000.0, 0.7, 0.1, 2.0, 10.0, 90.0 * 1000.0, 1.0 * 1000.0, 10.0 * 1000.0, 0.1])
    moore_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    def __init__(self, w, h, dt, max_t, video_name):
        super().__init__(w, h, dt, max_t, video_name)
        seed = ClockBacterium(cx=self.get_center()[0],
                              cy=self.get_center()[1],
                              vel=np.zeros(2),
                              y=self.init_conditions.copy(),
                              specialization=MatrixProducing())
        self.cells.append(seed)
        self.frontier = [seed]
        self.pos = np.zeros((self.h, self.w))
        self.pos[seed.row, seed.col] = 1
        self.diffusion = np.zeros((self.h, self.w, 2))
        self.diffusion_kernel = np.array([[self.D / 2, self.D, self.D / 2],
                                          [self.D, 0, self.D],
                                          [self.D / 2, self.D, self.D / 2]])
        self._tree = None

    def _metabolize(self, t):
        self.diffusion.fill(0)
        dists, _ = self._tree.query([[cell.cx, cell.cy] for cell in self.cells], k=1, p=2)
        for cell, d in zip(self.cells, dists):
            cell.metabolize(t=t, dt=self.dt, k=d)
            self.diffusion[cell.row, cell.col, 0] += cell.y[5]
            self.diffusion[cell.row, cell.col, 1] += cell.y[7]

    def _diffuse(self):
        gradient_1 = self.dt * convolve2d(self.diffusion[:, :, 0], self.diffusion_kernel, mode="same")
        gradient_2 = self.dt * convolve2d(self.diffusion[:, :, 1], self.diffusion_kernel, mode="same")
        for cell in self.cells:
            cell.y[5] += gradient_1[cell.row, cell.col]
            cell.y[7] += gradient_2[cell.row, cell.col]

    def _grow(self):
        for parent_cell in self.frontier:
            neighborhood = [(x, y) for x, y in self._get_neighborhood(parent_cell.row, parent_cell.col)
                            if self.pos[x, y] == 0]
            if neighborhood:
                child = parent_cell.divide(parent_cell=parent_cell, neighborhood=neighborhood)
                if child is not None:  # TODO: fix OOP
                    self.cells.append(child)
                    self.pos[parent_cell.row, parent_cell.col] += 1

    def _get_neighborhood(self, row, col):
        return [(row + dy, col + dx) for dx, dy in self.moore_offsets if
                0 <= row + dy <= self.h - 1 and 0 <= col + dx <= self.w - 1]

    def _update_pos(self):
        self.pos.fill(0)
        for cell in self.cells:
            cell.move(dt=self.dt, k=self.friction)
            self.pos[cell.row, cell.col] += 1

    def _update_frontier(self):
        self.frontier.clear()
        for cell in self.cells:
            if sum(1 for x, y in self._get_neighborhood(cell.row, cell.col) if self.pos[x, y] == 0) > 1:
                self.frontier.append(cell)

    def _update_ages(self):
        for cell in self.cells:
            cell.age += 1

    def set_params(self, params):
        ClockBacterium.alpha_e = params[0]
        ClockBacterium.alpha_o = params[1]

    def step(self, t):
        # print(t)
        # 1) metabolism
        self._tree = cKDTree([[cell.cx, cell.cy] for cell in self.frontier], leafsize=50)
        self._metabolize(t=t)
        # 1.b) diffuse
        # self._diffuse()
        # 2) grow
        self._grow()
        # 3) update positions
        self._update_pos()
        # 4) update frontier
        self._update_frontier()
        # self._update_ages()
        # 5) render
        if self.renderer is not None:
            self.render()

    def solve(self):
        for t in range(self.max_t):
            self.step(t=t)

    def _draw_cell(self, cell, image, min_val, max_val):
        cv2.rectangle(image,
                      (round((cell.cx - self.cell_width / 2) * self.magnify),
                       round((cell.cy - self.cell_height / 2) * self.magnify)),
                      (round((cell.cx + self.cell_width / 2) * self.magnify),
                       round((cell.cy + self.cell_height / 2) * self.magnify)),
                      color=cell.draw(min_val=min_val, max_val=max_val),
                      thickness=-1)

    def render(self):
        min_val = 0.0
        max_val = 1.5
        self.image.fill(255)
        for cell in self.cells:
            self._draw_cell(image=self.image, cell=cell, min_val=min_val, max_val=max_val)
        cv2.putText(self.image,
                    text="Min response: {}".format(round(min_val, 3)),
                    org=(round((self.w - 50) * self.magnify), round(10 * self.magnify)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1,
                    color=(0, 0, 0),
                    thickness=2)
        cv2.putText(self.image,
                    text="Max response: {}".format(round(max_val, 3)),
                    org=(round((self.w - 50) * self.magnify), round(15 * self.magnify)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1,
                    color=(0, 0, 0),
                    thickness=2)
        self.renderer.write(self.image)

    def get_fitness(self):
        target = np.load(os.path.join("targets", self.task + ".npy"))
        prediction = np.zeros_like(target)
        for cell in self.cells:
            prediction[round(cell.cx), round(cell.cy)] = cell.y[8]
        # np.save("targets/one.npy", prediction)
        return np.sqrt(np.concatenate(np.square(prediction - target)).sum())
