import abc
import os
import random

import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.signal import convolve2d

from bacteria.bacterium import ClockBacterium


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
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.renderer = cv2.VideoWriter(video_name, fourcc, 20, (round(self.h * self.magnify),
                                                                     round(self.w * self.magnify)))
        else:
            self.renderer = None

    @abc.abstractmethod
    def set_params(self, params):
        pass

    def get_center(self):
        return self.w // 2, self.h // 2

    def should_step(self, dt):
        return self.t <= self.max_t * dt

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
    D = 0.0075
    init_conditions = [0.6 * 1000.0, 0.7, 0.1, 2.0, 10.0, 90.0 * 1000.0, 1.0 * 1000.0, 10.0 * 1000.0, 0.1]
    moore_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    def __init__(self, w, h, dt, max_t, video_name):
        super().__init__(w, h, dt, max_t, video_name)
        seed = ClockBacterium(idx=0, cx=self.get_center()[0], cy=self.get_center()[1], init_y=self.init_conditions)
        self.cells.append(seed)
        self.frontier = [seed]
        self.pos = np.zeros((self.h, self.w))
        self.pos[round(seed.cx), round(seed.cy)] = 1
        self.diffusion = np.zeros((self.h, self.w, 2))
        self._idx = 0
        self._tree = None

    @property
    def idx(self):
        self._idx += 1
        return self._idx

    @idx.setter
    def idx(self, value):
        self._idx = value

    def _metabolize(self, t):
        self.diffusion.fill(0)
        distances, _ = self._tree.query([(cell.cx, cell.cy) for cell in self.cells], k=1, p=2)
        for cell, d in zip(self.cells, distances):
            cell.propagate(t=t, dt=self.dt, k=d)
            self.diffusion[round(cell.cx), round(cell.cy), :] += cell.y[5]
            self.diffusion[round(cell.cx), round(cell.cy), :] += cell.y[7]

    def _diffuse(self):
        gradient_1 = self.dt * convolve2d(self.diffusion[:, :, 0], np.array([[self.D / 2, self.D, self.D / 2],
                                                                             [self.D, 0, self.D],
                                                                             [self.D / 2, self.D, self.D / 2]]),
                                          mode="same")
        gradient_2 = self.dt * convolve2d(self.diffusion[:, :, 1], np.array([[self.D / 2, self.D, self.D / 2],
                                                                             [self.D, 0, self.D],
                                                                             [self.D / 2, self.D, self.D / 2]]),
                                          mode="same")
        for cell in self.cells:
            cell.y[5] += gradient_1[round(cell.cx), round(cell.cy)]
            cell.y[7] += gradient_2[round(cell.cx), round(cell.cy)]

    def _grow(self):
        for parent_cell in self.frontier:
            neighborhood = [(x, y) for x, y in self._get_neighborhood(round(parent_cell.cx), round(parent_cell.cy))
                            if self.pos[x, y] == 0]
            if neighborhood:
                self.cells.append(ClockBacterium(idx=self.idx,
                                                 cx=parent_cell.cx,
                                                 cy=parent_cell.cy,
                                                 init_y=parent_cell.y.copy()))
                cx, cy = random.choice(neighborhood)
                parent_cell.cx = cx
                parent_cell.cy = cy
                self.idx += 1
                self.pos[cx, cy] += 1

    def _get_neighborhood(self, row, col):
        return [(row + dy, col + dx) for dx, dy in self.moore_offsets if
                0 <= row + dy <= self.h - 1 and 0 <= col + dx <= self.w - 1]

    def _update_pos(self):
        self.pos.fill(0)
        for cell in self.cells:
            self.pos[round(cell.cx), round(cell.cy)] += 1

    def _update_frontier(self):
        self.frontier.clear()
        for cell in self.cells:
            if sum(1 for x, y in self._get_neighborhood(round(cell.cx), round(cell.cy)) if self.pos[x, y] == 0) > 1:
                self.frontier.append(cell)

    def _update_ages(self):
        for cell in self.cells:
            cell.age += 1

    def set_params(self, params):
        ClockBacterium.alpha_e = params[0]
        ClockBacterium.alpha_o = params[1]

    def solve(self):
        for t in range(self.max_t):
            # 1) metabolism
            self._tree = cKDTree([(cell.cx, cell.cy) for cell in self.frontier], leafsize=50)
            self._metabolize(t=t)
            # 1.b) diffuse
            self._diffuse()
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
        image = self._fill_canvas()
        for cell in self.cells:
            self._draw_cell(image=image, cell=cell, min_val=min_val, max_val=max_val)
        cv2.putText(image,
                    text="Min response: {}".format(round(min_val, 3)),
                    org=(round((self.w - 50) * self.magnify), round(10 * self.magnify)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1,
                    color=(0, 0, 0),
                    thickness=2)
        cv2.putText(image,
                    text="Max response: {}".format(round(max_val, 3)),
                    org=(round((self.w - 50) * self.magnify), round(15 * self.magnify)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1,
                    color=(0, 0, 0),
                    thickness=2)
        self.renderer.write(image)

    def get_fitness(self):
        target = np.load(os.path.join("targets", self.task + ".npy"))
        prediction = np.zeros_like(target)
        for cell in self.cells:
            prediction[round(cell.cx), round(cell.cy)] = cell.y[8]
        # np.save("targets/one.npy", prediction)
        return np.sqrt(np.concatenate(np.square(prediction - target)).sum())
