import abc
import os
import random

import numpy as np
import cv2

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
    D = 0.5
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
        self.idx = 1  # TODO: property

    # def diffuse(self, i, cell, idx):  # TODO: IS DIFFUSION AMONG BACTERIA ONLY?
    #     return - self.D * sum([cell["cell"].y[i - 1, idx] - n["cell"].y[i - 1, idx]
    #                            for n in self.get_neighborhood(cell=cell) if n["cell"] is not None])

    def _metabolize(self, t):
        for cell in self.cells:
            cell.propagate(t=t, dt=self.dt)

    def _grow(self, t):
        for parent_cell in self.frontier:
            neighborhood = self._get_neighborhood(round(parent_cell.cx), round(parent_cell.cy))
            if neighborhood:
                self.cells.append(ClockBacterium(idx=self.idx,
                                                  cx=parent_cell.cx,
                                                  cy=parent_cell.cy,
                                                  init_y=parent_cell.y))
                cx, cy = random.choice(neighborhood)
                parent_cell.cx = cx
                parent_cell.cy = cy
                self.idx += 1
                self.pos[cx, cy] += 1

    def _get_neighborhood(self, row, col):
        return [(row + dy, col + dx) for dx, dy in self.moore_offsets if
                0 <= row + dy <= self.h - 1 and 0 <= col + dx <= self.w - 1 and self.pos[row + dy, col + dx] == 0]

    def _update_pos(self):
        self.pos.fill(0)
        for cell in self.cells:
            self.pos[round(cell.cx), round(cell.cy)] += 1

    def _update_ages(self):
        for cell in self.cells:
            cell.age += 1

    def set_params(self, params):
        ClockBacterium.alpha_e = params[0]
        ClockBacterium.alpha_o = params[1]

    def solve(self):
        for t in range(self.max_t):
            # 1) metabolism
            self._metabolize(t=t)
            # 2) grow
            self._grow(t=t)
            # 3) update positions
            self._update_pos()
            if self.renderer is not None:
                self.render()

    def _draw_cell(self, cell, image, min_val, max_val):
        cv2.rectangle(image,
                      (round((cell.cx - self.cell_width / 2) * self.magnify),
                       round((cell.cy - self.cell_height / 2) * self.magnify)),
                      (round((cell.cx + self.cell_width / 2) * self.magnify),
                       round((cell.cy + self.cell_height / 2) * self.magnify)),
                      color=0.0,  # cell.draw(min_val=min_val, max_val=max_val),
                      thickness=-1)

    def render(self):
        min_val = 0.0
        max_val = 0.0
        image = self._fill_canvas()
        # print(np.max(self.pos), np.min(self.pos))
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
