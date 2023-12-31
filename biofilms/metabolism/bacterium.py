import abc
import random

import numpy as np


class Specialization(abc.ABC):

    @abc.abstractmethod
    def move(self, cx, cy, vel, dt, k):
        pass

    @abc.abstractmethod
    def divide(self, parent, neighborhood):
        pass


class SurfactinProducing(Specialization):

    def move(self, cx, cy, vel, dt, k):
        cx += vel[0] * dt
        cy += vel[1] * dt
        vel -= k * vel * dt
        return cx, cy, vel

    def divide(self, parent, neighborhood):
        return


class MatrixProducing(Specialization):

    def move(self, cx, cy, vel, dt, k):
        return cx, cy, vel

    def divide(self, parent_cell, neighborhood):
        child_cell = ClockBacterium(cx=parent_cell.cx,
                                    cy=parent_cell.cy,
                                    vel=np.random.random(2),
                                    y=parent_cell.y.copy(),
                                    specialization=MatrixProducing())
        cx, cy = random.choice(neighborhood)
        parent_cell.cx = cx
        parent_cell.cy = cy
        return child_cell


class ClockBacterium(object):
    alpha_e = 2.0 * 1000 * 10.0  # 2.0
    gamma_e = 0.1
    alpha_n = 0.1
    alpha_q = 0.01 * 0.001  # 10.0
    beta_q = 0.001  # 1000.0
    alpha_o = 10.0 * 1000.0 * 10.0  # 10.0
    gamma_o = 10.0
    alpha_s = 10.0
    beta_s = 1.0
    alpha_f = 1.0
    gamma_q = 1.0
    beta_f = 1.0
    t_t = 1.0
    alpha_t = 1.0
    beta_t = 10.0
    alpha_a = 10.0
    beta_a = 1.0
    alpha_r = 10.0
    beta_r = 1.0
    gamma_r = 0.05
    kappa_f = 1000.0  # 1.0
    kappa_t = 0.1  # 1.0
    kappa_r = 0.3
    n1 = 5.0
    n2 = 5.0
    k = 0.1
    r_0 = 1.5
    eta = 2.0
    epsilon = 0.13

    def __init__(self, cx, cy, vel, y, specialization):
        self._cx = cx
        self._cy = cy
        self.row = round(self._cx)
        self.col = round(self._cy)
        self.vel = vel
        self.y = y
        self._specialization = specialization
        self.age = 0

    @property
    def cx(self):
        return self._cx

    @cx.setter
    def cx(self, value):
        self._cx = value
        self.row = round(self._cx)

    @property
    def cy(self):
        return self._cy

    @cy.setter
    def cy(self, value):
        self._cy = value
        self.col = round(self._cy)

    @property
    def specialization(self):
        return self._specialization

    @specialization.setter
    def specialization(self, new_specialization):
        self._specialization = new_specialization

    def move(self, dt, k):
        self._cx, self._cy, self.vel = self.specialization.move(self._cx, self._cy, self.vel, dt, k)

    def divide(self, parent_cell, neighborhood):
        return self.specialization.divide(parent_cell=parent_cell, neighborhood=neighborhood)

    @staticmethod
    def update_E(e, a, o, q, s, n):
        return ClockBacterium.alpha_e - ClockBacterium.alpha_n * e + ClockBacterium.beta_q * a * o * q - ClockBacterium.alpha_q * s * e * n - ClockBacterium.gamma_e * e

    @staticmethod
    def update_N(e, s, n):
        return ClockBacterium.alpha_n * e - ClockBacterium.alpha_q * s * e * n

    @staticmethod
    def update_O(e, a, o, q):
        return ClockBacterium.alpha_o + ClockBacterium.alpha_n * e - ClockBacterium.beta_q * a * o * q - ClockBacterium.gamma_o * o

    @staticmethod
    def update_S(s, q):
        return ClockBacterium.alpha_s - ClockBacterium.beta_s * s - ClockBacterium.alpha_f * (
                (s * q ** ClockBacterium.n1) / (ClockBacterium.kappa_f ** ClockBacterium.n1 + q ** ClockBacterium.n1))

    @staticmethod
    def update_Q(s, e, n, a, o, q):
        return ClockBacterium.alpha_q * s * e * n - ClockBacterium.beta_q * a * o * q - ClockBacterium.gamma_q * q

    @staticmethod
    def update_F(s, q, f):
        return ClockBacterium.alpha_f * ((s * q ** ClockBacterium.n1) / (
                ClockBacterium.kappa_f ** ClockBacterium.n1 + q ** ClockBacterium.n1)) - \
               ClockBacterium.beta_f * f

    @staticmethod
    def update_T(t, f):
        return ClockBacterium.alpha_t * (ClockBacterium.t_t - t) - ClockBacterium.beta_t * f * t

    @staticmethod
    def update_A(t, a):
        return ClockBacterium.alpha_a * (
                (ClockBacterium.kappa_t ** ClockBacterium.n2) / (
                ClockBacterium.kappa_t ** ClockBacterium.n2 + t ** ClockBacterium.n2)) - ClockBacterium.beta_a * a

    @staticmethod
    def update_R(t, r):
        return ClockBacterium.alpha_r * ((t ** ClockBacterium.n2) / (
                ClockBacterium.kappa_r ** ClockBacterium.n2 + t ** ClockBacterium.n2)) - ClockBacterium.beta_r * r - ClockBacterium.gamma_r * r

    @staticmethod
    def update_W():
        return ClockBacterium.eta

    def metabolize(self, t, **kwargs):
        dt, k = kwargs["dt"], kwargs["k"]
        self.y += dt * self.NasA_oscIII_eta(y=self.y, t=t, k=k)

    @staticmethod
    def deltas(y):
        return np.array([ClockBacterium.update_Q(s=y[4], e=y[5], n=y[6], a=y[3], o=y[7], q=y[0]),
                         ClockBacterium.update_F(s=y[4], q=y[0], f=y[1]),
                         ClockBacterium.update_T(t=y[2], f=y[1]),
                         ClockBacterium.update_A(t=y[2], a=y[3]),
                         ClockBacterium.update_S(s=y[4], q=y[0]),
                         ClockBacterium.update_E(e=y[5], a=y[3], o=y[7], q=y[0], s=y[4], n=y[6]),
                         ClockBacterium.update_N(e=y[5], s=y[4], n=y[6]),
                         ClockBacterium.update_O(e=y[5], a=y[3], o=y[7], q=y[0]),
                         ClockBacterium.update_R(t=y[2], r=y[8])
                         ])

    @staticmethod
    def NasA_oscIII_D(y, t):
        dy = ClockBacterium.deltas(y=y)
        dy *= ClockBacterium.epsilon
        return dy

    @staticmethod
    def NasA_oscIII_eta(y, t, k):
        dy = ClockBacterium.deltas(y=y)
        dy *= (ClockBacterium.epsilon / (1.0 + ClockBacterium.eta * k))
        return dy

    def draw(self, min_val, max_val):
        import matplotlib.pyplot as plt
        c = plt.cm.Greens((self.y[8] - min_val) / (max_val - min_val))
        return c[0] * 255.0, c[1] * 255.0, c[2] * 255.0
