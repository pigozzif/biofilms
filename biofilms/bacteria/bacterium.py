import abc

from scipy.integrate import quad
import numpy as np


class Bacterium(abc.ABC):

    def __init__(self, idx):
        self.idx = idx

    @abc.abstractmethod
    def propagate(self, lattice, t, dt, d):
        pass

    @abc.abstractmethod
    def draw(self, t, min_val, max_val):
        pass


class SignallingBacterium(Bacterium):
    epsilon = 10.0
    u0 = 0.02
    firing_threshold = 0.6

    def __init__(self, idx, u_init, phi_c):
        super().__init__(idx)
        # self.ut = u_init
        self.u_old = u_init
        # self.firing = random.random() <= phi_c[0]
        # self.ut = 1.0 if self.firing else 0.0
        self.integral = 0.0
        self.us = []

    def _is_firing(self, ut):
        return ut > self.firing_threshold

    def _compute_t_prime(self):
        return self.us[-1]

    # @staticmethod
    # def _compute_delta(lattice, t, dt):
    #     tau = 300 if self.firing else 5
    #     self.integral = dt * (quad(lambda x: self._compute_t_prime(x, dt), t - dt, t)[0] + self.integral)
    #     messages = sum([lattice._get_coupling(self.idx, neigh["cell"].idx) * (neigh["cell"].u_old - self.ut)
    #                     for neigh in lattice.get_neighborhood(self)])
    #     if self.idx == 0:
    #         print(round(t / dt), self.ut, self.firing, self.integral, messages)
    #     return self.epsilon * (self.ut * (1 - self.ut) * (self.ut - self.u0) - self.integral / tau) + messages

    # def propagate(self, lattice, t, dt):
    #     du = dt * self._compute_delta(lattice=lattice, t=t, dt=dt)
    #     self.ut += du
    #     self.us.append(self.ut)
    #     self.u_old = self.ut
    #     self.firing = self._is_firing(self.u_old)

    def propagate(self, lattice, t, dt, d):
        return

    def FitzHughNagumo_percolate(self, t, y, lattice, dt):
        self.us.append(y[self.idx])
        tau = 300 if self._is_firing(ut=y[self.idx]) else 5
        self.integral += quad(lambda x: self._compute_t_prime(), t - dt, t)[0]
        messages = sum([lattice._get_coupling(self.idx, neigh["cell"].idx) * (y[neigh["cell"].idx] - y[self.idx])
                        for neigh in lattice.get_neighborhood(self)])
        dy = self.epsilon * (y[self.idx] * (1 - y[self.idx]) * (y[self.idx] - self.u0) - self.integral / tau) + messages
        return dy

    def draw(self, t, min_val, max_val):
        raise NotImplementedError


class ClockBacterium(Bacterium):
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

    def __init__(self, idx):
        super().__init__(idx)
        self.is_frontier = True
        self.age = 0

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
        return ClockBacterium.alpha_f * ((s * q ** ClockBacterium.n1) / (ClockBacterium.kappa_f ** ClockBacterium.n1 + q ** ClockBacterium.n1)) -\
               ClockBacterium.beta_f * f

    @staticmethod
    def update_T(t, f):
        return ClockBacterium.alpha_t * (ClockBacterium.t_t - t) - ClockBacterium.beta_t * f * t

    @staticmethod
    def update_A(t, a):
        return ClockBacterium.alpha_a * (
                (ClockBacterium.kappa_t ** ClockBacterium.n2) / (ClockBacterium.kappa_t ** ClockBacterium.n2 + t ** ClockBacterium.n2)) - ClockBacterium.beta_a * a

    @staticmethod
    def update_R(t, r):
        return ClockBacterium.alpha_r * ((t ** ClockBacterium.n2) / (
                ClockBacterium.kappa_r ** ClockBacterium.n2 + t ** ClockBacterium.n2)) - ClockBacterium.beta_r * r - ClockBacterium.gamma_r * r

    @staticmethod
    def update_W():
        return ClockBacterium.eta

    def propagate(self, lattice, t, dt, d):
        return

    @staticmethod
    def _deltas(y):
        return np.array([ClockBacterium.update_Q(s=y[4], e=y[5], n=y[6], a=y[3], o=y[7], q=y[0]),
                         ClockBacterium.update_F(s=y[4], q=y[0], f=y[1]),
                         ClockBacterium.update_T(t=y[2], f=y[1]),
                         ClockBacterium.update_A(t=y[2], a=y[3]),
                         ClockBacterium.update_S(s=y[4], q=y[0]),
                         ClockBacterium.update_E(e=y[5], a=y[3], o=y[7], q=y[0], s=y[4], n=y[6]),
                         ClockBacterium.update_N(e=y[5], s=y[4], n=y[6]),
                         ClockBacterium.update_O(e=y[5], a=y[3], o=y[7], q=y[0]),
                         ClockBacterium.update_R(t=y[2], r=y[8]),
                         ClockBacterium.update_W()
                         ])

    @staticmethod
    def NasA_oscIII_D(t, y):
        dy = ClockBacterium._deltas(y=y)
        dy *= ClockBacterium.epsilon
        return dy[: -1]

    @staticmethod
    def NasA_oscIII_eta(t, y):
        dy = ClockBacterium._deltas(y=y)
        dy[: -1] *= (ClockBacterium.epsilon / (1.0 + y[9]))
        return dy

    def draw(self, val, min_val, max_val):
        import matplotlib.pyplot as plt
        c = plt.cm.Greens((val - min_val) / (max_val - min_val))
        return c[0] * 255.0, c[1] * 255.0, c[2] * 255.0
