"""Minimal CMA-ES (minimization) in pure numpy.

Follows Hansen, "The CMA Evolution Strategy: A Tutorial",
arXiv:1604.00772, using the default parameter settings from its appendix
(same constants as Hansen's purecma reference code). No restarts, no
boundary handling: the search space here is unbounded controller weights.
"""

import numpy as np


class CMAES:
    def __init__(self, x0, sigma0: float, popsize: int | None = None, seed: int = 0):
        self.n = len(x0)
        n = self.n
        self.mean = np.asarray(x0, dtype=np.float64).copy()
        self.sigma = float(sigma0)
        self.lam = int(popsize) if popsize else 4 + int(3 * np.log(n))
        self.mu = self.lam // 2
        w = np.log((self.lam + 1) / 2.0) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.gen = 0
        self.rng = np.random.default_rng(seed)
        self._eigen()

    def _eigen(self):
        self.C = (self.C + self.C.T) / 2.0
        d2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(d2, 1e-20))

    def ask(self) -> np.ndarray:
        """Sample lam candidates, shape (lam, n)."""
        z = self.rng.standard_normal((self.lam, self.n))
        y = (z * self.D) @ self.B.T
        return self.mean + self.sigma * y

    def tell(self, xs: np.ndarray, fitnesses):
        """Update from candidates xs and their fitness values (lower is better)."""
        idx = np.argsort(np.asarray(fitnesses))[: self.mu]
        y = (np.asarray(xs)[idx] - self.mean) / self.sigma
        y_w = self.weights @ y
        self.mean = self.mean + self.sigma * y_w

        c_inv_half_yw = self.B @ ((self.B.T @ y_w) / self.D)
        self.ps = ((1 - self.cs) * self.ps
                   + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * c_inv_half_yw)
        self.gen += 1
        hsig = (np.linalg.norm(self.ps)
                / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen))
                / self.chi_n) < 1.4 + 2 / (self.n + 1)
        self.pc = ((1 - self.cc) * self.pc
                   + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w)

        rank_mu = (y * self.weights[:, None]).T @ y
        delta = (1 - hsig) * self.cc * (2 - self.cc)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + delta * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= np.exp((self.cs / self.damps)
                             * (np.linalg.norm(self.ps) / self.chi_n - 1))
        self._eigen()
