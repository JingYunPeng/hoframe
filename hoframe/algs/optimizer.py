from collections import deque

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List

from cmaes import CMA


class EaOptimizer(ABC):
    @abstractmethod
    def ask(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def tell(self, fitness_list) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_result(self):
        raise NotImplementedError

    @abstractmethod
    def continue_condition(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def recent_5(self):
        raise NotImplementedError


class OneDES(EaOptimizer):
    def __init__(
            self,
            lb,
            ub,
            lam=10,
            seed=None,
    ):
        self.lb = lb
        self.ub = ub
        self.rng = np.random.default_rng(seed)

        self.lam = lam
        self.gen = 1000

        self.best_x = self.rng.uniform(lb, ub, 1)

        self.best_f = np.inf
        self.sigma = 0.3 * (ub - lb)

        self.iter = 0
        self.history = deque(maxlen=5)
        self.history.append(0)

        self._pop = None
        self._asked = False

    def ask(self) -> np.ndarray:
        if not self._asked:
            noise = self.rng.normal(size=self.lam)
            self._pop = self.best_x + self.sigma * noise
            self._pop = np.clip(self._pop, self.lb, self.ub)
            self._asked = True
        return self._pop

    def tell(self, fitness_list) -> np.ndarray:

        idx = np.argmin(fitness_list)
        x_new = self._pop[idx]
        f_new = fitness_list[idx]

        if f_new < self.best_f:
            self.best_x = np.asarray([x_new])
            self.best_f = f_new
            self.sigma *= 1.5
        else:
            self.sigma *= 0.82

        # 防止 sigma 过小或过大
        self.sigma = np.clip(self.sigma, 1e-12, (self.ub - self.lb))

        self.iter += 1
        self.history.append(self.best_f)

        self.gen -= 1
        self._asked = False

    def get_result(self):
        return self.best_x, self.best_f

    def continue_condition(self) -> bool:
        return self.gen > 0

    def recent_5(self):
        return self.history


class CmaEsOptimizer(EaOptimizer):
    def __init__(
            self,
            dim: int,
            sigma: float = 30,
            gen: int = 1000,
            bounds: Optional[np.ndarray] = None,
            seed: Optional[int] = None,
    ):
        mean = np.random.uniform(bounds[:, 0], bounds[:, 1], dim)

        self.sigma = sigma
        self.bounds = bounds
        self.seed = seed

        self.gen = gen
        self._cma = CMA(
            mean=mean,
            sigma=sigma,
            bounds=bounds,
            seed=seed,
        )

        self._solutions = None  # 当前 ask 的解缓存
        self._asked = False
        self.best_x = mean
        self.best_f = np.inf
        self._history = deque(maxlen=5)  # 存储最近5代
        self._history.append(0)

    def ask(self) -> np.ndarray:
        """
        返回当前一代的候选解
        shape: (pop_size, dim)
        """
        if not self._asked:
            solutions = []
            for _ in range(self._cma.population_size):
                x = self._cma.ask()
                solutions.append(x)

            self._solutions = np.array(solutions)
            self._asked = True
        return self._solutions

    def tell(self, fitness_list: List[float]) -> np.ndarray:
        """
        输入每个解的 fitness（越小越好）
        """
        assert self._solutions is not None, "ask() must be called before tell()"
        assert len(fitness_list) == len(self._solutions)

        solutions_with_fitness = [
            (x, f) for x, f in zip(self._solutions, fitness_list)
        ]

        self._cma.tell(solutions_with_fitness)

        # 更新 best
        min_idx = int(np.argmin(fitness_list))

        self._history.append(self.best_f - fitness_list[min_idx])

        if fitness_list[min_idx] < self.best_f:
            self.best_f = fitness_list[min_idx]
            self.best_x = self._solutions[min_idx].copy()

        self.gen -= 1
        self._asked = False

        return self._solutions

    def get_result(self):
        """
        返回最优解
        """
        return self.best_x, self.best_f

    def continue_condition(self) -> bool:
        """
        是否继续优化
        """
        if self.gen <= 0:
            return False

        # 无限重启直到资源耗尽
        if self._cma.should_stop():
            self._cma = CMA(
                mean=self.best_x,
                sigma=self.sigma,
                bounds=self.bounds,
                seed=self.seed,
            )

        return True

    def recent_5(self):
        return self._history
