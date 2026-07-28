from itertools import combinations
from math import ceil

import numpy as np
from cmaes import CMA


def create_overlapping_components(adj_matrix):
    """Create OCC components from an interaction matrix.

    This follows the local CBCCO-style grouping already used in this
    repository: each component is built around a low-degree variable and its
    interacting variables, and overlaps are preserved.
    """
    n = len(adj_matrix)
    variables = set(range(n))
    graph = {i: set() for i in range(n)}
    for i, j in combinations(range(n), 2):
        if adj_matrix[i, j] == 1:
            graph[i].add(j)
            graph[j].add(i)

    remaining = variables.copy()
    components = []
    while remaining:
        ordered = list(remaining)
        degrees = [len(graph[v]) for v in ordered]
        center = ordered[int(np.argmin(degrees))]

        component = {center}
        component.update(graph[center])
        remaining -= component
        components.append(component)

    merged = True
    while merged:
        merged = False
        i = 1
        while i < len(components):
            for j in range(i):
                overlap = components[i] & components[j]
                if not overlap:
                    continue

                ratio_i = len(overlap) / len(components[i])
                ratio_j = len(overlap) / len(components[j])
                if max(ratio_i, ratio_j) < 1:
                    continue

                components[i] |= components[j]
                del components[j]
                merged = True
                break
            else:
                i += 1

    return [np.asarray(sorted(component), dtype=int) for component in components]


class ComponentOptimizer:
    def __init__(self, dim, bounds, seed=None):
        self.dim = dim
        self.bounds = bounds
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.best_x = self.rng.uniform(bounds[:, 0], bounds[:, 1], dim)
        self.best_f = np.inf

        if dim == 1:
            self._cma = None
            self.sigma = 0.3 * (bounds[0, 1] - bounds[0, 0])
            self.population_size = 10
            return

        self.sigma = 30
        self._cma = CMA(
            mean=self.best_x.copy(),
            sigma=self.sigma,
            bounds=bounds,
            seed=seed,
        )
        self.population_size = self._cma.population_size

    def ask(self):
        if self.dim == 1:
            noise = self.rng.normal(size=self.population_size)
            samples = self.best_x[0] + self.sigma * noise
            samples = np.clip(samples, self.bounds[0, 0], self.bounds[0, 1])
            return samples.reshape(-1, 1)

        if self._cma.should_stop():
            self._cma = CMA(
                mean=self.best_x.copy(),
                sigma=self.sigma,
                bounds=self.bounds,
                seed=self.seed,
            )

        return np.asarray([self._cma.ask() for _ in range(self.population_size)])

    def tell(self, solutions, fitness_list):
        old_best = self.best_f
        min_idx = int(np.argmin(fitness_list))
        if fitness_list[min_idx] < self.best_f:
            self.best_f = fitness_list[min_idx]
            self.best_x = solutions[min_idx].copy()

        if self.dim == 1:
            if fitness_list[min_idx] < old_best:
                self.sigma *= 1.2
            else:
                self.sigma *= 0.82
            width = self.bounds[0, 1] - self.bounds[0, 0]
            self.sigma = float(np.clip(self.sigma, 1e-12, width))
            return

        self._cma.tell([(x, f) for x, f in zip(solutions, fitness_list)])


class OCC:
    def __init__(
            self,
            func,
            adj_matrix,
            bounds,
            max_fe,
            seed,
            award_ratio=0.25,
            components=None,
    ):
        self.func = func.compute
        self.dim = func.D
        self.bounds = bounds
        self.max_fe = int(max_fe)
        self.remain_fe = int(max_fe)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.award_ratio = award_ratio

        if components is None:
            components = create_overlapping_components(adj_matrix)

        # The interaction matrix is in graph coordinates. The objective uses
        # permuted coordinates, matching the convention in test_graph.py.
        self.components = [np.asarray(func.p[c], dtype=int) for c in components]
        self.raw_components = [np.asarray(c, dtype=int) for c in components]
        self.optimizers = [
            ComponentOptimizer(len(c), bounds=self.bounds[c], seed=seed)
            for c in self.components
        ]

        self.adjacent = self._build_component_adjacency(self.raw_components)
        self.iters = np.ones(len(self.components), dtype=int)
        self.contrs = np.zeros(len(self.components), dtype=float)

        self.best_x = self.rng.uniform(bounds[:, 0], bounds[:, 1], self.dim)
        self.best_f = self._evaluate(self.best_x)

    @staticmethod
    def _build_component_adjacency(components):
        sets = [set(c.tolist()) for c in components]
        adjacent = [set() for _ in components]
        for i, j in combinations(range(len(components)), 2):
            if sets[i] & sets[j]:
                adjacent[i].add(j)
                adjacent[j].add(i)
        return adjacent

    def _evaluate(self, x):
        self.remain_fe -= 1
        return self.func(x)

    def _optimize_once(self, idx):
        optimizer = self.optimizers[idx]
        variables = self.components[idx]
        context = self.best_x.copy()

        solutions = optimizer.ask()
        fitness_list = []
        best_candidate_x = None
        best_candidate_f = np.inf

        for solution in solutions:
            context[variables] = solution
            fitness = self._evaluate(context)
            fitness_list.append(fitness)
            if fitness < best_candidate_f:
                best_candidate_f = fitness
                best_candidate_x = context.copy()

        if not fitness_list:
            return False

        used_solutions = solutions[:len(fitness_list)]
        optimizer.tell(used_solutions, fitness_list)

        if best_candidate_f < self.best_f:
            improvement = self.best_f - best_candidate_f
            self.contrs[idx] = 0.5 * self.contrs[idx] + 0.5 * improvement
            self.best_f = best_candidate_f
            self.best_x = best_candidate_x
            return True

        self.contrs[idx] *= 0.5
        return False

    def _update_iter_counters(self, idx, iters_to_add):
        for adj_idx in self.adjacent[idx]:
            iters_to_add[adj_idx] += 1
        self.iters[idx] = max(int(self.iters[idx]) - 2, 0)
        iters_to_add[idx] = 0

    def _award_list(self):
        positive = np.flatnonzero(self.contrs > 0)
        if len(positive) == 0:
            return []

        ordered = sorted(positive, key=lambda i: self.contrs[i], reverse=True)
        count = max(1, ceil(len(ordered) * self.award_ratio))
        return ordered[:count]

    def run(self):
        cycle = 0
        component_indices = np.arange(len(self.components))

        while self.remain_fe > 0:
            cycle += 1
            self.rng.shuffle(component_indices)
            iters_to_add = np.zeros(len(self.components), dtype=int)

            for idx in component_indices:
                if self.remain_fe <= 0:
                    break

                improved = False
                max_iters = min(int(self.iters[idx]), len(self.components[idx]))
                for _ in range(max_iters):
                    if self.remain_fe <= 0:
                        break
                    if self._optimize_once(idx):
                        improved = True

                if improved:
                    self._update_iter_counters(idx, iters_to_add)

            award = self._award_list()
            self.rng.shuffle(award)
            for idx in award:
                if self.remain_fe <= 0:
                    break
                if self._optimize_once(idx):
                    self._update_iter_counters(idx, iters_to_add)

            self.iters += iters_to_add

            if cycle % 10 == 0:
                print(f'{self.remain_fe}:{self.best_f:e}')
