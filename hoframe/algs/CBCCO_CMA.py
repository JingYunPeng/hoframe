from itertools import combinations

import numpy as np
from cmaes import CMA


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


def nonshared_variable_allocation(adj_matrix, zeta=0.3):
    n = len(adj_matrix)
    variables = set(range(n))
    graph = {i: set() for i in range(n)}
    for i, j in combinations(range(n), 2):
        if adj_matrix[i, j] == 1:
            graph[i].add(j)
            graph[j].add(i)

    remaining = variables.copy()
    groups = []
    while remaining:
        ordered = list(remaining)
        degrees = [len(graph[v]) for v in ordered]
        center = ordered[int(np.argmin(degrees))]
        group = {center}
        group.update(graph[center])
        remaining -= group
        groups.append(group)

    overlaps = []
    i = 1
    while i < len(groups):
        for j in range(i):
            overlap = groups[i] & groups[j]
            if not overlap:
                continue

            ratio_i = len(overlap) / len(groups[i])
            ratio_j = len(overlap) / len(groups[j])
            if ratio_i >= zeta or ratio_j >= zeta:
                groups[i] |= groups[j]
                del groups[j]
                overlaps = [
                    item for item in overlaps
                    if item[0] != j and item[1] != j
                ]
                overlaps = [
                    (
                        item[0] - 1 if item[0] > j else item[0],
                        item[1] - 1 if item[1] > j else item[1],
                        item[2],
                    )
                    for item in overlaps
                ]
                i = max(i - 1, 1)
                break

            overlaps.append((i, j, set(overlap)))
        else:
            i += 1

    for _, _, overlap in overlaps:
        for group in groups:
            group -= overlap

    groups = [group for group in groups if group]
    return groups, overlaps


class CBCCO:
    def __init__(
            self,
            func,
            adj_matrix,
            bounds,
            max_fe,
            seed,
            zeta=0.3,
            test_generations=5,
    ):
        self.func = func.compute
        self.dim = func.D
        self.p = func.p
        self.bounds = bounds
        self.max_fe = int(max_fe)
        self.remain_fe = int(max_fe)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.zeta = zeta
        self.test_generations = test_generations

        groups, overlaps = nonshared_variable_allocation(adj_matrix, zeta=zeta)
        self.raw_nonshared_groups = [set(g) for g in groups]
        self.overlaps = overlaps

        self.best_x = np.zeros(self.dim, dtype=np.float64)
        self.best_f = self._evaluate(self.best_x)

        self.contrs = np.zeros(len(self.raw_nonshared_groups), dtype=float)
        self._shared_variable_allocation()
        self._reset_final_optimizers()

    def _evaluate(self, x):
        self.remain_fe -= 1
        return self.func(x)

    def _optimize_generation(self, optimizer, variables, update_global=True):
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

        optimizer.tell(solutions, fitness_list)

        old_best = self.best_f
        if update_global and best_candidate_f < self.best_f:
            self.best_f = best_candidate_f
            self.best_x = best_candidate_x

        return old_best - self.best_f if update_global else old_best - best_candidate_f

    def _shared_variable_allocation(self):
        optimizers = []
        mapped_groups = []
        for group in self.raw_nonshared_groups:
            variables = np.asarray(self.p[np.asarray(sorted(group), dtype=int)], dtype=int)
            mapped_groups.append(variables)
            optimizers.append(
                ComponentOptimizer(len(variables), self.bounds[variables], seed=self.seed)
            )

        for idx, (variables, optimizer) in enumerate(zip(mapped_groups, optimizers)):
            old_best = self.best_f
            for _ in range(self.test_generations):
                if self.remain_fe <= 0:
                    break
                self._optimize_generation(optimizer, variables, update_global=True)
            self.contrs[idx] = old_best - self.best_f

        final_groups = [set(group) for group in self.raw_nonshared_groups]
        for i, j, overlap in self.overlaps:
            if i >= len(final_groups) or j >= len(final_groups):
                continue
            if self.contrs[i] > self.contrs[j]:
                final_groups[i].update(overlap)
            else:
                final_groups[j].update(overlap)

        assigned = set().union(*final_groups) if final_groups else set()
        missing = set(range(self.dim)) - assigned
        if missing:
            if final_groups:
                final_groups[int(np.argmax(self.contrs))].update(missing)
            else:
                final_groups.append(missing)

        self.raw_groups = [set(g) for g in final_groups if g]
        self.groups = [
            np.asarray(self.p[np.asarray(sorted(group), dtype=int)], dtype=int)
            for group in self.raw_groups
        ]
        self.contrs = np.resize(self.contrs, len(self.groups))

    def _reset_final_optimizers(self):
        self.optimizers = [
            ComponentOptimizer(len(group), self.bounds[group], seed=self.seed)
            for group in self.groups
        ]

    def _update_contribution(self, idx, improvement):
        self.contrs[idx] = (self.contrs[idx] + improvement) / 2.0

    def _award_list(self):
        if len(self.contrs) == 0:
            return []
        eta_max = float(np.max(self.contrs))
        if eta_max <= 0:
            return []

        award = [
            idx for idx, eta in enumerate(self.contrs)
            if eta > 0 and eta_max / eta < 2
        ]
        if len(award) == len(self.groups):
            return []
        return award

    def run(self):
        cycle = 0
        while self.remain_fe > 0:
            cycle += 1

            for idx, (variables, optimizer) in enumerate(zip(self.groups, self.optimizers)):
                if self.remain_fe <= 0:
                    break
                improvement = self._optimize_generation(optimizer, variables)
                self._update_contribution(idx, improvement)

            award = self._award_list()
            for idx in award:
                if self.remain_fe <= 0:
                    break
                improvement = self._optimize_generation(
                    self.optimizers[idx],
                    self.groups[idx],
                )
                self._update_contribution(idx, improvement)

            if cycle % 10 == 0:
                print(f'{self.remain_fe}:{self.best_f:e}')
