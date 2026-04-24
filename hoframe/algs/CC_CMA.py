import numpy as np

from algs.optimizer import CmaEsOptimizer


class CC:
    def __init__(self, func, group_list, bounds, max_fe, seed):
        self.func = func.compute
        self.dim = func.D
        self.bounds = bounds
        self.remain_fe = max_fe

        # runtime x
        self.rtx = np.zeros(func.D, dtype=np.float64)

        # record best one
        self.best_x = None
        self.best_f = np.inf

        self.group_list = group_list
        self.optimizers = []

        for group in group_list:
            opz = CmaEsOptimizer(len(group), bounds=self.bounds[group], seed=seed)
            self.optimizers.append((group, opz))

    def allocate(self):
        for _, opz in self.optimizers:
            opz.gen = 1

    def run(self):
        i = 0
        while self.remain_fe > 0:
            i += 1
            if i % 1 == 0:
                print(f'{self.remain_fe}:{self.best_f:e}')

            self.allocate()
            for variables, opz in self.optimizers:
                temp_x = self.rtx.copy()
                while opz.continue_condition():
                    _x_list_ = opz.ask()
                    fitness_list = []
                    for _x_ in _x_list_:
                        temp_x[variables] = _x_
                        fitness = self.func(temp_x)
                        fitness_list.append(fitness)
                        if fitness < self.best_f:
                            self.best_f = fitness
                            self.best_x = temp_x.copy()
                    self.remain_fe -= len(fitness_list)
                    opz.tell(fitness_list)

                self.rtx[variables] = opz.best_x.copy()
