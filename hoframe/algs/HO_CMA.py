import math
from typing import List

import numpy as np

from algs.optimizer import EaOptimizer, CmaEsOptimizer, OneDES


class EvolveTree:
    def __init__(self, level, opz: EaOptimizer, variables: np.ndarray, children: List['EvolveTree'],
                 parent: 'EvolveTree' = None):
        self.level = level
        self.opz = opz
        self.variables = variables
        self.children = children
        self.parent = parent
        self.ancestors = []

        # runtime data
        self.rtx = None

    def init(self, x0):
        # call init after built the tree
        self.rtx = x0.copy()
        for child in self.children:
            child.ancestors += self.ancestors.copy()
            child.ancestors.append(self)
            child.init(x0)

    def push_to_ancestors(self, best_x_):
        # push best_x of current node to its ancestors
        self.rtx[self.variables] = best_x_
        for ancestor in self.ancestors:
            ancestor.rtx[self.variables] = best_x_

    def pull_from_ancestors(self):
        # pull best xes of ancestors to current node
        for ancestor in self.ancestors:
            if ancestor.opz.best_x is None:
                self.rtx[ancestor.variables] = np.mean(ancestor.opz.ask(), axis=0)
            else:
                self.rtx[ancestor.variables] = ancestor.opz.best_x

    def dfs(self):
        yield self
        for child in self.children:
            yield from child.dfs()


class ResourceAllocator:
    def __init__(self, evt, max_fe=30000):
        self.evt = evt
        self.max_fe = max_fe
        self.remain_fe = max_fe

        self.gen = 0

    def _allocate_1(self, evt_list):
        for evt in evt_list:
            evt.opz.gen = 1

    def _allocate_level(self, evt_list):
        for evt in evt_list:
            evt.opz.gen = evt.level + 1

    def _allocate_3level(self, evt_list):
        for evt in evt_list:
            evt.opz.gen = 2 ** evt.level

    def _allocate_bc(self, evt_list):
        values = [evt.opz.recent_5()[-1] for evt in evt_list]
        positive_indices = [i for i, v in enumerate(values) if v > 0]
        sorted_pos_indices = sorted(positive_indices, key=lambda i: values[i])

        m = len(sorted_pos_indices)
        k = math.ceil(m * 2 / 3)

        for evt in evt_list:
            evt.opz.gen = 1
        for rank, idx in enumerate(sorted_pos_indices):
            if rank > k:
                evt_list[idx].opz.gen = 2

    def allocate(self):
        evt_list = list(self.evt.dfs())

        self._allocate_level(evt_list)

        self.gen += 1

    def consume(self, fe):
        self.remain_fe -= fe

    def continue_condition(self):
        return self.remain_fe > 0


class HoGa:
    def __init__(self, func, vst, bounds, max_fe=3000, seed=0):
        self.func = func.compute
        self.dim = func.D
        self.p = func.p
        self.bounds = bounds

        # runtime x
        self.rtx = np.zeros(func.D, dtype=np.float64)

        # record best one
        self.best_x = None
        self.best_f = np.inf

        # utils
        self.evt = self._init_evt(0, vst, seed=seed)
        self.ra = ResourceAllocator(self.evt, max_fe)

    def _init_evt(self, level, vst, seed):
        cut, children = vst
        evt_children = []
        for child in children:
            evt_children.append(self._init_evt(level + 1, child, seed))
        variables = np.asarray(cut)
        variables = self.p[variables]
        # ea = SimpleGA(len(cut), pop_size=50, bounds=self.bounds[variables])
        dim = len(cut)
        if dim == 1:
            ea = OneDES(-100, 100, seed=seed)
        else:
            ea = CmaEsOptimizer(len(cut), bounds=self.bounds[variables], seed=seed)
        self.rtx[variables] = ea.best_x.copy()
        cur_node = EvolveTree(level, ea, variables, evt_children)
        for evt_child in evt_children:
            evt_child.parent = cur_node
        return cur_node

    def run(self):
        i = 0
        while self.ra.continue_condition():
            self.ra.allocate()
            self.compute(self.evt)

            i += 1
            if i % 100 == 0:
                print(f'{self.ra.remain_fe}:{self.best_f:e}')

    def compute(self, evt: EvolveTree):
        for child in evt.children:
            self.compute(child)

        opz = evt.opz
        temp_x = self.rtx.copy()

        while opz.continue_condition():
            _x_list_ = evt.opz.ask()
            fitness_list = []
            for _x_ in _x_list_:
                temp_x[evt.variables] = _x_
                fitness = self.func(temp_x)
                fitness_list.append(fitness)
                if fitness < self.best_f:
                    self.best_f = fitness
                    self.best_x = temp_x.copy()
            self.ra.consume(len(_x_list_))
            opz.tell(fitness_list)

        self.rtx[evt.variables] = opz.best_x.copy()
