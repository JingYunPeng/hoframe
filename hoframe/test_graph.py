import os
from functools import lru_cache

import numpy as np
from joblib import Parallel, delayed
import setproctitle

from algs.CC_CMA import CC
from algs.occ_group import occ_group
from algs.rdg3_group import rdg3_group
from algs.vst_group import vst_group
from algs.vstd_group import vstd_group
from cached_init import load_or_init_graph
from randg import GraphFunction, adjacent_matrix_from_datum


def run(name, func_id, func, group_list, seed, max_fe=1e5):
    custom_name = f"jing_python_worker_{name}_{func_id}"
    setproctitle.setproctitle(custom_name)

    np.random.seed(seed)

    bounds = [[-100, 100]] * func.D
    c = CC(func, group_list, np.asarray(bounds), max_fe, seed)
    c.run()

    print(f'{c.remain_fe}:{c.best_f:e}')
    os.makedirs('record', exist_ok=True)
    with open(f'record/{name}.txt', 'a') as f:
        f.write(f"f{func_id}:{seed}:{c.best_f}\n")


group_methods = {
    'vst_graph': vst_group,
    'rdg3_graph': rdg3_group,
    'occ_graph': occ_group,
    'vstd_graph': vstd_group,
}

@lru_cache
def group(func_id):
    graph_data = load_or_init_graph()
    datum = graph_data[func_id - 1]
    func = GraphFunction(datum)

    m = adjacent_matrix_from_datum(datum)
    vst = vstd_group(m)
    group_list = [func.p[variables] for variables in vst]
    return group_list


def add_vstd():
    tasks = []

    graph_data = load_or_init_graph()
    for seed in range(11):
        for func_id in range(1, 11):
            datum = graph_data[func_id - 1]
            func = GraphFunction(datum)
            group_list = group(func_id)
            tasks.append(delayed(run)('vastd_graph', func_id, func, group_list, seed))
    total_cores = os.cpu_count()
    safe_jobs = max(1, total_cores - 2)

    with  Parallel(n_jobs=safe_jobs, verbose=10, max_nbytes=0) as parallel:
        print(f"开始并行执行 {len(tasks)} 个任务")
        parallel(tasks)

    print("所有任务完成!")


def main():
    tasks = []

    graph_data = load_or_init_graph()
    for seed in range(11):
        for func_id in range(1, 11):
            datum = graph_data[func_id - 1]
            func = GraphFunction(datum)

            m = adjacent_matrix_from_datum(datum)

            for name, method in group_methods.items():
                group_list = [func.p[variables] for variables in method(m)]
                tasks.append(delayed(run)(name, func_id, func, group_list, seed))

    total_cores = os.cpu_count()
    safe_jobs = max(1, total_cores - 2)

    with  Parallel(n_jobs=safe_jobs, verbose=10, max_nbytes=0) as parallel:
        print(f"开始并行执行 {len(tasks)} 个任务")
        parallel(tasks)

    print("所有任务完成!")


if __name__ == '__main__':
    add_vstd()
