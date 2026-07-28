import argparse
import os

import numpy as np
from joblib import Parallel, delayed

from algs.CBCCO_CMA import CBCCO
from algs.CC_CMA import CC
from algs.OCC_CMA import OCC
from algs.rdg3_group import rdg3_group
from algs.vst_group import vst_group
from algs.vstd_group import vstd_group
from cached_init import load_or_init_graph
from randg import GraphFunction, adjacent_matrix_from_datum

try:
    import setproctitle
except ImportError:
    setproctitle = None


CC_GROUP_METHODS = {
    'RDG3': rdg3_group,
    'vst': vst_group,
    'vstd': vstd_group,
}


def parse_int_list(value):
    result = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result


def set_worker_title(name, func_id):
    if setproctitle is not None:
        setproctitle.setproctitle(f"jing_python_worker_{name}_{func_id}")


def load_graph_function(func_id):
    graph_data = load_or_init_graph()
    datum = graph_data[func_id - 1]
    func = GraphFunction(datum)
    adj_matrix = adjacent_matrix_from_datum(datum)
    bounds = np.asarray([[-100, 100]] * func.D)
    return func, adj_matrix, bounds


def write_record(name, func_id, seed, best_f):
    os.makedirs('record', exist_ok=True)
    with open(f'record/{name}.txt', 'a') as f:
        f.write(f"f{func_id}:{seed}:{best_f}\n")


def run_cc_group(name, func_id, seed, max_fe):
    set_worker_title(name, func_id)
    np.random.seed(seed)

    func, adj_matrix, bounds = load_graph_function(func_id)
    method = CC_GROUP_METHODS[name]
    group_list = [func.p[variables] for variables in method(adj_matrix)]

    optimizer = CC(func, group_list, bounds, max_fe, seed)
    optimizer.run()

    print(f'{name}:f{func_id}:seed{seed}:{optimizer.remain_fe}:{optimizer.best_f:e}')
    write_record(name, func_id, seed, optimizer.best_f)


def run_occ(name, func_id, seed, max_fe):
    set_worker_title(name, func_id)
    np.random.seed(seed)

    func, adj_matrix, bounds = load_graph_function(func_id)
    optimizer = OCC(
        func=func,
        adj_matrix=adj_matrix,
        bounds=bounds,
        max_fe=max_fe,
        seed=seed,
    )
    optimizer.run()

    print(f'{name}:f{func_id}:seed{seed}:{optimizer.remain_fe}:{optimizer.best_f:e}')
    write_record(name, func_id, seed, optimizer.best_f)


def run_cbcco(name, func_id, seed, max_fe, test_generations):
    set_worker_title(name, func_id)
    np.random.seed(seed)

    func, adj_matrix, bounds = load_graph_function(func_id)
    optimizer = CBCCO(
        func=func,
        adj_matrix=adj_matrix,
        bounds=bounds,
        max_fe=max_fe,
        seed=seed,
        test_generations=test_generations,
    )
    optimizer.run()

    print(f'{name}:f{func_id}:seed{seed}:{optimizer.remain_fe}:{optimizer.best_f:e}')
    write_record(name, func_id, seed, optimizer.best_f)


def build_tasks(algorithms, func_ids, seeds, max_fe, test_generations):
    tasks = []
    for seed in seeds:
        for func_id in func_ids:
            for name in algorithms:
                if name in CC_GROUP_METHODS:
                    tasks.append(delayed(run_cc_group)(name, func_id, seed, max_fe))
                elif name == 'OCC':
                    tasks.append(delayed(run_occ)(name, func_id, seed, max_fe))
                elif name == 'CBCCO':
                    tasks.append(delayed(run_cbcco)(
                        name,
                        func_id,
                        seed,
                        max_fe,
                        test_generations,
                    ))
                else:
                    raise ValueError(f"Unknown algorithm: {name}")
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Run OCC, CBCCO, RDG3, vst, and vstd on graph benchmarks."
    )
    parser.add_argument('--algorithms', default='OCC,CBCCO,RDG3,vst,vstd')
    parser.add_argument('--functions', default='1-10')
    parser.add_argument('--seeds', default='0-19')
    parser.add_argument('--max-fe', type=int, default=100000)
    parser.add_argument('--test-generations', type=int, default=5)
    parser.add_argument('--jobs', type=int, default=None)
    args = parser.parse_args()

    algorithms = [item.strip() for item in args.algorithms.split(',') if item.strip()]
    func_ids = parse_int_list(args.functions)
    seeds = parse_int_list(args.seeds)

    tasks = build_tasks(
        algorithms=algorithms,
        func_ids=func_ids,
        seeds=seeds,
        max_fe=args.max_fe,
        test_generations=args.test_generations,
    )

    if args.jobs is None:
        total_cores = os.cpu_count() or 1
        jobs = max(1, total_cores - 2)
    else:
        jobs = args.jobs

    with Parallel(n_jobs=jobs, verbose=10, max_nbytes=0) as parallel:
        print(f"Starting {len(tasks)} graph tasks for: {', '.join(algorithms)}")
        parallel(tasks)

    print("All graph tasks finished")


if __name__ == '__main__':
    main()
