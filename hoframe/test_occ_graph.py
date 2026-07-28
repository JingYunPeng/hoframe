import argparse
import os

import numpy as np
from joblib import Parallel, delayed

from algs.OCC_CMA import OCC
from cached_init import load_or_init_graph
from randg import GraphFunction, adjacent_matrix_from_datum

try:
    import setproctitle
except ImportError:
    setproctitle = None


def run_occ(name, func_id, seed, max_fe=100000):
    if setproctitle is not None:
        setproctitle.setproctitle(f"jing_python_worker_{name}_{func_id}")

    np.random.seed(seed)

    graph_data = load_or_init_graph()
    datum = graph_data[func_id - 1]
    func = GraphFunction(datum)
    adj_matrix = adjacent_matrix_from_datum(datum)
    bounds = np.asarray([[-100, 100]] * func.D)

    optimizer = OCC(
        func=func,
        adj_matrix=adj_matrix,
        bounds=bounds,
        max_fe=max_fe,
        seed=seed,
    )
    optimizer.run()

    print(f'{optimizer.remain_fe}:{optimizer.best_f:e}')
    os.makedirs('record', exist_ok=True)
    with open(f'record/{name}.txt', 'a') as f:
        f.write(f"f{func_id}:{seed}:{optimizer.best_f}\n")


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


def main():
    parser = argparse.ArgumentParser(description="Run full OCC-CMA on graph benchmarks.")
    parser.add_argument('--name', default='occ_cma_graph')
    parser.add_argument('--functions', default='1-10')
    parser.add_argument('--seeds', default='0-10')
    parser.add_argument('--max-fe', type=int, default=100000)
    parser.add_argument('--jobs', type=int, default=None)
    args = parser.parse_args()

    func_ids = parse_int_list(args.functions)
    seeds = parse_int_list(args.seeds)

    tasks = [
        delayed(run_occ)(args.name, func_id, seed, args.max_fe)
        for seed in seeds
        for func_id in func_ids
    ]

    if args.jobs is None:
        total_cores = os.cpu_count() or 1
        jobs = max(1, total_cores - 2)
    else:
        jobs = args.jobs

    with Parallel(n_jobs=jobs, verbose=10, max_nbytes=0) as parallel:
        print(f"Starting {len(tasks)} OCC graph tasks")
        parallel(tasks)

    print("All OCC graph tasks finished")


if __name__ == '__main__':
    main()
