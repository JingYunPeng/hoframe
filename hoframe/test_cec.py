import os
from functools import lru_cache
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed

from algs.CC_CMA import CC
from algs.HO_CMA import HoGa
from algs.vst_cec import compress
from cached_init import load_or_init_vst, get_f13, get_f14, load_or_init_rdg3, load_or_init_vstd


def run_hoga(overlap, seed, func, vst, max_fe=None):
    bounds = [[-100, 100]] * func.D
    np.random.seed(seed)

    if max_fe is None:
        max_fe = 3000000 - 0.5 * func.D * (func.D - 1)
    optimizer = HoGa(
        func=func,
        vst=vst,
        bounds=np.asarray(bounds),
        max_fe=max_fe,
        seed=seed)
    optimizer.run()
    best_f = optimizer.best_f
    print(optimizer.ra.remain_fe, f"{best_f:e}")

    fun_num = (0 if '13' in func.__class__.__name__.lower() else 10) + overlap

    os.makedirs('record', exist_ok=True)
    with open('record/hoga.txt', 'a') as f:
        f.write(f"f{fun_num}:{seed}:{best_f}\n")


def run_cc(name, overlap, func, group_list, seed, max_fe=100000):
    np.random.seed(seed)

    bounds = [[-100, 100]] * func.D
    c = CC(func, group_list, np.asarray(bounds), max_fe, seed)
    c.run()

    print(f'{c.remain_fe}:{c.best_f:e}')
    os.makedirs('record', exist_ok=True)
    fun_num = (0 if '13' in func.__class__.__name__.lower() else 10) + overlap
    with open(f'record/{name}.txt', 'a') as f:
        f.write(f"f{fun_num}:{seed}:{c.best_f}\n")


def compare_run(func_id, seed):
    conflicting = func_id > 10
    overlap = func_id - 10 if conflicting else func_id
    fun_method = get_f14 if conflicting else get_f13

    func = fun_method(overlap)
    max_fe = 100000

    print('testing max-degree vst')
    vstd_cache = load_or_init_vstd()
    vst = vstd_cache[overlap - 1]

    group_list = [func.p[variables] for variables in vst]
    run_cc("vstd", overlap, func, group_list, seed, max_fe)

    # print('testing hoga')
    # vst_cache = load_or_init_vst()
    # vst = vst_cache[overlap - 1]
    # print(format_json(vst))
    # run_hoga(overlap, 0, func, vst, max_fe)

    print('testing cc rdg3')
    rdg3_groups = load_or_init_rdg3()
    group_list = rdg3_groups[func_id - 1]
    run_cc("rdg3", overlap, func, group_list, seed, max_fe)

    # print('testing global')
    # run_cc(func, np.asarray([range(func.D)]), seed, max_fe)

    print('testing cc vst')
    vst_cache = load_or_init_vst()
    vst = vst_cache[overlap - 1]
    vst = compress(vst)
    group_list = [func.p[variables] for variables in vst]
    run_cc("vst", overlap, func, group_list, seed, max_fe)


def check_new_vst(func_id, seed):
    conflicting = func_id > 10
    overlap = func_id - 10 if conflicting else func_id
    fun_method = get_f14 if conflicting else get_f13

    func = fun_method(overlap)
    max_fe = 3e6 - func.D * (func.D - 1) // 2

    print('testing cc vst')
    vst_cache = load_or_init_vst()
    vst = vst_cache[overlap - 1]
    vst = compress(vst)
    group_list = [func.p[variables] for variables in vst]
    run_cc("full_vst", overlap, func, group_list, seed, max_fe)
def main():
    # 生成所有任务
    tasks = []
    vst_cache = load_or_init_vst()
    for func_method in [get_f14, get_f13]:
        for seed in range(40):
            for overlap in range(1, 11):
                func = func_method(overlap)
                max_fe = 3e6 - func.D * (func.D - 1) // 2
                vst = vst_cache[overlap - 1]
                vst = compress(vst)
                group_list = [func.p[variables] for variables in vst]
                tasks.append(delayed(run_cc)(
                    "full_vst", overlap, func, group_list, seed, max_fe))

    total_cores = os.cpu_count()
    safe_jobs = max(1, total_cores - 2)

    with  Parallel(n_jobs=safe_jobs, verbose=10, max_nbytes=0) as parallel:
        print(f"开始并行执行 {len(tasks)} 个任务")
        parallel(tasks)


if __name__ == '__main__':
    main()
