from joblib import Parallel

from test_all_graph import (
    build_tasks,
)

MAX_FE = 100000
TEST_GENERATIONS = 5

tasks = []

# vstd
tasks += build_tasks(
    algorithms=["vstd"],
    func_ids=[8],
    seeds=[19],
    max_fe=MAX_FE,
    test_generations=TEST_GENERATIONS,
)

# CBCCO
tasks += build_tasks(
    algorithms=["CBCCO"],
    func_ids=[1, 6],
    seeds=[19],
    max_fe=MAX_FE,
    test_generations=TEST_GENERATIONS,
)

tasks += build_tasks(
    algorithms=["CBCCO"],
    func_ids=[8],
    seeds=list(range(20)),
    max_fe=MAX_FE,
    test_generations=TEST_GENERATIONS,
)

tasks += build_tasks(
    algorithms=["CBCCO"],
    func_ids=[9],
    seeds=list(range(11, 20)),
    max_fe=MAX_FE,
    test_generations=TEST_GENERATIONS,
)

with Parallel(n_jobs=30, verbose=10, max_nbytes=0) as parallel:
    parallel(tasks)
