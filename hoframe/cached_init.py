import json
import os
from functools import lru_cache

import numpy as np

from algs.rdg3_cec import load_groups
from algs.vst_cec import vst_list
from cec import F13, F14, BenchmarkSuite
from randg import get_graph_data, GraphFunction


@lru_cache
def load_or_init_vst():
    file_path = os.path.join('datafiles', 'vst.json')
    try:
        # 尝试读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            vst_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        vst_data = vst_list()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vst_data, f, ensure_ascii=False)
    return vst_data

@lru_cache
def load_or_init_graph():
    file_path = os.path.join('datafiles', 'graph.json')
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        graph_data = get_graph_data()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False)
    return graph_data

@lru_cache
def load_or_init_rdg3():
    folder = os.path.join('datafiles', 'rdg3')
    rdg3_groups = load_groups(folder)
    return rdg3_groups


@lru_cache
def get_graph_function(i):
    data = load_or_init_graph()
    datum = data[i]
    return GraphFunction(datum)


@lru_cache
def suite():
    return BenchmarkSuite('datafiles/cec')


@lru_cache
def get_f13(overlap):
    return F13(suite(), overlap)


@lru_cache
def get_f14(overlap):
    return F14(suite(), overlap)


### tests

def test_graph():
    data = load_or_init_graph()
    funcid = 3
    datum = data[funcid]
    ws = [_[2] for _ in datum['coes']]
    sum_ws = sum(ws)

    x = np.ones(1000)
    f3 = get_graph_function(3)

    print(f3.compute(x), sum_ws)

def test_rdg3():
    data = load_or_init_rdg3()
    print(data)

if __name__ == '__main__':
    test_rdg3()
