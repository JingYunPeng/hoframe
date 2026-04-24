from itertools import combinations

import numpy as np
from collections import defaultdict


def cbcco_group(adj_matrix, zeta=0.3, remove_overlap=True):
    n = len(adj_matrix)
    variables = set(range(n))
    G = defaultdict(set)
    for i, j in combinations(variables, 2):
        if adj_matrix[i, j] == 1:
            G[i].add(j)
            G[j].add(i)

    omega_bar = variables.copy()
    GS, OS = [], []
    while omega_bar:
        remain_list = list(omega_bar)
        degs = [len(G[v]) for v in remain_list]
        x_bar = remain_list[np.argmin(degs)]

        if x_bar is None:
            break

        current_group = set()
        current_group.add(x_bar)
        interacting_vars = G[x_bar]
        current_group.update(interacting_vars)

        omega_bar -= current_group

        GS.append(current_group)

    i = 1
    while i < len(GS):
        for j in range(i):
            overlap = GS[i] & GS[j]
            if not overlap:
                continue
            ratio_i = len(overlap) / len(GS[i])
            ratio_j = len(overlap) / len(GS[j])

            if max(ratio_i, ratio_j) < zeta:
                OS.append((i, j, overlap))
                continue
            GS[i] = GS[i] | GS[j]
            del GS[j]
            break
        else:
            i += 1
            continue  # while i < len(GS)

    if remove_overlap:
        for i, j, overlap in OS:
            GS[i] -= overlap
            GS[j] -= overlap

    return GS, OS


def occ_group(adj_matrix):
    GS, OS = cbcco_group(adj_matrix, zeta=1, remove_overlap=False)
    return [list(group) for group in GS]


if __name__ == '__main__':
    dims = np.array([50, 50, 25, 25, 100, 100, 25, 25, 50, 25, 100, 25, 100, 50, 25, 25, 25, 100, 50, 25])
    overlap = 5
    D = 1000 - 19 * overlap
    components = []
    A = np.zeros((D, D))

    cur = 0
    for i, d in enumerate(dims):
        components.append(range(cur, cur + d))
        cur += d - overlap

    print(components)

    for comp in components:
        for i, j in combinations(comp, 2):
            A[i][j] = A[j][i] = 1

    GS = occ_group(A)
    for g in GS:
        print(len(g), min(g), max(g))
