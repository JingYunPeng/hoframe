from itertools import combinations

import numpy as np
from numpy.matlib import zeros


def interact(adj_matrix, sub1, sub2, xremain):
    new_group = set(sub1)
    for j in sub2:
        for i in sub1:
            if adj_matrix[i, j] == 1:
                new_group.add(j)
                break
    xremain_new = list(set(xremain) - new_group)
    return list(new_group), xremain_new


def rdg3(adj_matrix, dim, tn=50):
    seps = []
    nongroups = []

    xremain = list(range(dim))

    if len(xremain) == 0:
        return seps, nongroups

    sub1 = [xremain[0]]
    sub2 = xremain[1:]

    while len(xremain) > 0:
        sub1_a, xremain_new = interact(adj_matrix, sub1, sub2, xremain)

        if len(sub1_a) != len(sub1) and len(sub1_a) < tn:
            sub1 = sub1_a
            sub2 = xremain_new
            xremain = xremain_new

            if len(xremain) == 0:
                nongroups.append(sub1)
                break
        else:
            if len(sub1_a) == 1:
                seps.append(sub1_a[0])
            else:
                nongroups.append(sub1_a)

            xremain = xremain_new

            if len(xremain) > 1:
                sub1 = [xremain[0]]
                sub2 = xremain[1:]
            elif len(xremain) == 1:
                seps.append(xremain[0])
                break

    seps = [seps[i:i + 100] for i in range(0, len(seps), 100)]

    return seps, nongroups


def rdg3_group(m):
    seps, nongroups = rdg3(m, len(m))
    return seps + nongroups


if __name__ == "__main__":
    # dims = np.array([50, 50, 25, 25, 100, 100, 25, 25, 50, 25, 100, 25, 100, 50, 25, 25, 25, 100, 50, 25])
    # overlap = 5
    # components = []
    # D = 1000 - 19 * overlap
    # m = np.zeros((D, D))
    #
    # cur = 0
    # for i, d in enumerate(dims):
    #     components.append(range(cur, cur + d))
    #     cur += d - overlap
    #
    # print(components)
    #
    # for comp in components:
    #     for i, j in combinations(comp, 2):
    #         m[i][j] = 1
    #         m[j][i] = 1
    D = 234
    m = zeros((D, D))
    seps, nonseps = rdg3(m, D)

    print("Separable groups:", seps)
    print("Nonseparable groups:", nonseps)
    print([len(sep) for sep in seps])
    print(len(nonseps))
