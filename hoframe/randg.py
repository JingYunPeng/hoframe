from itertools import combinations, product

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


def a_f1():
    # random linear
    np.random.seed(1)
    p = np.random.permutation(10)
    a = np.zeros((10, 10))

    for i in range(9):
        a[p[i], p[i + 1]] = 1
    return a


def a_f2():
    # circle
    np.random.seed(2)
    p = np.random.permutation(10)
    a = np.zeros((10, 10))

    for i in range(9):
        a[p[i], p[i + 1]] = 1
    a[p[0], p[9]] = 1
    return a


def a_f3():
    # star 1
    np.random.seed(3)
    center = np.random.randint(0, 9)
    a = np.zeros((10, 10))
    for i in range(10):
        if i < center:
            a[i, center] = 1
        elif i > center:
            a[center, i] = 1
    return a


def a_f4():
    # star 2
    np.random.seed(4)
    p = np.random.permutation(10)
    a = np.zeros((10, 10))
    c1, c2 = p[0], p[1]
    a[min(c1, c2), max(c1, c2)] = 1
    for i in [2, 3, 4]:
        a[min(c1, p[i]), max(c1, p[i])] = 1
    for i in [5, 6, 7, 8, 9]:
        a[min(c2, p[i]), max(c2, p[i])] = 1
    return a


def a_f567(seed):
    rng = np.random.default_rng(seed)

    while True:
        g = nx.random_tree(10, seed=rng)
        degrees = [d for _, d in g.degree()]
        if min(degrees) == 1 and max(degrees) == 4:
            full_matrix = nx.to_numpy_array(g)
            return np.triu(full_matrix)


def a_f890(seed):
    rng = np.random.default_rng(seed)

    while True:
        g = nx.fast_gnp_random_graph(10, 0.25, seed=rng)
        if not nx.is_connected(g):
            continue
        degrees = [d for _, d in g.degree()]
        if min(degrees) != 1 or max(degrees) > 3:
            continue
        if any(nx.triangles(g).values()):
            continue
        try:
            nx.find_cycle(g)
            full_matrix = nx.to_numpy_array(g)
            return np.triu(full_matrix)
        except:
            continue


def get_b(a):
    dims = [200, 200, 200, 100, 100, 50, 50, 50, 25, 25]
    components = []
    cur_sum = 0
    for dim in dims:
        components.append(list(range(cur_sum, cur_sum + dim)))
        cur_sum += dim
    b = np.zeros((cur_sum, cur_sum))

    for comp in components:
        for k, t in combinations(comp, 2):
            if k != t:
                b[k, t] = 1

    for i, j in combinations(range(10), 2):
        if a[i, j] == 1 or a[j, i] == 1:
            for k, t in product(components[i], components[j]):
                b[k, t] = b[t, k] = 1
    return b


def get_graph_data():
    np.random.seed(0)
    all_a = [
        a_f1(),
        a_f2(),
        a_f3(),
        a_f4(),
        a_f567(5),
        a_f567(6),
        a_f567(7),
        a_f890(8),
        a_f890(9),
        a_f890(10),
    ]
    graph_data = []
    for a in all_a:
        coes = []
        p = [int(_) for _ in np.random.permutation(1000)]
        datum = {'coes': coes, 'p': p}
        b = get_b(a)
        for i, j in combinations(range(1000), 2):
            if b[i][j] == 1:
                w = 0
                while abs(w) < 0.01:
                    w = np.random.rand()
                alpha = int(np.random.choice([-2, -1, 1, 2]))
                beta = int(np.random.choice([-2, -1, 1, 2]))
                coes.append((i, j, w, alpha, beta))
        graph_data.append(datum)
    return graph_data


def adjacent_matrix_from_datum(datum):
    D = 1000
    m = np.zeros((D, D))
    for i, j, w, a, b in datum['coes']:
        m[i, j] = 1
        m[j, i] = 1
    return m


def pow(a, n):
    if n < 0:
        return np.sign(a) * np.power(abs(a) + 1, n)
    return np.power(a, n)


class GraphFunction:
    def __init__(self, datum):
        self.coes = []
        self.p = np.asarray(datum['p'])
        self.D = 1000

        self.M = np.zeros((self.D, self.D))
        for i, j, w, a, b in datum['coes']:
            self.M[self.p[i], self.p[j]] = w
            # 忽略其他参数以提高速度

    def compute(self, x):
        return x @ self.M @ x


def show(a, i):
    print(a,i)
    a = np.array(a)
    n = a.shape[0]
    m = a + a.T
    G = nx.from_numpy_array(m, create_using=nx.Graph)
    pos = nx.spring_layout(G, seed=42, k=0.9, iterations=50)

    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=600, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # plt.title(f"Graph for F{i}")
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout()
    plt.text(0.5, -0.05, f"Graph for F{i}", ha='center', transform=plt.gca().transAxes)
    # plt.show()
    plt.savefig(f"figs/f{i}.pdf", bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    all_a = [
        a_f1(),
        a_f2(),
        a_f3(),
        a_f4(),
        a_f567(5),
        a_f567(6),
        a_f567(7),
        a_f890(8),
        a_f890(9),
        a_f890(10),
    ]
    for i, a in enumerate(all_a):
        show(a, i + 1)
