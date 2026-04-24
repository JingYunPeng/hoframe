from itertools import combinations

import networkx as nx

def vst_group(m):
    G = nx.from_numpy_array(m)
    cliques = []
    for cliq in nx.find_cliques(G):
        cliques.append(set(cliq))

    nodes = []
    U = set()
    for i, j in combinations(cliques, 2):
        a = i & j
        if len(a) != 0 and a not in nodes:
            nodes.append(a)
            U |= a
    for cliq in cliques:
        b = cliq - U
        if len(b) != 0 and b not in nodes:
            nodes.append(b)

    new_G = nx.Graph()
    weights = [len(node) for node in nodes]

    for cliq in cliques:
        node_idx_list = []
        for idx, node in enumerate(nodes):
            if node.issubset(cliq):
                node_idx_list.append(idx)

        for i, j in combinations(node_idx_list, 2):
            new_G.add_edge(i, j)


    tree, cost = compute_optimal_vst(new_G, weights)

    post_order_arrays = []

    def post_order(tree_node):

        for child in tree_node.children:
            post_order(child)

        if tree_node.is_leaf():
            post_order_arrays.extend([list(nodes[i]) for i in tree_node.leaf_nodes])
        else:
            post_order_arrays.extend([list(nodes[i]) for i in tree_node.separator])

    post_order(tree)

    final_result = []
    buffer = []

    for arr in post_order_arrays:
        current_len = len(arr)
        if current_len >= 50:
            if buffer:
                final_result.append(buffer)
                buffer = []
            final_result.append(arr)
        else:
            buffer.extend(arr)
            if len(buffer) >= 50:
                final_result.append(buffer)
                buffer = []
    if buffer:
        final_result.append(buffer)
    return final_result


def is_biconnected(G):
    return nx.is_biconnected(G)

def find_vertex_separators(G):
    nodes = list(G.nodes())
    separators = []

    for r in range(1, len(nodes)):
        for subset in combinations(nodes, r):
            H = G.copy()
            H.remove_nodes_from(subset)

            if not nx.is_connected(H):
                separators.append(set(subset))

    return separators


def split_graph(G, S):
    H = G.copy()
    H.remove_nodes_from(S)
    components = []

    for comp in nx.connected_components(H):
        components.append(G.subgraph(comp).copy())

    return components

class VSTNode:
    def __init__(self, separator=None, children=None, leaf_nodes=None):
        self.separator = separator      # set
        self.children = children or []  # list of VSTNode
        self.leaf_nodes = leaf_nodes    # set

    def is_leaf(self):
        return self.separator is None

def build_vst(G, weights, memo):
    key = tuple(sorted(G.nodes()))
    if key in memo:
        return memo[key]

    if len(G.nodes()) <= 1 or is_biconnected(G):
        cost = sum(weights[v] for v in G.nodes())
        node = VSTNode(leaf_nodes=set(G.nodes()))
        memo[key] = (node, cost)
        return memo[key]

    best_cost = float('inf')
    best_tree = None

    separators = find_vertex_separators(G)

    for S in separators:
        subgraphs = split_graph(G, S)

        subtrees = []
        max_sub_cost = 0
        valid = True

        for sub in subgraphs:
            subtree, cost = build_vst(sub, weights, memo)
            subtrees.append(subtree)
            max_sub_cost = max(max_sub_cost, cost)


            if max_sub_cost >= best_cost:
                valid = False
                break

        if not valid:
            continue

        total_cost = sum(weights[v] for v in S) + max_sub_cost

        if total_cost < best_cost:
            best_cost = total_cost
            best_tree = VSTNode(separator=set(S), children=subtrees)

    memo[key] = (best_tree, best_cost)
    return memo[key]


def compute_optimal_vst(G, weights):
    memo = {}
    tree, cost = build_vst(G, weights, memo)
    return tree, cost


def print_vst(node, depth=0):
    indent = "  " * depth
    if node.is_leaf():
        print(f"{indent}Leaf: {node.leaf_nodes}")
    else:
        print(f"{indent}Separator: {node.separator}")
        for child in node.children:
            print_vst(child, depth + 1)
