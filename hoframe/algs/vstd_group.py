import networkx as nx

from algs.vst_cec import compress


def degree_based_cut(G):
    nodes_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
    nodes_sorted = [node for node, deg in nodes_sorted]

    children = []
    cut = []
    H = G.copy()
    for node in nodes_sorted:
        H.remove_node(node)
        cut.append(node)

        if len(H) == 0:
            return cut, []

        if not nx.is_connected(H):
            components = list(nx.connected_components(H))
            subgraphs = [H.subgraph(c).copy() for c in components]
            for subgraph in subgraphs:
                children.append(degree_based_cut(subgraph))

            return cut, children
    else:
        return cut, []


def vstd_group(m):
    G = nx.from_numpy_array(m)
    vst = degree_based_cut(G)
    return compress(vst)

