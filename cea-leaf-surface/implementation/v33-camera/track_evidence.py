"""Training-only cross-view track check. One physical feature can appear once per view."""
from collections import defaultdict
import numpy as np

def classify_tracks(pairs):
    adjacency = defaultdict(set)
    edges = defaultdict(list)
    for pair_index, pair in enumerate(pairs):
        views = pair["pair"]
        for edge_index in np.flatnonzero(~pair["protected"]):
            first, second = pair["features"][edge_index]
            u, v = (views[0], int(first)), (views[1], int(second))
            adjacency[u].add(v)
            adjacency[v].add(u)
            edges[u].append((pair_index, int(edge_index)))
    flags = [np.zeros(len(pair["features"]), dtype=np.int8) for pair in pairs]
    visited = set()
    for start in adjacency:
        if start in visited:
            continue
        pending = [start]
        component = set()
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(adjacency[node] - component)
        visited.update(component)
        edge_ids = [idx for node in component for idx in edges[node]]
        distinct_views = {node[0] for node in component}
        if len(distinct_views) != len(component):
            flag = -1
        elif len(component) == 3 and len(edge_ids) == 3:
            flag = 1
        else:
            flag = 0
        for pair_index, edge_index in edge_ids:
            flags[pair_index][edge_index] = flag
    return flags

def tiered_training(core, indices, obs, labels, flags):
    """Same label budgets and spatial ordering, with closed tracks first within each pool."""
    indices = np.asarray(indices, dtype=int)
    indices = indices[flags[indices] >= 0]
    def pool(label):
        members = indices[labels[indices] == label]
        result = []
        for tier in (1, 0):
            subset = members[flags[members] == tier]
            result.extend(core.spatial_select(subset, obs, len(subset)))
        return result
    positive_pools = [pool(label) for label in sorted(set(labels[indices]) - {0})]
    positive = []
    for offset in range(max((len(p) for p in positive_pools), default=0)):
        positive.extend(p[offset] for p in positive_pools if offset < len(p))
    unassigned = pool(0)
    selected = positive[:96] + unassigned[:32]
    selected.extend(positive[96:96+128-len(selected)])
    selected.extend(unassigned[32:32+128-len(selected)])
    return np.asarray(selected, dtype=int)
