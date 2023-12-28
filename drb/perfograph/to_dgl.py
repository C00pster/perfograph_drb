import json
import dgl
import torch

with open('DRB001-antidep1-orig-yes.json') as f:
    G = json.load(f)

    # breakpoint()

edge_data = {}

for edge in G[1]:
    parts = edge.split("_")
    from_node = parts[0]
    edge_type = parts[1]
    to_node = parts[2]
    
    edges = [torch.tensor(edge) for edge in G[1][edge]]
    edge_data[(from_node, edge_type, to_node)] = edges

g = dgl.heterograph(edge_data)

print(g)

        