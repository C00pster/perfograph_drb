import json
import dgl
import torch


with open("perfograph/DRB001-antidep1-orig-yes.json", "r") as f:
    data = json.load(f)
with open("llvm_ir_instruction_mapping.json", "r") as f:
    llvm_mapper = json.load(f)

nodes = data[0]
edges = data[1]
edge_pos = data[2]

node_types = []
edge_types = []
dgl_dict = {}

for etype in edges:
    edge_tuple = tuple(etype.split("_"))
    dgl_dict[edge_tuple] = []
    for edge in edges[etype]:
        dgl_dict[edge_tuple].append(torch.tensor(edge))        

dgl_graph = dgl.heterograph(dgl_dict)

enum_array = []
for node in nodes["instruction"]:
    enum_array.append(llvm_mapper[node[0]])
dgl_graph.nodes["instruction"].data['instruction_mapping'] = torch.tensor(enum_array)

print(dgl_graph)