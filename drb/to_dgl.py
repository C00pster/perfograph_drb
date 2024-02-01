import json
import dgl
from os import listdir
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import IntTensor
import re
import numpy as np
import logging
from torch_geometric.data import HeteroData
from torch_geometric.data import Data

from torch_geometric.loader import DataLoader
import re
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from torch.nn import LayerNorm

def is_numeric(text):
    if re.match(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", text):
        return True
    return False

def get_digit_and_pos(text):
    digits_array = []
    digits_pos_array = []
    if is_numeric(text):
        point_position = text.find(".")
        if point_position > -1:
            for i in range(0, point_position):
                digits_array.append(text[i])
            for i in reversed(range(point_position)):
                digits_pos_array.append(str(i))
            j = -1
            for i in range(point_position + 1, len(text)):
                digits_array.append(text[i])
                digits_pos_array.append(str(j))
                j -= 1
        else:
            for i in range(0, len(text)):
                digits_array.append(text[i])
            for i in reversed(range(len(text))):
                digits_pos_array.append(str(i))
    return digits_array, digits_pos_array

def get_digit_emb_of_number(token, feature_map):
    digits = []
    digits_pos = []
    digit_embedding_vector = []
    digit_pos_vector = []
    reduced_final_embedding = []
    if is_numeric(token):
        digits, digits_pos = get_digit_and_pos(token)
        for digit in digits:
            if digit in feature_map:
                lookup_tensor = torch.tensor([feature_map[digit]], dtype=torch.long)
                node_embed = embeds(lookup_tensor)
                node_embed_real_numpy = node_embed.detach().numpy()
                node_embed_list = []
                for value in node_embed_real_numpy:
                    node_embed_list = value.tolist()
                digit_embedding_vector.append(node_embed_list)
            else:
                digit_embedding_vector.append([0.0, 0.0, 0.0])
        for digit_pos in digits_pos:
            if digit_pos in feature_map:
                lookup_tensor = torch.tensor([feature_map[digit_pos]], dtype=torch.long)
                node_embed = embeds(lookup_tensor)
                node_embed_real_numpy = node_embed.detach().numpy()
                node_embed_list = []
                for value in node_embed_real_numpy:
                    node_embed_list = value.tolist()
                digit_pos_vector.append(node_embed_list)
            else:
                digit_pos_vector.append([0.0, 0.0, 0.0])

        final_embedding_vector = []
        final_embedding_list_of_np_arrays = list((np.array(digit_embedding_vector) + np.array(digit_pos_vector)))

        for embedding in final_embedding_list_of_np_arrays:
            final_embedding_vector.append(list(embedding))

        final_embedding_vector_np_array = np.array(final_embedding_vector)
        final_embedding_vector_np_array_sum = np.sum(final_embedding_vector_np_array, axis=0)
        reduced_final_embedding = list(final_embedding_vector_np_array_sum)

        max_of_reduced_final_embedding = max(reduced_final_embedding, key=abs)

        for i3 in range(len(reduced_final_embedding)):
            reduced_final_embedding[i3] = reduced_final_embedding[i3] / (abs(max_of_reduced_final_embedding) + 1)
    else:
        reduced_final_embedding = [0.0, 0.0, 0.0]
    return reduced_final_embedding

feat_count = 0
feature_file = open("feature_map_file_pg_plus_text_all_dev_map_with_nvidia.txt", 'r')
feature_lines = feature_file.readlines()
feature_map = {}
for feature_line in feature_lines:
    feature_key = feature_line.split(",")[0]
    if feature_key not in feature_map:
        feature_map[feature_key] = feat_count
        feat_count += 1
embeds = nn.Embedding(feat_count, 3) # feat_count words in vocab, 3 dimensional embeddings

graphs = []
with open("perfograph/DRB001-antidep1-orig-yes.json", "r") as f:
    graphs.append(json.load(f))

data_list = []

count = 0

for graph in graphs:
    data = HeteroData()

    nodes = graph[0]
    edges = graph[1]
    edge_positions = graph[2]

    # get instruction node embeddings
    instruction_node_counter = 0
    instruction_node_file_strings = ["graph_id,node_id,feat\n"]
    for instruction_node in nodes["instruction"]:
        full_text_of_instruction_node = str(instruction_node[0])
        instruction_node_str = ""
        for i in range(2, len(full_text_of_instruction_node) - 2):
            instruction_node_str += full_text_of_instruction_node[i]
        tokens = instruction_node_str.split(" ")
        instruction_node_embed_str = ""
        prev_token_length = len(tokens)
        if len(tokens) < 40:
            for i in range(40 - prev_token_length):
                tokens.append("QQ")
        for i in range(len(tokens)):
            # digit embedding
            digits = []
            digits_pos = []
            digit_embedding_vector = []
            digit_pos_vector = []
            if is_numeric(tokens[i]):
                reduced_final_embedding = get_digit_emb_of_number(tokens[i], feature_map)
                if i == 0:
                    instruction_node_embed_str += ("\"" + str(reduced_final_embedding[0]) + ',' +
                                        str(reduced_final_embedding[1]) + ',' +
                                        str(reduced_final_embedding[2]) + ',')
                elif i == len(tokens) - 1:
                    instruction_node_embed_str += (str(reduced_final_embedding[0]) + ',' +
                                        str(reduced_final_embedding[1]) + ',' +
                                        str(reduced_final_embedding[2]) + "\"")
                else:
                    instruction_node_embed_str += (str(reduced_final_embedding[0]) + ',' +
                                        str(reduced_final_embedding[1]) + ',' +
                                        str(reduced_final_embedding[2]) + ',')
            elif tokens[i] in feature_map:
                lookup_tensor = torch.tensor([feature_map[tokens[i]]], dtype=torch.long)
                instruction_node_embed = embeds(lookup_tensor)
                instruction_node_embed_real_numpy = instruction_node_embed.detach().numpy()
                instruction_node_embed_list = []
                for value in instruction_node_embed_real_numpy[0]:
                    instruction_node_embed_list.append(str(value))
                if i == 0:
                    instruction_node_embed_str += ("\"" + instruction_node_embed_list[0] + ',' +
                                        instruction_node_embed_list[1] + ',' +
                                        instruction_node_embed_list[2] + ',')
                elif i == len(tokens) - 1:
                    instruction_node_embed_str += (instruction_node_embed_list[0] + ',' +
                                        instruction_node_embed_list[1] + ',' +
                                        instruction_node_embed_list[2] + "\"")
                else:
                    instruction_node_embed_str += (instruction_node_embed_list[0] + ',' +
                                        instruction_node_embed_list[1] + ',' +
                                        instruction_node_embed_list[2] + ',')
            else:
                if i == 0:
                    instruction_node_embed_str += ("\"0.0,0.0,0.0,")
                elif i == len(tokens) - 1:
                    instruction_node_embed_str += ("0.0,0.0,0.0\"")
                else:
                    instruction_node_embed_str += ("0.0,0.0,0.0,")
        instruction_node_file_string = str(count) + "," + str(instruction_node_counter) + "," + instruction_node_embed_str + "\n"
        instruction_node_file_strings.append(instruction_node_file_string)
        instruction_node_counter += 1

    with open("dgl/nodes_0.csv", "w") as f:
        f.writelines(instruction_node_file_strings)

    # get embeddings variable nodes
    variable_node_counter = 0
    variable_node_file_strings = ["graph_id,node_id,feat\n"]
    for variable_node in nodes["variable"]:
        text_of_variable_node = str(variable_node[0])
        digits = []
        digits_pos = []
        digit_embedding_vector = []
        digit_pos_vector = []
        variable_node_embed_str = ""
        if is_numeric(text_of_variable_node) == True:
            reduced_final_embedding = get_digit_emb_of_number(text_of_variable_node, feature_map)
            variable_node_embed_str += ("\"" + str(reduced_final_embedding[0]) + ',' +
                                str(reduced_final_embedding[1]) + ',' +
                                str(reduced_final_embedding[2]) + ',') # + "\""
        
            for pad in range(0,108):
                variable_node_embed_str += "0.0,"
            variable_node_embed_str += "0.0\""
        
        elif text_of_variable_node in feature_map:
            lookup_tensor = torch.tensor([feature_map[text_of_variable_node]], dtype=torch.long)
            variable_node_embed = embeds(lookup_tensor)
            variable_node_embed_real_numpy = variable_node_embed.detach().numpy()
            variable_node_embed_list = []
            for value in variable_node_embed_real_numpy[0]:
                variable_node_embed_list.append(str(value))
            variable_node_embed_str += ("\"" + variable_node_embed_list[0] + ',' +
                                variable_node_embed_list[1] + ',' +
                                variable_node_embed_list[2] + ',')
        
            for pad in range(0,108):
                variable_node_embed_str += "0.0,"
            variable_node_embed_str += "0.0\""
        
        else:
            variable_node_embed_str += ("\"0.0,0.0,0.0,")
        
            for pad in range(0,108):
                variable_node_embed_str += "0.0,"
            variable_node_embed_str += "0.0\""
        
        variable_node_file_string = str(count) + "," + str(variable_node_counter) + "," + variable_node_embed_str + "\n"
        variable_node_file_strings.append(variable_node_file_string)
        variable_node_counter += 1

    with open("dgl/nodes_1.csv", "w") as f:
        f.writelines(variable_node_file_strings)

    # get embeddings for constant nodes
    constant_node_counter = 0
    constant_node_file_strings = ["graph_id,node_id,feat\n"]
    for constant_node in nodes["constant"]:
        constant_node_embed_str = ""
        text_embed_list = []
        text_type_of_constant_node = str(constant_node[0])
        if text_type_of_constant_node in feature_map:
            lookup_tensor = torch.tensor([feature_map[text_type_of_constant_node]], dtype=torch.long)
            text_embed = embeds(lookup_tensor)
            text_embed_real_numpy = text_embed.detach().numpy()
            for value in text_embed_real_numpy[0]:
                text_embed_list.append(str(value))
        else :
            text_embed_list = ["0.0", "0.0", "0.0"]
        text_value_of_constant_node = str(constant_node[1])
        digit_emb_vec_of_text_value = get_digit_emb_of_number(text_value_of_constant_node, feature_map)
        for component in digit_emb_vec_of_text_value:
            text_embed_list.append(str(component))
        for i in range(0, 6-len(text_embed_list)):
            text_embed_list.append("0.0")

        constant_node_embed_str += ("\"" + text_embed_list[0] + ',' +
                            text_embed_list[1] + ',' +
                            text_embed_list[2] + ',' +
                            text_embed_list[3] + ',' +
                            text_embed_list[4] + ',' +
                            text_embed_list[5] + ",")
        
        for pad in range(0, 113):
            constant_node_embed_str += "0,0,"
        constant_node_embed_str += "0,0\""

        constant_node_file_string = str(count) + "," + str(constant_node_counter) + "," + constant_node_embed_str + "\n"
        constant_node_file_strings.append(constant_node_file_string)
        constant_node_counter += 1

    with open("dgl/nodes_2.csv", "w") as f:
        f.writelines(constant_node_file_strings)

    # get embeddings for varray
    # TODO
        
    # get embeddings for vvector
    # TODO
        
    # get embeddings for carray
    # TODO
    
    # get embeddings for cvector
    # TODO

    # get node embeddings
    for node_type in nodes.keys():
        feature_vector = []
        node_feature_vector = []
        node_counter = 0
        for node in nodes[node_type]:
            text_of_node = str(nodes[node_type][0])

            if text_of_node in feature_map:
                lookup_tensor = torch.tensor([feature_map[text_of_node]], dtype=torch.long)
                text_embed = embeds(lookup_tensor)
                text_embed_real_numpy = text_embed.detach().numpy()
                text_embed_list = []
                for value in text_embed_real_numpy:
                    text_embed_list = value.tolist()
                feature_vector.append(text_embed_list)
            else:
                feature_vector.append([0.0, 0.0, 0.0])
        data[node_type].x = torch.tensor(feature_vector)

    # get edge embeddings
    edges_lines = []
    for edge_type in edges.keys():
        edge_type_split = edge_type.split("_")
        edges_index = []
        source_node = []
        des_node = []
        edge_lines = ["graph_id,src_id,dst_id\n"]
        for edge in edges[edge_type]:
            source_node.append(edge[0])
            des_node.append(edge[1])
            edge_string = str(count) + "," + str(edge[0]) + "," + str(edge[1]) + "\n"
            edge_lines.append(edge_string)
        edges_lines.append(edge_lines)
        edges_index.append(source_node)
        edges_index.append(des_node)
        data[edge_type_split[0], edge_type_split[1], edge_type_split[2]].edge_index = edges_index
    for i in range(len(edges.keys())):
        filename = f"dgl/edges_{i}.csv"
        with open(filename, "w") as f:
            f.writelines(edges_lines[i])

    # get edge position embeddings
    for edge_type in edge_positions.keys():
        edge_type_split = edge_type.split("_")
        edge_type_positions = []
        for edge_position in edge_positions[edge_type]:
            edge_type_positions.append(edge_position)
        data[edge_type_split[0], edge_type_split[1], edge_type_split[2]].edge_pos = torch.tensor(edge_type_positions)

    if count < 395:
        data['y'] = torch.tensor([0])
    else:
        data['y'] = torch.tensor([1])
    data_list.append(data)

    count += 1

print(count)





# node_types = []
# edge_types = []
# dgl_dict = {}

# for etype in edges:
#     edge_tuple = tuple(etype.split("_"))
#     dgl_dict[edge_tuple] = []
#     for edge in edges[etype]:
#         dgl_dict[edge_tuple].append(torch.tensor(edge))        

# dgl_graph = dgl.heterograph(dgl_dict)

# enum_array = []
# for node in nodes["instruction"]:
#     enum_array.append(llvm_instruction_map[node[0]])
# dgl_graph.nodes["instruction"].data['instruction_mapping'] = torch.tensor(enum_array)

# # Need a way to tokenize information about the nodes

# node_data = dgl_graph.ndata
# node_df = pd.DataFrame({key: node_data[key] for key in node_data.keys()})

# print(node_df)