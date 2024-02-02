import json
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import HeteroData
import re

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

def get_embedding_for_token(token, feature_map, embeds):
    if is_numeric(token):
        return get_digit_emb_of_number(token, feature_map)
    elif token in feature_map:
        lookup_tensor = torch.tensor([feature_map[token]], dtype=torch.long)
        return embeds(lookup_tensor).detach().numpy().flatten().tolist()
    else:
        return [0.0, 0.0, 0.0]
    
def write_embeddings_to_csv(file_path, headers, data_strings):
    with open(file_path, "w") as f:
        f.writelines([headers] + data_strings)

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
    instruction_node_file_strings = []
    for instruction_node in nodes["instruction"]:
        instruction_node_str = "" if instruction_node[1] is None else instruction_node[1][0]
        tokens = instruction_node_str.split(" ")
        instruction_node_embed_str = ""
        prev_token_length = len(tokens)
        if len(tokens) < 40:
            for i in range(40 - prev_token_length):
                tokens.append("QQ")
        for i in range(len(tokens)):
            reduced_final_embedding = get_embedding_for_token(tokens[i], feature_map, embeds)
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
        instruction_node_file_string = str(count) + "," + str(instruction_node_counter) + "," + instruction_node_embed_str + "\n"
        instruction_node_file_strings.append(instruction_node_file_string)
        instruction_node_counter += 1

    write_embeddings_to_csv("dgl/nodes_0.csv", "graph_id,node_id,feat\n", instruction_node_file_strings)

    # get embeddings variable nodes
    variable_node_counter = 0
    variable_node_file_strings = []
    for variable_node in nodes["variable"]:
        text_of_variable_node = str(variable_node[0])
        digits = []
        digits_pos = []
        digit_embedding_vector = []
        digit_pos_vector = []
        variable_node_embed_str = ""
        reduced_final_embedding = get_embedding_for_token(text_of_variable_node, feature_map, embeds)
        variable_node_embed_str += ("\"" + str(reduced_final_embedding[0]) + ',' +
                                str(reduced_final_embedding[1]) + ',' +
                                str(reduced_final_embedding[2]) + ',')
        for pad in range(0,108):
            variable_node_embed_str += "0.0,"
        variable_node_embed_str += "0.0\""
        variable_node_file_string = str(count) + "," + str(variable_node_counter) + "," + variable_node_embed_str + "\n"
        variable_node_file_strings.append(variable_node_file_string)
        variable_node_counter += 1

    write_embeddings_to_csv("dgl/nodes_1.csv", "graph_id,node_id,feat\n", variable_node_file_strings)

    # get embeddings for constant nodes
    constant_node_counter = 0
    constant_node_file_strings = []
    for constant_node in nodes["constant"]:
        constant_node_embed_str = ""
        text_type_of_constant_node = str(constant_node[0])
        text_embed_list = get_embedding_for_token(text_type_of_constant_node, feature_map, embeds)
        text_value_of_constant_node = str(constant_node[1])
        digit_emb_vec_of_text_value = get_digit_emb_of_number(text_value_of_constant_node, feature_map)

        constant_node_embed_str += ("\"" + str(text_embed_list[0]) + ',' +
                            str(text_embed_list[1]) + ',' +
                            str(text_embed_list[2]) + ',' +
                            str(text_embed_list[3]) + ',' +
                            str(text_embed_list[4]) + ',' +
                            str(text_embed_list[5]) + ",")
        
        for pad in range(0, 113):
            constant_node_embed_str += "0,0,"
        constant_node_embed_str += "0,0\""

        constant_node_file_string = str(count) + "," + str(constant_node_counter) + "," + constant_node_embed_str + "\n"
        constant_node_file_strings.append(constant_node_file_string)
        constant_node_counter += 1

    write_embeddings_to_csv("dgl/nodes_2.csv", "graph_id,node_id,feat\n", constant_node_file_strings)

    # get embeddings for varray
    varray_node_counter = 0
    varray_node_file_strings = []
    for varray_node in nodes["varray"]:
        text_embed_list = []
        text_of_varray_node = varray_node[1]
        for token in text_of_varray_node.split(" "):
            reduced_final_embedding = get_embedding_for_token(token, feature_map, embeds)
            for value in reduced_final_embedding:
                text_embed_list.append(str(value))
        
        for pad in range(0, 120 - len(text_embed_list)):
            text_embed_list.append("0.0")

        varray_node_embed_str = ""
        for component in text_embed_list:
            varray_node_embed_str += str(component) + ","
        varray_node_embed_str += text_embed_list[len(text_embed_list) - 1]

        varray_node_file_string = str(count) + "," + str(varray_node_counter) + "," + "\"" + varray_node_embed_str + "\"\n"
        varray_node_file_strings.append(varray_node_file_string)

        varray_node_counter += 1

    write_embeddings_to_csv("dgl/nodes_3.csv", "graph_id,node_id,feat\n", varray_node_file_strings)
        
    # get embeddings for vvector
    vvector_node_counter = 0
    vvector_node_file_strings = ["graph_id,node_id,feat\n"]
    for vvector_node in nodes["vvector"]:
        text_of_vvector_node = str(vvector_node[0])
        text_embed_list = []
        for token in text_of_vvector_node.split(" "):
            reduced_final_embedding = get_embedding_for_token(token, feature_map, embeds)
            for value in reduced_final_embedding:
                text_embed_list.append(str(value))
    
        for pad in range(0, 120 - len(text_embed_list)):
            text_embed_list.append("0.0")

        vvector_node_embed_str = ""
        for component in text_embed_list:
            vvector_node_embed_str += str(component) + ","
        vvector_node_embed_str += text_embed_list[len(text_embed_list) - 1]

        vvector_node_file_string = str(count) + "," + str(vvector_node_counter) + "," + "\"" + vvector_node_embed_str + "\"\n"
        vvector_node_file_strings.append(vvector_node_file_string)

        vvector_node_counter += 1

    write_embeddings_to_csv("dgl/nodes_4.csv", "graph_id,node_id,feat\n", vvector_node_file_strings)
        
    # get embeddings for carray
    carray_node_counter = 0
    carray_node_file_strings = ["graph_id,node_id,feat\n"]
    for carray_node in nodes["carray"]:
        text_of_carray_node = str(carray_node[0])
        text_embed_list = []
        for token in text_of_carray_node.split(" "):
            reduced_final_embedding = get_embedding_for_token(token, feature_map, embeds)
            for value in reduced_final_embedding:
                text_embed_list.append(str(value))
    
        for pad in range(0, 120 - len(text_embed_list)):
            text_embed_list.append("0.0")

        carray_node_embed_str = ""
        for component in text_embed_list:
            carray_node_embed_str += str(component) + ","
        carray_node_embed_str += text_embed_list[len(text_embed_list) - 1]

        carray_node_file_string = str(count) + "," + str(carray_node_counter) + "," + "\"" + carray_node_embed_str + "\"\n"
        carray_node_file_strings.append(carray_node_file_string)

        carray_node_counter += 1

    write_embeddings_to_csv("dgl/nodes_5.csv", "graph_id,node_id,feat\n", carray_node_file_strings)
    
    # get embeddings for cvector
    cvector_node_counter = 0
    cvector_node_file_strings = ["graph_id,node_id,feat\n"]
    for cvector_node in nodes["cvector"]:
        text_of_cvector_node = str(cvector_node[0])
        text_embed_list = []
        for token in text_of_cvector_node.split(" "):
            reduced_final_embedding = get_embedding_for_token(token, feature_map, embeds)
            for value in reduced_final_embedding:
                text_embed_list.append(str(value))
    
        for pad in range(0, 120 - len(text_embed_list)):
            text_embed_list.append("0.0")

        cvector_node_embed_str = ""
        for component in text_embed_list:
            cvector_node_embed_str += str(component) + ","
        cvector_node_embed_str += text_embed_list[len(text_embed_list) - 1]

        cvector_node_file_string = str(count) + "," + str(cvector_node_counter) + "," + "\"" + cvector_node_embed_str + "\"\n"
        cvector_node_file_strings.append(cvector_node_file_string)

        cvector_node_counter += 1

    write_embeddings_to_csv("dgl/nodes_6.csv", "graph_id,node_id,feat\n", cvector_node_file_strings)

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
        write_embeddings_to_csv(f"dgl/edges_{i}.csv", "graph_id,src_id,dst_id\n", edges_lines[i])

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