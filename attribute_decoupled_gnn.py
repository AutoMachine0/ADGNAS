import os
import torch
import utils
import pickle
import numpy as np
import torch.nn as nn
from estimator import Estimator
from adgnn_search_space.mlp import MLP
from adgnn_search_space.act_pool import ActPool
from adgnn_search_space.conv_pool import ConvPool


class NodeAttributeAggregator(torch.nn.Module):

    def __init__(self,
                 num_node_features,
                 architecture):

        super(NodeAttributeAggregator, self).__init__()

        aggregation_manner = architecture[0]
        hidden_dimension = int(architecture[1])
        activation_manner = architecture[2]

        self.layer1_act_pool = ActPool()
        self.layer2_act_pool = ActPool()

        self.pre_process_mlp = MLP(input_dim=num_node_features,
                                   output_dim=hidden_dimension)

        self.post_process_mlp = MLP(input_dim=hidden_dimension,
                                    output_dim=hidden_dimension)

        self.layer1_aggregation = ConvPool(hidden_dimension, hidden_dimension).get_conv(aggregation_manner)
        self.layer1_act = self.layer1_act_pool.get_act(activation_manner)

        self.layer2_aggregation = ConvPool(hidden_dimension, hidden_dimension).get_conv(aggregation_manner)
        self.layer2_act = self.layer2_act_pool.get_act(activation_manner)

    def forward(self, graph):

        x = graph.x

        edge_index = graph.edge_index

        x = self.pre_process_mlp(x)

        x = self.layer1_aggregation(x, edge_index)

        x = self.layer1_act(x)

        x = self.layer2_aggregation(x, edge_index)

        x = self.layer2_act(x)

        y = self.post_process_mlp(x)

        return y


class EdgeAttributeAggregator(torch.nn.Module):

    def __init__(self,
                 architecture):

        super(EdgeAttributeAggregator, self).__init__()

        self.concatenate_k = int(architecture[0])
        mlp_layer_num = 3
        hidden_dimension = architecture[1]
        activation_manner = architecture[2]

        updating_mlp_hidden_dimension_list = [hidden_dimension for _ in range(mlp_layer_num)]
        updating_act_list = [activation_manner for _ in range(mlp_layer_num)]

        self.updating_mlp = MLP(input_dim=self.concatenate_k,
                                output_dim=hidden_dimension,
                                hidden_dim_list=updating_mlp_hidden_dimension_list,
                                act_list=updating_act_list)

    def forward(self, graph):

       x = graph.edge_attr

       # distance k nieghbors concatenate aggregation
       x = x.reshape(-1, self.concatenate_k)

       # updating
       y = self.updating_mlp(x)

       return y


class AttributeDecoupledGNN(torch.nn.Module):

    def __init__(self,
                 data_name,
                 attribute_decoupled_gnn_architecture):

        super(AttributeDecoupledGNN, self).__init__()

        # 获取持久化本地的K_graph数据，并且数据经过了扰动强度与概率的预处理
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        I, P, D = (str(attribute_decoupled_gnn_architecture[0]),
                   str(attribute_decoupled_gnn_architecture[1]),
                   attribute_decoupled_gnn_architecture[2])

        # 当 D为None时给D赋值保证k_graph获取不出错,但后续建模adgnn不会使用到distance aggregator
        if D == "None":
            D = "5"

        self.pre_k_graph_name = data_name + "_I_" + str(I) + "_P_" + str(P) + "_F_1_D_" + D + ".pkl"
        file_path = os.getcwd() + "/knng_pickle/"
        k_graph_data_name = file_path + self.pre_k_graph_name

        if not os.path.exists(k_graph_data_name):
            raise ValueError(self.pre_k_graph_name, "not exist!")
        else:
            with open(k_graph_data_name, "rb") as file:
                self.k_graph = pickle.load(file)
                num_node_features = self.k_graph[0].x.shape[1]
                num_classes = self.k_graph[0].num_classes
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.feat_architecture = [attribute_decoupled_gnn_architecture[3],
                                  attribute_decoupled_gnn_architecture[4],
                                  attribute_decoupled_gnn_architecture[5]]

        self.dist_architecture = [attribute_decoupled_gnn_architecture[2],
                                  attribute_decoupled_gnn_architecture[4],
                                  attribute_decoupled_gnn_architecture[5]]

        self.merge = attribute_decoupled_gnn_architecture[6]

        input_dim = 0

        if self.feat_architecture[0] == "None":

            self.dist_net = EdgeAttributeAggregator(self.dist_architecture)

            input_dim = self.dist_architecture[1]

            self.merge = "None"

        elif self.dist_architecture[0] == "None":

            self.feat_net = NodeAttributeAggregator(num_node_features,
                                                       self.feat_architecture)

            input_dim = self.feat_architecture[1]

            self.merge = "None"

        else:

            self.dist_net = EdgeAttributeAggregator(self.dist_architecture)

            self.feat_net = NodeAttributeAggregator(num_node_features,
                                                       self.feat_architecture)

            self.dist_dimension = self.dist_architecture[1]

            self.feat_dimension = self.feat_architecture[1]

        if "weighted" in self.merge:

            self.feat_weight = nn.Parameter(torch.randn(self.feat_dimension))
            self.dist_weight = nn.Parameter(torch.randn(self.dist_dimension))

        if self.merge in {"cat", "weighted_cat"}:

            self.post_process_mlp = MLP(input_dim=self.dist_architecture[1] + self.feat_architecture[1],
                                        output_dim=num_classes)

        elif self.merge in {"sum", "max", "mean", "weighted_sum"}:

            self.post_process_mlp = MLP(input_dim=self.dist_dimension,
                                        output_dim=num_classes)

        elif self.merge == "None":

            self.post_process_mlp = MLP(input_dim=input_dim,
                                        output_dim=num_classes)
        else:
            raise ValueError(self.merge + " is the wrong merge value!!!")

    def forward(self, k_graph):

        if self.k_graph != None:

            if self.feat_architecture[0] == "None":

                x = self.dist_net(k_graph[1])

            elif self.dist_architecture[0] == "None":

                x = self.feat_net(k_graph[0])

            else:
                feat_out = self.feat_net(k_graph[0])

                dist_out = self.dist_net(k_graph[1])

                x = self.merge_computation(feat_out, dist_out)

            y = self.post_process_mlp(x)

            y = torch.squeeze(y, 1)

            y = torch.sigmoid(y)

        else:
            raise ValueError("Forward computation fail, K_graph not exist !!!")

        return y

    def merge_computation(self, feat_out, dist_out):

        if "weighted" in self.merge:

            norm_feat_weight = torch.softmax(self.feat_weight, dim=-1)
            norm_dist_weight = torch.softmax(self.dist_weight, dim=-1)

        if self.merge == "cat":
            merge_out = torch.cat((feat_out, dist_out), dim=1)

        elif self.merge == "sum":
            merge_out = torch.add(feat_out, dist_out)

        elif self.merge == "max":
            merge_out = torch.max(feat_out, dist_out)

        elif self.merge == "mean":
            merge_out = torch.add(feat_out, dist_out) / 2

        elif self.merge == "weighted_sum":
            merge_out = feat_out * norm_feat_weight + dist_out * norm_dist_weight

        elif self.merge == "weighted_cat":
            merge_out = torch.cat((feat_out * norm_feat_weight, dist_out * norm_dist_weight), dim=1)

        else:
            raise ValueError(self.merge + " is the wrong merge value!!!")

        return merge_out

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # best architecture
    gnn_ANNTHYROID = [0.05, 1.0, '10', 'SGConv', 1024, 'Relu6', 'weighted_sum']
    gnn_HRSS = [0.1, 0.1, '5', 'SGConv', 512, 'Relu6', 'cat']
    gnn_SATELLITE = [0.01, 1.0, '10', 'GCNConv', 1024, 'Relu', 'sum']
    gnn_MI_F = [1.0, 0.7, "10", "GCNConv", 128, "Relu", "weighted_cat"]
    gnn_MI_V = [0.5, 0.1, '1', 'SGConv', 256, 'Relu6', 'mean']

    data_list = ["HRSS", "MI-F", "MI-V", "SATELLITE", "ANNTHYROID"]
    gnn_list = [gnn_HRSS, gnn_MI_F, gnn_MI_V, gnn_SATELLITE, gnn_ANNTHYROID]

    utils.k_neighbor_graph_preprocessing_based_on_search_space(data_list=data_list)
    adgnn_test_flag = True

    for data_name, gnn in zip(data_list, gnn_list):
        avg_score = []
        for _ in range(5):
            adgnn = AttributeDecoupledGNN(data_name, gnn).to(device)
            hp_dict = {"learning_rate": 0.001,
                       "weight_decay": 0.1,
                       "training_epoch": 200}

            estimator = Estimator(adgnn.k_graph, hp_dict)
            if adgnn_test_flag:
                score, estimate_name = estimator.estimate(adgnn, adgnn_test=adgnn_test_flag)
            else:
                feedback, estimate_name = estimator.estimate(adgnn, adgnn_test=adgnn_test_flag)
                score = feedback[0]

            print(estimate_name, ":", score)
            avg_score.append(score)

        print("Data Set:", data_name, "Avg", estimate_name, ":", np.mean(avg_score))