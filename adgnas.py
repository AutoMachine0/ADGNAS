import utils
import torch
import numpy as np
from estimator import Estimator
from attribute_decoupled_gnn import AttributeDecoupledGNN
from monte_carlo_tree_search import Node, cosine_annealing, mcts

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_list = ["HRSS", "ANNTHYROID", "SATELLITE", "MI-F", "MI-V"]

utils.k_neighbor_graph_preprocessing_based_on_search_space(data_list=data_list)
c_max = 1

for data_name in data_list:

    if data_name == "MI-V":
        c_max = 2.0
    else:
        c_max = 3.0

    if data_name in {"HRSS", "MI-V"}:
        training_epoch = 10
    else:
        training_epoch = 20

    # 设置迭代次数
    max_iterations = 500
    iteration_list = [epoch for epoch in range(max_iterations)]

    # 初始化根节点
    root_node = Node()

    exploration_weight = cosine_annealing(max_iterations, [1.0, c_max])

    # 最优路径选择方式
    best_architecture_select = "avg_value"

    # GNN评估超参
    estimator_hp_dict = {"learning_rate": 0.001,
                         "weight_decay": 0.1,
                         "training_epoch": training_epoch}

    # 输出有前途GNN结构个数
    top_gnn = 1
    discount_factor = 0.7
    stop_threshold = 0.1

    # 执行蒙特卡洛树搜索
    best_architecture, best_val_score, top_architectures = mcts(data_name,
                                                                root_node,
                                                                iteration_list,
                                                                exploration_weight,
                                                                best_architecture_select,
                                                                top_gnn,
                                                                estimator_hp_dict,
                                                                stop_threshold,
                                                                discount_factor)

    print("Optimal Search Result ADGNN Architecture:", best_architecture,
          " Optimal Reward Score:", best_val_score)

    print("Optimal ADGNN Testing......")
    estimate_name = "Test"

    for archi in (top_architectures):

        print("Test ADGNN Architecture:", archi["architecture"])
        avg_score = []

        for _ in range(5):
            adgnn = AttributeDecoupledGNN(data_name, archi["architecture"]).to(device)

            hp_dict = {"learning_rate": 0.001,
                       "weight_decay": 0.1,
                       "training_epoch": 200}

            estimator = Estimator(adgnn.k_graph, hp_dict)

            score, estimate_name = estimator.estimate(adgnn, adgnn_test=True)

            print(estimate_name, ":", score)
            avg_score.append(score)

        print("Data Set:", data_name, "Avg", estimate_name, ":", np.mean(avg_score))