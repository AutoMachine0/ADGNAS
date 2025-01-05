import torch
import utils
import numpy as np
from estimator import Estimator
from attribute_decoupled_gnn import AttributeDecoupledGNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# optimal ADGNN for Different KNNGs

gnn_HRSS = [0.1, 0.1, '5', 'SGConv', 512, 'Relu6', 'cat']
gnn_MI_F = [1.0, 0.7, "10", "GCNConv", 128, "Relu", "weighted_cat"]
gnn_MI_V = [0.5, 0.1, '1', 'SGConv', 256, 'Relu6', 'mean']
gnn_SATELLITE = [0.01, 1.0, '10', 'GCNConv', 1024, 'Relu', 'sum']
gnn_ANNTHYROID = [0.05, 1.0, '10', 'SGConv', 1024, 'Relu6', 'weighted_sum']

data_list = ["HRSS", "MI-F", "MI-V", "SATELLITE", "ANNTHYROID"]
gnn_list = [gnn_HRSS, gnn_MI_F, gnn_MI_V, gnn_SATELLITE, gnn_ANNTHYROID]

adgnn_test_flag = True
estimate_name = "Test"

for data_name, gnn in zip(data_list, gnn_list):
    avg_score = []
    print("Generalization Performance Test for Data set " + data_name)
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