import math
import torch
import knng_utils
import random
import numpy as np
from estimator import Estimator
from adgnn_model import ADGNN
from adgnn_search_space.search_space_encoding import (perturbation_intensity_candidate,
                                                      perturbation_probability_candidate,
                                                      edge_attribute_aggregator_candidate,
                                                      node_attribute_aggregator_candidate,
                                                      updator_dimension_candidate,
                                                      updator_activation_candidate,
                                                      fusion_candidate,
                                                      node_attribute_aggregator_neighbors)

# 将所有组件的候选项按照指定顺序组合成列表
components_candidates = [
    perturbation_intensity_candidate,
    perturbation_probability_candidate,
    edge_attribute_aggregator_candidate,
    node_attribute_aggregator_candidate,
    updator_dimension_candidate,
    updator_activation_candidate,
    fusion_candidate,
    node_attribute_aggregator_neighbors]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义节点类，表示蒙特卡洛树搜索中的一个节点
class Node:

    def __init__(self, parent=None, child_component_index=0, component_value=None):
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表
        self.child_component_index = child_component_index  # 当前节点对应其子节点的GNN组件索引
        self.component_value = component_value  # 当前节点选择的组件值
        self.visits = 0  # 节点被访问的次数
        self.total_value = 0.0  # 节点的累计评估值
        # 未尝试的候选项列表
        if child_component_index < len(components_candidates):
            self.untried_values = components_candidates[child_component_index][:]
        else:
            self.untried_values = []
        self._is_fully_expanded = False  # 节点是否已完全扩展，命名加下划线

    # 判断节点是否是终端节点（所有组件都已选择）
    def is_terminal(self):
        return self.child_component_index == len(components_candidates)

    # 判断节点是否已完全扩展，方法名与属性名区分
    def is_fully_expanded(self):
        return self._is_fully_expanded

    # 使用 UCB1 算法选择最佳子节点
    def selection(self, exploration_weight):
        best_score = float('-inf')
        best_children = []
        for child in self.children:

            if child.visits == 0:
                # 如果子节点未被访问，赋予无限大的 UCB1 值, 以确保未访问的节点优先被选择，从而实现探索新节点的目的
                score = float('inf')
            else:
                # 确保父节点的访问次数不为零
                parent_visits = max(self.visits, 1)

                # 计算平均价值
                exploit = child.total_value / child.visits
                # 计算探索项
                explore = exploration_weight * math.sqrt(math.log(parent_visits) / child.visits)

                score = exploit + explore

            if score == best_score:
                best_children.append(child)

            if score > best_score:
                best_children = [child]
                best_score = score

        # 随机选择得分最高的子节点
        return random.choice(best_children)

    # 扩展节点，创建一个新的子节点
    def expansion(self):

        # 当该节点有可扩展的节点时,直到创建一个可扩展节点或达到完全可扩展条件才退出此节点的扩展操作
        while self.untried_values:

            # 从未尝试的候选项中选择一个值
            child_value = self.untried_values.pop()
            # 临时构建架构，检查约束条件
            temp_architecture = []
            current_node = self
            while current_node.parent is not None:
                temp_architecture.append(current_node.component_value)
                current_node = current_node.parent
            temp_architecture = temp_architecture[::-1]
            temp_architecture.append(child_value)

            current_node = self
            is_valid = True

            # 在搜索过程达到选择aggregator组件候选项扩展时,才开启扩展有效性检查,其他情况不用检查

            # 对aggregator不能全是None的扩展约束
            if current_node.child_component_index == 3:
                if current_node.component_value == "None" and child_value == "None":
                    is_valid = False

            # 对子节点merge在aggregator其中一个是None时子节点merge只能扩展为None的约束
            if current_node.child_component_index == 6:

                if temp_architecture[2] == "None" or temp_architecture[3] == "None":

                    if temp_architecture[6] != "None":

                        is_valid = False
                else:
                    if temp_architecture[6] == "None":

                        is_valid = False

            # 判断此次扩展的有效性
            if is_valid:
                # 创建新的子节点
                child_node = Node(
                    parent=self,
                    child_component_index=self.child_component_index + 1,
                    component_value=child_value
                )
                self.children.append(child_node)

                # 如果没有未尝试的值，标记为已完全扩展
                if not self.untried_values:
                    self._is_fully_expanded = True

                return child_node

        # 如果所有未尝试的值都无效扩展，标记此节点为已完全扩展
        self._is_fully_expanded = True

        return None

# 树策略：选择和扩展节点
def monte_carlo_tree_construction(node, exploration_weight):

    while not node.is_terminal():
        if not node.is_fully_expanded():
            # 节点未完全扩展，进行扩展
            child_node = node.expansion()
            if child_node is not None:
                return child_node
            else:
                # 节点node无法扩展当前选择的子节点child_node,重新扩展节点node
                continue
        else:
            # 节点已完全扩展，选择最佳子节点
            node = node.selection(exploration_weight)
    return node

# 默认策略：模拟（随机选择剩余的组件）
def simulation(node, data_name, estimator_hp_dict):

    feedback_name = None
    current_node = node
    architecture = []
    # 回溯获取已选择的组件值
    while current_node.parent is not None:
        architecture.append(current_node.component_value)
        current_node = current_node.parent
    architecture = architecture[::-1]  # 反转列表，得到正确的顺序

    # 随机选择剩余的组件，构建完整的架构
    for index in range(node.child_component_index, len(components_candidates)):
        value = random.choice(components_candidates[index])
        architecture.append(value)

    print("Search Depth: ", node.child_component_index-1,
          " Search GNN Candidate: ", node.component_value,
          " Search Architecture: ", architecture)

    # dist_agg=None,feat_agg=None结构无效模拟
    if architecture[2] == "None" and architecture[3] == "None":

        score = 0
        print("Invalid Search Architecture Situation 1:", architecture)
        print(24 * "=")

    # dist_agg, merge=None 或 feat_agg,merge!=None结构无效模拟
    elif architecture[2] == "None" or architecture[3] == "None":

        if architecture[6] != "None":
            print("Invalid Search Architecture Situation 2:", architecture)
            print(24 * "=")
            score = 0
        else:
            score, feedback_name = estimator(data_name, estimator_hp_dict, architecture)
            print(24 * "=")

    # dist_agg,feat_agg!=None merge=None结构无效性模拟
    elif architecture[2] != "None" and architecture[3] != "None":

        if architecture[6] == "None":
            print("Invalid Search Architecture Situation 3:", architecture)
            print(24 * "=")
            score = 0

        else:
            score, feedback_name = estimator(data_name, estimator_hp_dict, architecture)
            print(24 * "=")
    else:
        score, feedback_name = estimator(data_name, estimator_hp_dict, architecture)
        print(24 * "=")

    if feedback_name != None:
        print("Feedback Name: ", feedback_name)
    return score

# 回溯更新节点的访问次数和累计评估值，增加折扣因子
def backpropagation(node, reward, discount_factor=0.9):

    current_reward = reward  # 当前回报
    while node is not None:
        node.visits += 1
        node.total_value += current_reward
        # 折扣回报
        current_reward *= discount_factor
        node = node.parent

# 蒙特卡洛树搜索算法

# 收集深度为7（完整架构）的叶子节点
def collect_full_architecture_nodes(node):

    full_nodes = []

    if node.is_terminal():
        full_nodes.append(node)
    else:
        for child in node.children:
            full_nodes.extend(collect_full_architecture_nodes(child))

    return full_nodes

def mcts(data_name,
         root,
         iterations,
         exploration_weight_list,
         best_architecture_select,
         top_gnn,
         estimator_hp_dict,
         stop_threshold,
         discount_factor=0.9):

    search_converged = False
    for epoch, exploration_weight in zip(iterations, exploration_weight_list):

        if search_converged:
            print("MCTS Search Has Converged The Converged Iteration is: ", epoch-1)
            break

        print("Search Epoch:", epoch+1)
        # 选择和扩展
        leaf = monte_carlo_tree_construction(root, exploration_weight)
        # 模拟
        reward = simulation(leaf, data_name, estimator_hp_dict)
        # 回溯更新，加入折扣因子
        backpropagation(leaf, reward, discount_factor)

        # ++++++++++++++++++++++++++++++++++++
        # 输出MCT中达到终端节点的所有Path

        # 找到mct此轮迭代后所有的叶子节点
        leaf_nodes = collect_leaf_nodes(root)

        print("Promising Search Path:")

        terminal_node_num = 0

        for leaf_node in leaf_nodes:
            # 获取从根节点到该叶子节点的路径
            path = []
            current = leaf_node

            tree_depth = 0
            while current.parent is not None:
                path.append(f"{current.component_value}")
                current = current.parent
                tree_depth += 1
            # 输出终端节点信息
            if tree_depth == 7:
                terminal_node_num += 1
                path = path[::-1]  # Reverse to get correct order
                path_str = " -> ".join(path)
                print(f"Search Path: {path_str} | Visits: {leaf_node.visits} | Total Value: {leaf_node.total_value:.4f}")

                if leaf_node.visits > int(len(iterations)*stop_threshold):
                    search_converged = True
        #++++++++++++++++++++++++++++++++++++

    ##################################################################################
    # 选择MCT中终端节点基于平均评估值或总访问次数前top_gnn条Path作为有前途的GNN Architecture返回
    # 由于探索深度为7的路径是从root节点出发, 所以需要访问节点的子节点进行递归调用
    full_architecture_nodes = collect_full_architecture_nodes(root)

    # 根据节点的访问次数或平均评估值对架构进行排序
    if best_architecture_select == "max_visit":
        # 按照访问次数排序
        sorted_nodes = sorted(full_architecture_nodes, key=lambda x: x.visits, reverse=True)
    elif best_architecture_select == "avg_value":
        # 按照平均评估值排序
        sorted_nodes = sorted(full_architecture_nodes, key=lambda x: x.total_value / x.visits if x.visits > 0 else 0,
                              reverse=True)
    else:
        # 默认按照平均评估值排序
        sorted_nodes = sorted(full_architecture_nodes, key=lambda x: x.total_value / x.visits if x.visits > 0 else 0,
                              reverse=True)

    # 选取前top_gnn个有前途的架构
    if top_gnn > terminal_node_num:

        top_gnn = terminal_node_num

    top_architectures = []

    for node in sorted_nodes[:top_gnn]:
        # 构建架构
        architecture = []
        current_node = node
        while current_node.parent is not None:
            architecture.append(current_node.component_value)
            current_node = current_node.parent
        architecture = architecture[::-1]  # Reverse to get correct order

        # 计算平均评估值
        avg_value = node.total_value / node.visits if node.visits > 0 else 0

        top_architectures.append({
            'architecture': architecture,
            'visits': node.visits,
            'total_value': node.total_value,
            'avg_value': avg_value})

    # 打印前10个有前途的架构
    print("Top ", top_gnn, " Promising Architectures:")
    for idx, arch_info in enumerate(top_architectures, 1):
        arch = arch_info['architecture']
        print(f"Rank {idx}: Architecture: {arch} | Visits: {arch_info['visits']} | Total Value: {arch_info['total_value']:.4f} | Avg Value: {arch_info['avg_value']:.4f}")

    # 返回最优架构和其平均评估值
    best_architecture = top_architectures[0]['architecture'] if top_architectures else []
    best_val_score = top_architectures[0]['avg_value'] if top_architectures else 0
    ##################################################################################

    return best_architecture, best_val_score, top_architectures


def estimator(data_name, estimator_hp_dict, adgnn_architecture):

    adgnn = ADGNN(data_name, adgnn_architecture).to(device)

    estimator = Estimator(adgnn.k_graph, estimator_hp_dict)

    feedback, _ = estimator.estimate(adgnn, adgnn_test=False)

    score, feedback_name = feedback[0], feedback[1]

    return score, feedback_name

def cosine_annealing(max_iterations, interval=[]):

    eta_min = interval[0]
    eta_max = interval[1]
    T_max = max_iterations
    iterations = max_iterations

    weight_list = [eta_min + 0.5 * (eta_max - eta_min) * (np.cos(np.pi * t / T_max) + 1) for t in range(iterations)]

    return weight_list

def constant(max_iterations, c):

    weight_list = [c for iter in range(max_iterations)]

    return weight_list

# 收集整棵树的叶子节点
def collect_leaf_nodes(node):

    leaf_nodes = []
    if not node.children:
        leaf_nodes.append(node)
    else:
        for child in node.children:
            leaf_nodes.extend(collect_leaf_nodes(child))
    return leaf_nodes

data_list = ["ANNTHYROID", "SATELLITE", "MI-F"]

# 基于数据集名称与搜索空间预处理K_Graph图数据
knng_utils.k_neighbor_graph_preprocessing_based_on_search_space(data_list=data_list)

# 设置迭代次数
max_iterations = 200
iteration_list = [epoch for epoch in range(max_iterations)]

# 初始化根节点
root_node = Node()
# 最优路径选择方式
best_architecture_select = "avg_value"

# data_name = "HRSS"
# exploration_weight = cosine_annealing(max_iterations, [1, 2])
# estimator_hp_dict = {"learning_rate": 0.01,
#                      "weight_decay": 0.1,
#                      "training_epoch": 5}

# data_name = "MI-F"
# exploration_weight = cosine_annealing(max_iterations, [1, 2])
# estimator_hp_dict = {"learning_rate": 0.001,
#                      "weight_decay": 0.1,
#                      "training_epoch": 10}

data_name = "MI-V"
exploration_weight = cosine_annealing(max_iterations, [10, 20])
estimator_hp_dict = {"learning_rate": 0.001,
                     "weight_decay": 0.1,
                     "training_epoch": 5}

# data_name = "SATELLITE"
# exploration_weight = cosine_annealing(max_iterations, [1, 2])
# estimator_hp_dict = {"learning_rate": 0.001,
#                      "weight_decay": 0.1,
#                      "training_epoch": 5}

# data_name = "ANNTHYROID"
# exploration_weight = cosine_annealing(max_iterations, [19, 20])
# estimator_hp_dict = {"learning_rate": 0.0000001,
#                      "weight_decay": 0.001,
#                      "training_epoch": 20}

# 输出有前途GNN结构个数
top_gnn = 1
discount_factor = 0.9
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

print("Best Search Result ADGNN Architecture:", best_architecture,
      " Best Feedback Score:", best_val_score)

print("Top Promising Architectures:")
for arch in (top_architectures):
    print(arch)

print("Best ADGNN Architecture Testing......")


for archi in (top_architectures):

    print("Test  ADGNN Architecture:", archi["architecture"])
    avg_score = []
    for _ in range(5):

        adgnn = ADGNN(data_name, archi["architecture"]).to(device)

        hp_dict = {"learning_rate": 0.001,
                   "weight_decay": 0.1,
                   "training_epoch": 200}

        estimator = Estimator(adgnn.k_graph, hp_dict)

        score, estimate_name = estimator.estimate(adgnn, adgnn_test=True)

        print(estimate_name, ":", score)
        avg_score.append(score)

    print("Data Set:", data_name, "Avg", estimate_name, ":", np.mean(avg_score))