import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

class Estimator(object):

    def __init__(self, k_graph, hp_config):

        self.k_graph = k_graph
        self.learning_rate = hp_config["learning_rate"]
        self.weight_decay = hp_config["weight_decay"]
        self.training_epoch = hp_config["training_epoch"]

    def estimate(self, adgnn, adgnn_test=False, validation_test_correlation=False):

        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(adgnn.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        best_train_auc = 0
        best_val_auc = 0
        best_test_auc_process = 0
        best_val_score_loss = 0
        best_train_score_loss = 0
        for epoch in range(self.training_epoch):

            adgnn.train()
            optimizer.zero_grad()
            y_pred = adgnn(self.k_graph)
            # 前向计算出一个Batch多个样本的loss后,可以对多个样本的loss求和.sum()或.mean()进行一次反向传播,
            # 也可对单个样本的loss进行反向传播,但会增加计算开销容易导致梯度爆炸问题
            loss = criterion(y_pred[self.k_graph[0].train_mask == 1], self.k_graph[0].y[self.k_graph[0].train_mask == 1]).sum()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                adgnn.eval()

                y_pred = adgnn(self.k_graph)
                # 计算全部样本的损失
                loss = criterion(y_pred,
                                 self.k_graph[0].y)

                y_label_train = self.k_graph[0].y[self.k_graph[0].train_mask == 1].cpu()
                y_pred_train = y_pred[self.k_graph[0].train_mask == 1].cpu()
                # 计算train样本的平均损失
                train_loss = loss[self.k_graph[0].train_mask == 1].mean()
                train_auc = 100 * roc_auc_score(y_label_train, y_pred_train)


                y_label_val = self.k_graph[0].y[self.k_graph[0].val_mask == 1].cpu()
                y_pred_val = y_pred[self.k_graph[0].val_mask == 1].cpu()
                # 计算val样本的平均损失
                val_loss = loss[self.k_graph[0].val_mask == 1].mean()
                val_auc = 100 * roc_auc_score(y_label_val, y_pred_val)

                y_label_test = self.k_graph[0].test_y
                y_pred_test = y_pred[self.k_graph[0].test_mask == 1].cpu()
                test_auc = 100 * roc_auc_score(y_label_test, y_pred_test)

                if train_auc > best_train_auc:
                    best_train_auc = train_auc
                    best_train_score_loss = train_loss.item()

                if test_auc > best_test_auc_process:
                    best_test_auc_process = test_auc

                if val_auc > best_val_auc:
                    best_val_score_loss = val_loss.item()
                    best_val_auc = val_auc

        if adgnn_test:

                estimate_name = "Test Score"

                if validation_test_correlation:

                    return best_train_auc, best_val_auc, best_train_score_loss, best_val_score_loss, best_test_auc_process

                return best_test_auc_process, estimate_name

        estimate_name = "Validation Score"

        feedback = [val_auc, 'Final Val AUC ']

        return feedback, estimate_name

if __name__ == "__main__":
    
   pass