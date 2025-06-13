import torch
import torch.nn as nn
import torch.nn.functional as F
import xlrd
import math
import torch.autograd as autograd
import openpyxl
import math
import torch

# 打开 codetable.xlsx 文件
workbook = openpyxl.load_workbook('codetable.xlsx')
sheet = workbook.active  # 获取活动的 sheet

# 获取指定的单元格的值
row_index = 17  # 因为 openpyxl 从 1 开始计数，行号为 16 的应改为 17
col_index = math.ceil(math.log(8, 2)) + 1  # 列号也是从 1 开始计数

# 读取指定单元格的值
threshold = sheet.cell(row=row_index, column=col_index).value
print(threshold)

# 设置设备
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


class HyP(torch.nn.Module):
    # def __init__(self,hash_bit, numclass):
    def __init__(self,hash_bit, numclass ):
        torch.nn.Module.__init__(self)
        torch.manual_seed(0)
        # Initialization
        self.proxies = torch.nn.Parameter(torch.randn(numclass, hash_bit).to(device))

        nn.init.kaiming_normal_(self.proxies, mode = 'fan_out')

    def forward(self, x = None, batch_y = None, noise=None,alpha=0.5,grama=0.2):

        P_one_hot = batch_y

        cos = F.normalize(x, p = 2, dim = 1).mm(F.normalize(self.proxies, p = 2, dim = 1).T)
        pos = 1 - cos
        neg = F.relu(cos - threshold)

        P_num = len(P_one_hot.nonzero())
        N_num = len((P_one_hot == 0).nonzero())

        pos_term = torch.where(P_one_hot  ==  1, pos.to(torch.float32), torch.zeros_like(cos).to(torch.float32)).sum() / P_num
        neg_term = torch.where(P_one_hot  ==  0, neg.to(torch.float32), torch.zeros_like(cos).to(torch.float32)).sum() / N_num
        if alpha > 0:
            index = batch_y.sum(dim = 1) > 1
            y_ = batch_y[index].float()
            x_ = x[index]
            cos_sim = y_.mm(y_.T)
            if len((cos_sim == 0).nonzero()) == 0:
                reg_term = 0
            else:
                x_sim = F.normalize(x_, p = 2, dim = 1).mm(F.normalize(x_, p = 2, dim = 1).T)
                neg = alpha * F.relu(x_sim - threshold)
                reg_term = torch.where(cos_sim == 0, neg, torch.zeros_like(x_sim)).sum() / len((cos_sim == 0).nonzero())
        else:
            reg_term = 0
        # 约束损失
        # noise_loss = x.mul(noise).mean()
        # loss = pos_term + neg_term + reg_term + grama * noise_loss


        return pos_term + neg_term + reg_term



class RelaHashLoss(nn.Module):
    def __init__(self,
                 beta=8.0, ###
                 m=0.5,  ###
                 multiclass=True,
                 onehot=True,
                 **kwargs):
        super(RelaHashLoss, self).__init__()
        self.beta = beta
        self.m = m
        self.multiclass = multiclass
        self.onehot = onehot

    def compute_margin_logits(self, logits, labels):
        if self.multiclass:
            y_onehot = labels * self.m
            margin_logits = self.beta * (logits - y_onehot)
        else:
            y_onehot = torch.zeros_like(logits)
            y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
            margin_logits = self.beta * (logits - y_onehot)
        return margin_logits

    def forward(self, logits, z, labels):
        if self.multiclass:
            if not self.onehot:
                labels = F.one_hot(labels, logits.size(1))
            labels = labels.float()
            margin_logits = self.compute_margin_logits(logits, labels)
            # label smoothing
            log_logits = F.log_softmax(margin_logits, dim=1)
            # 加上
            A = ((labels==0).sum(dim=1) == labels.shape[1])
            labels[A==True] = 1
            labels_scaled = labels / labels.sum(dim=1, keepdim=True)
            loss = - (labels_scaled * log_logits).sum(dim=1)
            loss = loss.mean()
        else:
            if self.onehot:
                labels = labels.argmax(1)

            margin_logits = self.compute_margin_logits(logits, labels)
            loss = F.cross_entropy(margin_logits, labels)
        return loss
