import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.tool import load_preweights
from relative_similarity import RelativeSimilarity
import tensorflow as tf
from torchvision import models
import torchvision
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class FusionModel(nn.Module):
    def __init__(self, hash_bit,numclass,batchsize,init_method='M',device='cuda'):
        super(FusionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.localbranch = SpatialAttention(kernel_size=13)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256),
        )
        # self.fc = nn.Linear(512, hash_bit)
        # self.tanh = nn.Tanh()
        # 使用新的量化误差

        self.hash_fc = nn.Sequential(
            nn.Linear(512, hash_bit, bias=False),
            nn.BatchNorm1d(hash_bit, momentum=0.1)
        )
        self.tanh = nn.Tanh()

        nn.init.normal_(self.hash_fc[0].weight, std=0.01)

        self.relative_similarity = RelativeSimilarity(hash_bit, numclass, batchsize, init_method=init_method, device=device)

    def get_hash_params(self):
        return list(self.relative_similarity.parameters()) + list(self.hash_fc.parameters())

    # def get_backbone_params(self):
    #     return self.backbone.get_features_params()

    def get_centroids(self):
        return self.relative_similarity.centroids

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        attention = self.localbranch(x)
        local_fea = x * attention
        x = self.maxpool3(x)
        print(x.shape)
        x = x.view(x.size(0), 256 * 6 * 6)
        print(x.shape)
        glo_fea = self.classifier(x)
        glo_fea = glo_fea.unsqueeze(-1).unsqueeze(-1)
        fea = torch.cat([glo_fea.expand(local_fea.size()), local_fea], dim=1)
        fea = self.avgpool(fea)
        fea = fea.view(fea.size(0), -1)
        # fea = self.fc(fea)
        # 使用新的hash层

        fea = self.hash_fc(fea)
        fea = self.tanh(fea)
        fea = F.normalize(fea, dim=1)
        # 新添加的有关量化损失
        logits = self.relative_similarity(fea)

        return fea,logits

# 其他
# class RelaHash(nn.Module):
#     def __init__(self,
#                  nbit, nclass, batchsize,
#                  init_method='M',
#                  pretrained=True, freeze_weight=False,
#                  device='cuda',
#                  **kwargs):
#         super(RelaHash, self).__init__()
#
#         self.backbone = FusionModel(hash_bit=nbit)
#
#         self.hash_fc = nn.Sequential(
#             nn.Linear(512, nbit, bias=False),
#             nn.BatchNorm1d(nbit, momentum=0.1)
#         )
#         nn.init.normal_(self.hash_fc[0].weight, std=0.01)
#
#         self.relative_similarity = RelativeSimilarity(nbit, nclass, batchsize, init_method=init_method, device=device)
#
#     def get_hash_params(self):
#         return list(self.relative_similarity.parameters()) + list(self.hash_fc.parameters())
#
#     def get_backbone_params(self):
#         return self.backbone.get_features_params()
#
#     def get_centroids(self):
#         return self.relative_similarity.centroids
#
#     def forward(self, x):
#         x = self.backbone(x)
#         z = self.hash_fc(x)
#         logits = self.relative_similarity(z)
#         return  z,logits
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_result, _ = torch.max(x, dim=1, keepdim=True)
#         avg_result = torch.mean(x, dim=1, keepdim=True)
#         result = torch.cat([max_result, avg_result], 1)
#         output = self.conv(result)
#         output = self.sigmoid(output)
#         return output
#
# class FusionModel(nn.Module):
#     def __init__(self, hash_bit):
#         super(FusionModel, self).__init__()
#         model = torchvision.models.alexnet(pretrained = True)
#         self.fearture = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.localbranch = SpatialAttention(kernel_size=13)
#         self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, 256),
#         )
#         self.classifier = model.classifier[-1]
#         self.in_features = model.classifier[6].in_features
#
#     def get_features_params(self):
#         return list(self.fearture.parameters()) +\
#             list(self.localbranch.parameters()) + list(self.maxpool3.parameters()) +\
#             list(self.avgpool.parameters()) + list(self.fc.parameters())
#
#     def train(self, mode=True):
#         super(FusionModel, self).train(mode)
#         # all dropout set to eval
#         for mod in self.modules():
#             if isinstance(mod, nn.Dropout):
#                 mod.eval()
#
#     def forward(self, x):
#         x = self.fearture(x)
#         attention = self.localbranch(x)
#         local_fea = x * attention
#         x = self.maxpool3(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         glo_fea = self.fc(x)
#         glo_fea = glo_fea.unsqueeze(-1).unsqueeze(-1)
#         fea = torch.cat([glo_fea.expand(local_fea.size()), local_fea], dim=1)
#         fea = self.avgpool(fea)
#         fea = fea.view(fea.size(0), -1)
#         return fea


# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_result, _ = torch.max(x, dim=1, keepdim=True)
#         avg_result = torch.mean(x, dim=1, keepdim=True)
#         result = torch.cat([max_result, avg_result], 1)
#         output = self.conv(result)
#         output = self.sigmoid(output)
#         return output
#
# class FusionModel(nn.Module):
#     def __init__(self, hash_bit,numclass,batchsize,init_method='M',device='cuda'):
#         super(FusionModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
#         self.relu4 = nn.ReLU(inplace=True)
#         self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.relu5 = nn.ReLU(inplace=True)
#         self.localbranch = SpatialAttention(kernel_size=13)
#         self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, 256),
#         )
#         # self.fc = nn.Linear(512, hash_bit)
#         # self.tanh = nn.Tanh()
#         # 使用新的量化误差
#
#         self.hash_fc = nn.Sequential(
#             nn.Linear(512, hash_bit, bias=False),
#             nn.BatchNorm1d(hash_bit, momentum=0.1)
#         )
#         self.tanh = nn.Tanh()
#
#         nn.init.normal_(self.hash_fc[0].weight, std=0.01)
#
#         self.relative_similarity = RelativeSimilarity(hash_bit, numclass, batchsize, init_method=init_method, device=device)
#
#     # def get_hash_params(self):
#     #     return list(self.relative_similarity.parameters()) + list(self.hash_fc.parameters())
#
#     # def get_backbone_params(self):
#     #     return self.backbone.get_features_params()
#
#     def get_centroids(self):
#         return self.relative_similarity.centroids
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)
#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.conv4(x)
#         x = self.relu4(x)
#         x = self.conv5(x)
#         x = self.relu5(x)
#         attention = self.localbranch(x)
#         local_fea = x * attention
#         x = self.maxpool3(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         glo_fea = self.classifier(x)
#         glo_fea = glo_fea.unsqueeze(-1).unsqueeze(-1)
#         fea = torch.cat([glo_fea.expand(local_fea.size()), local_fea], dim=1)
#         fea = self.avgpool(fea)
#         fea = fea.view(fea.size(0), -1)
#         # fea = self.fc(fea)
#         # 使用新的hash层
#         fea = self.hash_fc(fea)
#         fea = self.tanh(fea)
#         fea = F.normalize(fea, dim=1)
#         # 新添加的有关量化损失
#         logits = self.relative_similarity(fea)
#
#         return fea,logits

