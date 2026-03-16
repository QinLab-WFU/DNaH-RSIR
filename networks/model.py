import torch
import torch.nn as nn
from torchvision.models import alexnet
# from test import FusionModel , SpatialAttention
from networks.relative_similarity import RelativeSimilarity
from networks.ca_net import *
from utils.attention_zoom import *
from networks.navigateNet import attention_net
from networks.hbp import HBP
from networks.configs import *
from networks.fg_vit import *
class RelaHash(nn.Module):
    def __init__(self,
                 nbit, nclass, batchsize,
                 init_method='M',
                 pretrained=True, freeze_weight=False,
                 device='cuda',
                 **kwargs):
        super(RelaHash, self).__init__()

        # self.backbone = FusionModel(hash_bit=nbit)
        # self.backbone = CANet(bit=nbit,nclass=nclass)

        # self.backbone = attention_net()
        self.backbone = HBP() # 9071350584181319

        # self.configs = get_b32_config() # est mAP: 0.883061
        # self.configs = get_b16_config() # 0.9184084181206936
        # self.backbone = VisionTransformer(self.configs,img_size=(224,224))


        # self.hash_fc = nn.Sequential(
        #     nn.BatchNorm1d(self.backbone.num_ftrs // 2 * 3, affine=True),
        #     nn.Linear(self.backbone.num_ftrs // 2 * 3, self.backbone.feature_size),
        #     nn.BatchNorm1d(self.backbone.feature_size, affine=True),
        #     nn.ELU(inplace=True),
        #     nn.Linear(self.backbone.feature_size, nbit),
        # )
        # self.hash_fc = nn.Sequential(
        #     nn.ReLU(),
        #     nn.BatchNorm1d(768,affine=True),
        #     nn.Linear(768,512),
        #     nn.BatchNorm1d(512, affine=True),
        #     nn.ReLU(),
        #     nn.Linear(512,nbit),
        #     nn.BatchNorm1d(nbit, affine=True),
        #     nn.Tanh()
        # )
        self.hash_fc = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(2048, affine=True),
            nn.Linear(2048, self.backbone.feature_size),
            nn.BatchNorm1d(self.backbone.feature_size, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone.feature_size, nbit),
            nn.Tanh(),
        )
        # nn.init.normal_(se.weight, std=0.01)
        self.relative_similarity = RelativeSimilarity(nbit, nclass, batchsize, init_method=init_method, device=device)

    def get_hash_params(self):
        return list(self.relative_similarity.parameters()) + list(self.hash_fc.parameters())
    
    def get_backbone_params(self):
        return self.backbone.get_features_params()
    
    def get_centroids(self):
        return self.relative_similarity.centroids
        
    def forward(self, x,if_zoom=False):
        # if if_zoom == True:
        #     _,_,_,y_zoom,_ = self.backbone(x)
        #     return y_zoom
        # f11, f22, f33, y33, feats = self.backkbone(x)
        f44= self.backbone(x)
        # print(f'f44 {f44.shape}')
        # exit()
        # f44 = torch.cat((f11, f22, f33), -1)
        # f44 = torch.tanh(f44)
        z = self.hash_fc(f44)
        # z2 = self.hash_fc(f33)
        logits = self.relative_similarity(z) # hashcode
        return logits, z,



# net = RelaHash(nclass=10 , nbit= 16 , batchsize= 32)
# print(net)

class ProxyParam(nn.Module):
    def __init__(self,cls,bit,device):
        super(ProxyParam,self).__init__()
        self.proxy = nn.Parameter(torch.FloatTensor(torch.randn((cls,bit),device=device)),requires_grad=True)