from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from models.FESM import FESM
from models.CFM import CFM
from models.mcm import MCM


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 输出的中间通道数
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_Backbone(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Backbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def SEMICON_backbone(pretrained=True, progress=True, **kwargs):
    model = ResNet_Backbone(Bottleneck, [3, 4, 6], **kwargs)
    if pretrained:
        state_dict = torch.load('/home/admin01/桌面/CXR/06-work/DAHNet-main/preweight/resnet50.pth')
        for name in list(state_dict.keys()):
            if 'fc' in name or 'layer4' in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict)
    return model


class TransLayer(nn.Module):
    def __init__(self, block):
        super(TransLayer, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.inplanes = 1024
        self.groups = 1
        self.base_width = 64
        self.layer4 = self._make_layer(block, 512, stride=2)

    def _make_layer(self, block, planes, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer4(x)
        return out

class Trans_Refine(nn.Module):

    def __init__(self, block, layer, is_local=True, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(Trans_Refine, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 1024
        self.dilation = 1
        self.is_local = is_local
        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self._make_layer(block, 512, layer, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        layers = []
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
            if _ == 1 and self.is_local:
                layers.append(CFM())
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer4(x)

        pool_x = self.avgpool(x)
        pool_x = torch.flatten(pool_x, 1)
        if self.is_local:
            return x, pool_x
        else:
            return pool_x

    def forward(self, x):
        return self._forward_impl(x)

import pywt
import pywt.data
from functools import partial

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# Wavelet Transform Conv(WTConv2d)
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        import matplotlib.pyplot as plt

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x


        # # 绘制输入图像
        # plt.imshow(curr_x_ll[0, 0].detach().cpu().numpy(), cmap='viridis')  # 假设通道0
        # plt.title("Input Image")
        # plt.axis('off')
        # plt.show()

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)


            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class StandardRefineBlock(nn.Module):
    """
    CSAM/self.refine_local 的标准替代模块：
    包含标准的卷积精炼，并返回双输出 (local_f, avg_local_f) 以匹配 DAHNET.forward 的签名。
    """

    def __init__(self, in_channels, out_channels):
        super(StandardRefineBlock, self).__init__()
        # 使用一个标准的 Conv + BN + ReLU 块作为特征精炼（无定制注意力）
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1. 局部精炼特征图
        local_f = self.refine_conv(x)

        # 2. 平均池化特征向量 (手动实现 Global Average Pooling, GAP)
        # torch.mean(input, dim=[2, 3]) 相当于 GAP
        avg_local_f = torch.mean(local_f, dim=[2, 3])

        # 返回两个输出以匹配 DAHNET.forward 的签名
        return local_f, avg_local_f

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class StandardConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super(StandardConv3x3, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# class SEBlock(nn.Module):
#     def __init__(self, in_channels, ratio=16):
#         super(SEBlock, self).__init__()
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Module 注册 1
#         self.fc = nn.Sequential(  # Module 注册 2
#             nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.fc(y)
#         attended_x = x * y.expand_as(x)
#
#
#         return attended_x, attended_x



class SimpleFusion_CSEM_Replacement(nn.Module):


    def __init__(self, in_channels):
        super(SimpleFusion_CSEM_Replacement, self).__init__()

        self.fusion = nn.Sequential(

            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.in_channels = in_channels

    def forward(self, x1, x2):

        x_concat = torch.cat([x1, x2], dim=1)


        fused_output = self.fusion(x_concat)

        return fused_output, fused_output

class ResNet_Refine(nn.Module):
    def __init__(self, block, layers, is_local=True, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Refine, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 1024
        self.dilation = 1
        self.is_local = is_local
        self.groups = groups
        self.base_width = width_per_group

        self.pre_layer4_conv = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1, bias=False)

        self.layer4 = WTConv2d(2048, 2048, kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 0)

    def _forward_impl(self, x):
        x = self.pre_layer4_conv(x)
        x = self.layer4(x)





        pool_x = self.avgpool(x)
        pool_x = torch.flatten(pool_x, 1)
        if self.is_local:
            return x, pool_x
        else:

            return pool_x

    def forward(self, x):
        return self._forward_impl(x)


def SEMICON_refine(is_local=True, pretrained=True, progress=True, **kwargs):
    model = ResNet_Refine(Bottleneck, 3, is_local, **kwargs)
    if pretrained:
        state_dict = torch.load('/home/admin01/桌面/CXR/06-work/DAHNet-main/preweight/resnet50.pth')
        for name in list(state_dict.keys()):
            if not 'layer4' in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict, strict=False)
    return model

def Trans(pretrained=True):
    model = TransLayer(Bottleneck)
    if pretrained:
        state_dict = torch.load('/home/admin01/桌面/CXR/06-work/DAHNet-main/preweight/resnet50.pth')
        pretrain_keys = []
        for name in list(state_dict.keys()):
            if 'layer4.0' in name:
                # print(name)
                pretrain_keys.append(name)
        for key in pretrain_keys:
            model.state_dict()[key].copy_(state_dict[key])
    #     model.load_state_dict(state_dict, strict=False)
    # print(model.state_dict())
    return model

def Trans_refine(is_local=True, pretrained=True, progress=True, **kwargs):
    model = Trans_Refine(Bottleneck, 3, is_local, **kwargs)
    if pretrained:
        state_dict = torch.load('/home/admin01/桌面/CXR/06-work/DAHNet-main/preweight/resnet50.pth')
        pretrain_keys = []
        for name in list(state_dict.keys()):
            if 'layer4.1' in name or 'layer4.2' in name:
                # if 'layer4' in name:
                pretrain_keys.append(name)
        for key in pretrain_keys:
            key2 = list(key)
            if int(key2[7]) == 1:
                key2[7] = '{}'.format(int(key2[7]) - 1)
            # key2[7] = '{}'.format(int(key2[7]) - 1)
            key2 = ''.join(key2)
            model.state_dict()[key2].copy_(state_dict[key])
    # print(model.state_dict())
    return model


"""
Visual
"""


class DAHNET(nn.Module):
    def __init__(self, code_length=32, num_classes=17, feat_size=2048, device='cpu', pretrained=False):
        super(DAHNET, self).__init__()

        self.backbone = SEMICON_backbone(pretrained=pretrained)
        self.attention = FESM()
        self.trans = Trans(pretrained=pretrained)

        self.mcm = MCM()


        self.refine_global = SEMICON_refine(is_local=False, pretrained=pretrained)
        self.refine_local = Trans_refine(pretrained=pretrained)


        self.cls = nn.Linear(feat_size, num_classes)
        self.cls_loc = nn.Linear(feat_size, num_classes)

        self.hash_layer_active = nn.Sequential(
            nn.Tanh(),
        )
        self.code_length = code_length

        self.W_G = nn.Parameter(torch.Tensor(code_length // 4, feat_size))
        torch.nn.init.kaiming_uniform_(self.W_G, a=math.sqrt(5))

        self.W_L1 = nn.Parameter(torch.Tensor(code_length // 4, feat_size))
        torch.nn.init.kaiming_uniform_(self.W_L1, a=math.sqrt(5))
        self.W_L2 = nn.Parameter(torch.Tensor(code_length // 4, feat_size))
        torch.nn.init.kaiming_uniform_(self.W_L2, a=math.sqrt(5))
        self.W_L3 = nn.Parameter(torch.Tensor(code_length // 4, feat_size))
        torch.nn.init.kaiming_uniform_(self.W_L3, a=math.sqrt(5))

        self.bernoulli = torch.distributions.Bernoulli(0.5)
        self.device = device

    def forward(self, x):
        out = self.backbone(x)  # .detach()

        global_f = self.refine_global(out)


        feature_boost1, fms_suppress1 = self.attention(out)
        feature_boost2, fms_suppress2 = self.attention(fms_suppress1)
        feature_boost3, _ = self.attention(fms_suppress2)



        out_local1 = self.trans(feature_boost1)
        out_local2 = self.trans(feature_boost2)
        out_local3 = self.trans(feature_boost3)


        f2_from_f1, f1_from_f2 = self.mcm(out_local2, out_local1)
        f3_from_f2, f2_from_f3 = self.mcm(out_local3, out_local2)
        out_local1 = out_local1 + f1_from_f2
        out_local2 = out_local2 + 0.5 * (f2_from_f1 + f2_from_f3)
        out_local3 = out_local3 + f3_from_f2
        local_f1, avg_local_f1 = self.refine_local(out_local1)
        local_f2, avg_local_f2 = self.refine_local(out_local2)
        local_f3, avg_local_f3 = self.refine_local(out_local3)

        deep_S_G = F.linear(global_f, self.W_G)

        deep_S_1 = F.linear(avg_local_f1, self.W_L1)
        deep_S_2 = F.linear(avg_local_f2, self.W_L2)
        deep_S_3 = F.linear(avg_local_f3, self.W_L3)

        deep_S = torch.cat([deep_S_G, deep_S_1, deep_S_2, deep_S_3], dim=1)


        ret = self.hash_layer_active(deep_S)


        cls = self.cls(global_f)
        cls1 = self.cls_loc(avg_local_f1)
        cls2 = self.cls_loc(avg_local_f2)
        cls3 = self.cls_loc(avg_local_f3)


        return ret, local_f1, cls, cls1, cls2, cls3

def dahnet(code_length=12, num_classes=21, feat_size=2048, device='cpu', pretrained=False, **kwargs):

    model = DAHNET(code_length, num_classes, feat_size, device, pretrained, **kwargs)
    return model


