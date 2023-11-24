import math
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models

# 9 - 166 ResNet

class Bottleneck(nn.Module):
    expansion = 4
    # stride是步长
    # rate是dilation, 卷积核的膨胀系数, 空洞卷积, 默认为1时, 即正常的卷积核, 与原来一致
    def __init__(self, inplanes, planes, stride = 1, rate = 1, downsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, dilation = rate, padding = rate, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        '''
        inplace = False 时:
        不会修改输入对象的值,而是返回一个新创建的对象,
        所以打印出对象存储地址不同,类似于C语言的值传递
        inplace = True 时：
        会改变输入数据的值,节省反复申请与释放内存的空间与时间,
        只是将原来的地址传递,效率更好
        '''
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None: # 什么时候会存在downsample?
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out       

class ResNet(nn.Module):

    def __init__(self, nInputChannels, block, layers, os =  16, pretrained = False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1] # 步长
            rates = [1, 1, 2, 2] # 空洞大小
            blocks = [1, 2, 1] # 
        else:
            raise NotImplementedError

        # Modules
        # 前两层卷积
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        # 根据Bottleneck叠加进行搭建：3, 4, 23, 4
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0]) # block, planes = 64, blocks = layers[0], stride = 1, rate = 1
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        self._init_weight() # 初始化权重

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride = 1, rate = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks = [1, 2, 4], stride = 1, rate = 1):
        downsample = None
        if stride != 1 or self.inplanes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate = blocks[0] * rate, downsample = downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride = 1, rate = blocks[i]*rate))
        
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input) # input (2, 3, 512, 512) x (2, 64, 256, 256)
        x = self.bn1(x) # (2, 64, 256, 256)
        x = self.relu(x) # (2, 64, 256, 256)
        x = self.maxpool(x) # (2, 64, 128, 128)

        x = self.layer1(x) # (2, 256, 128, 128)
        x = self.layer2(x) # (2, 512, 64, 64)
        x = self.layer3(x) # (2, 1024, 64, 64)
        x_aux = x
        x = self.layer4(x) # (2, 2048, 64, 64)

        return x, x_aux

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        # 获取预训练模型, 已下载预训练模型
        pretrain_dict = torch.load("E:/Deep Learning/Deeplab_v3plus/utils/resnet101-5d3b4d8f.pth", map_location=torch.device('cpu'))
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def ResNet101(nInputChannels = 3, os = 8, pretrained = False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 6, 3], os, pretrained = pretrained)
    # ResNet50  [3, 4,  6, 3]
    # ResNet101 [3, 4, 23, 3]
    # ResNet152 [3, 8, 36, 3]
    return model

# 167-320 MobileNetV2
BatchNorm2d = nn.BatchNorm2d

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        interverted_residual_setting = [
            # t, c, n, s
            # 473,473,3 -> 237,237,32
            # 237,237,32 -> 237,237,16
            [1, 16, 1, 1],
            # 237,237,16 -> 119,119,24
            [6, 24, 2, 2],
            # 119,119,24 -> 60,60,32
            [6, 32, 3, 2],
            # 60,60,32 -> 30,30,64
            [6, 64, 4, 2],
            # 30,30,64 -> 30,30,96
            [6, 96, 3, 1],
            # 30,30,96 -> 15,15,160
            [6, 160, 3, 2],
            # 15,15,160 -> 15,15,320
            [6, 320, 1, 1],
        ]
        
        assert input_size % 32 == 0
        # 建立stem层
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        self.features = [conv_bn(3, input_channel, 2)]
        
        # 根据上述列表进行循环，构建mobilenetv2的结构
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
                
        # mobilenetv2结构的收尾工作
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        # 最后的分类部分
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url,model_dir=model_dir)

def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNetV2(n_class=1000, **kwargs)
    if pretrained:
        model.load_state_dict(load_url('http://sceneparsing.csail.mit.edu/model/pretrained_resnet/mobilenet_v2.pth.tar'), strict=False)
    return model

class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size, norm_layer) 
                                                        for pool_size in pool_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(pool_sizes)), out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3] # features [1, 2048, 64, 64]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        # F.interpolate用于进行上下采样，根据size的大小确定进行上采样还是下采样，mode进行采样策略的选择
        # align_corners (bool, optional): 如果 align_corners=True，则对齐 input 和 output 的角点像素(corner pixels)，保持在角点像素的值. 
        # 只会对 mode=linear, bilinear 和 trilinear 有作用. 默认是 False.
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

class PSPNet(nn.Module):
    def __init__(self, nInputChannels = 3, num_classes = 6, os = 8, backbone="resnet50", pretrained=True, aux_branch=True):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        if backbone=="resnet50":
            self.backbone = ResNet101(nInputChannels, os, pretrained = pretrained)
            aux_channel = 2048
            out_channel = 1024
        elif backbone=="mobilenet":
            self.backbone = MobileNetV2(os, pretrained)
            aux_channel = 96
            out_channel = 320
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

        self.master_branch = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(out_channel//4, num_classes, kernel_size=1)
        )

        self.aux_branch = aux_branch
        self.sigmoid = nn.Sigmoid()

        if self.aux_branch:
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(aux_channel, out_channel//8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel//8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel//8, num_classes, kernel_size=1)
            )

        self.initialize_weights(self.master_branch)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x_aux, x = self.backbone(x) 

        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            return output_aux, output
        else:
            return self.sigmoid(output)

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()


if __name__ == "__main__":

    # PSPNet的CNN特征提取层为ResNet50架构
    model = PSPNet(nInputChannels = 1, num_classes = 1, os = 8, backbone="resnet50", pretrained = False, aux_branch=False)
    model.eval()
    image = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        output = model.forward(image)
    # print(output_aux.size())
    print(output.size())