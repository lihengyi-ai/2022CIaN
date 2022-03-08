"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1

Code Credit: https://raw.githubusercontent.com/weiaicunzai/pytorch-cifar100/master/models/resnet.py
"""

"""
Properties:
1. Convolutions do not have bias
2. All convolutions are followed with bn
"""

import torch
import torch.nn as nn
import copy

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, btn_config=None):
        super().__init__()
        if btn_config:
            # e.g., [[64, 64, 256], 256] or [64, 64, 256]
            if isinstance(btn_config[0], list):
                channels = btn_config[0]
                transit_channel = btn_config[1]
            else:
                channels = btn_config
            out_channels = None # no use
            if channels[0] == 0 or channels[1] == 0:
#                self.residual_function = nn.Sequential()
                # dummy
                self.residual_function = nn.Sequential(
                    nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
                    nn.BatchNorm2d(1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(1, 1, stride=stride, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(1, channels[2], kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels[2]),
                )


            else:
                self.residual_function = nn.Sequential(
                    nn.Conv2d(in_channels, channels[0], kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels[0]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[0], channels[1], stride=stride, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels[1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[1], channels[2], kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels[2]),
                )

            self.shortcut = nn.Sequential()

            if stride != 1 or in_channels != channels[2]:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, transit_channel, stride=stride, kernel_size=1, bias=False),
                    nn.BatchNorm2d(transit_channel)
                )
        else:
            self.residual_function = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )

            self.shortcut = nn.Sequential()

            if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels * BottleNeck.expansion)
                )

    def forward(self, x):

        if self.residual_function[0].out_channels == 1: # dummy
            return nn.ReLU(inplace=True)(self.shortcut(x))
        else:
            return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, cfg=None):
        super().__init__()

        self.num_block = num_block
        self.config = cfg

        if self.config is None:
            self.in_channels = 64

            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
            #we use a different inputsize than the original paper
            #so conv2_x's stride is 1
            self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
            self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
            self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
            self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            assert len(num_block) + 1 == len(self.config) # first conv + 4 blocks + final linear
            for i in range(len(num_block)):
                assert num_block[i] == len(self.config[i+1])
            # config = [64, [], [], [], []]; For each internal [], we have [[[64, 64, 256], 256], [64, 64, 256], [64, 64, 256]]
            self.in_channels = self.config[0]

            self.conv1 = nn.Sequential(
                nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(inplace=True))
            #we use a different inputsize than the original paper
            #so conv2_x's stride is 1
            self.conv2_x = self._make_layer(block, self.config[1][-1][-1], num_block[0], 1, layer_id=0)
            self.conv3_x = self._make_layer(block, self.config[2][-1][-1], num_block[1], 2, layer_id=1)
            self.conv4_x = self._make_layer(block, self.config[3][-1][-1], num_block[2], 2, layer_id=2)
            self.conv5_x = self._make_layer(block, self.config[4][-1][-1], num_block[3], 2, layer_id=3)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.config[4][-1][-1], num_classes)



    def _make_layer(self, block, out_channels, num_blocks, stride, layer_id=-1):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for idx, stride in enumerate(strides):
            if self.config:
                layers.append(block(self.in_channels, out_channels, stride, btn_config=self.config[layer_id + 1][idx]))
                self.in_channels = self.config[layer_id + 1][idx][-1]
            else:
                layers.append(block(self.in_channels, out_channels, stride))
                self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def get_config(self):
        config = [-1, [[[], -1],], [[[], -1],], [[[], -1],], [[[],-1],]]
        for idx, (name, param) in enumerate(self.named_parameters()):
            if len(param.shape) == 1: # ignore bn
                continue
            channel = param.shape[0]
            if idx == 0:
                config[0] = channel
                continue

            for i in range(len(self.num_block)):
                if idx < (3 + sum(self.num_block[:(i+1)]) * 9 + 3 * (i+1)):
                    if 'shortcut' in name:
                        config[i+1][0][1] = channel
                    elif config[i+1][0][1] == -1: # haven't met shortcut yet
                        config[i+1][0][0].append(channel)
                    elif '0.weight' in name:
                        config[i+1].append([channel])
                    else:
                        config[i+1][-1].append(channel)
                    break
        return config
            
    def create_zig_params(self, opt):
        optimizer_grouped_parameters = list()

        type_4_groups_name_regs = [
            'conv1',
            'residual_function.0.weight',
            'residual_function.1.weight',
            'residual_function.1.bias',
            'residual_function.3.weight',
            'residual_function.4.weight',
            'residual_function.4.bias',
        ]
        type_1_groups = dict()
        type_4_groups = dict()
        type_4_groups2 = dict()

        type_4_groups_pointer = 0

        for i, (name, param) in enumerate(self.named_parameters()):
            print(i, name, param.shape)
            if 'fc' in name:
                if len(type_1_groups) == 0:
                    type_1_groups['params'] = list()
                    type_1_groups['shapes'] = list()
                    type_1_groups['names'] = list()
                    type_1_groups['group_type'] = 1
                type_1_groups['params'].append(param)
                type_1_groups['shapes'].append(param.shape)
                type_1_groups['names'].append(name)
            else:
                is_type_4_group = False
                for reg in type_4_groups_name_regs:
                    if reg in name:
                        if type_4_groups_pointer not in type_4_groups:
                            type_4_groups[type_4_groups_pointer] = dict()
                            type_4_groups[type_4_groups_pointer]['params'] = list()
                            type_4_groups[type_4_groups_pointer]['shapes'] = list()
                            type_4_groups[type_4_groups_pointer]['names'] = list()
                            type_4_groups[type_4_groups_pointer]['epsilon'] = opt['epsilon'][4]
                            type_4_groups[type_4_groups_pointer]['upper_group_sparsity'] = opt['upper_group_sparsity'][4]
                            type_4_groups[type_4_groups_pointer]['group_type'] = 4
                        type_4_groups[type_4_groups_pointer]['params'].append(param)
                        type_4_groups[type_4_groups_pointer]['shapes'].append(param.shape)
                        type_4_groups[type_4_groups_pointer]['names'].append(name)
                        if 'bias' in name:
                            type_4_groups_pointer += 1
                        is_type_4_group = True
                        break
                if not is_type_4_group:
                    j = int(name[4]) - 2 # conv2-5
                    if j not in type_4_groups2:
                        type_4_groups2[j] = dict()
                        type_4_groups2[j]['params'] = list()
                        type_4_groups2[j]['shapes'] = list()
                        type_4_groups2[j]['names'] = list()
                        type_4_groups2[j]['epsilon'] = opt['epsilon'][4]
                        type_4_groups2[j]['upper_group_sparsity'] = opt['upper_group_sparsity'][4]
                        type_4_groups2[j]['group_type'] = 4
                    type_4_groups2[j]['params'].append(param)  
                    type_4_groups2[j]['shapes'].append(param.shape)
                    type_4_groups2[j]['names'].append(name)  

        optimizer_grouped_parameters.append(type_1_groups)
        optimizer_grouped_parameters += list(type_4_groups.values())
        optimizer_grouped_parameters += list(type_4_groups2.values())
        
        self.optimizer_grouped_parameters = optimizer_grouped_parameters
        return optimizer_grouped_parameters

    def get_group(self, idx, group_indices=[[9, 12, 21, 30], [39, 42, 51, 60, 69], [78, 81, 90, 99, 108, 117, 126], [135, 138, 147, 156]]): # resnet50_cifar
        for g_id, group_idx in enumerate(group_indices):
            if idx in group_idx:
                group = group_idx
                return group, g_id
        return None, None

    def is_conv_weights(self, shape):
        return len(shape) == 4

    def is_linear_weights(self, shape, num_classes):
        if len(shape) == 2 and shape[0] != num_classes:
            return True
        else:
            return False


    def get_config_location(self, idx, config):
        assert idx % 3 == 0 and idx <= 156
        idx = idx // 3
        location = []
        pointer = 0
        for i1, c1 in enumerate(config):
            if isinstance(c1, list):
                for i2, c2 in enumerate(c1):
                    if isinstance(c2, list):
                        for i3, c3 in enumerate(c2):
                            if isinstance(c3, list):
                                for i4, c4 in enumerate(c3):
                                    if isinstance(c4, list):
                                        raise ValueError
                                    else:
                                        if idx == pointer:
                                            return [i1, i2, i3, i4]
                                        pointer += 1
                            else:
                                if idx == pointer:
                                    return [i1, i2, i3]
                                pointer += 1
                    else:
                        if idx == pointer:
                            return [i1, i2]
                        pointer += 1
            else:
                if idx == pointer:
                    return [i1]
                pointer += 1
        raise ValueError

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output



def resnet50(cfg=None):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], cfg=cfg)




