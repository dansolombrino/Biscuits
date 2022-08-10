# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch.nn as nn
import torch.nn.init as init
import logging

pylogger = logging.getLogger(__name__)


def _init_layer_params(layer, init_method, init_range=None):

    # using this instead of switch-case to ensure Python < 3.10 compatibility
    if init_method.isnumeric():
        init.constant_(layer.weight, float(init_method))

    elif init_method == "he_kaiming_normal":
        init.kaiming_normal_(layer.weight)

    elif init_method == "uniform":
        if init_range is None:
            pylogger.warning(
                "Requested uniform init, but sampling range not given... "
                "defaulting to [0, 1] range"
            )

            init_range = [0, 1]

        init.uniform_(layer.weight, init_range[0], init_range[1])

    # default case
    else:
        init.kaiming_normal_(layer.weight)


def _init_layer_bias(layer, init_method, init_range=None):

    # using this instead of switch-case to ensure Python < 3.10 compatibility
    if init_method.isnumeric():
        init.constant_(layer.bias, float(init_method))

    # default case
    else:
        init.constant_(layer.bias, float(init_method))


def _freeze_layer_params(layer: nn.Module, should_freeze_parameters: bool):

    if should_freeze_parameters:
        for p in layer.parameters(recurse=False):
            p.requires_grad = False

def _count_num_parameters(param_list, trainable):
    import numpy as np

    total_params = 0

    if trainable is not None:
        param_list = filter(lambda p: p.requires_grad == trainable, param_list)

    for x in param_list:
        total_params += np.prod(torch.clone(x).cpu().data.numpy().shape)

    return total_params


def _get_num_parameters(net: nn.Module, trainable: bool) -> bool:

    param_list = net.parameters()

    return _count_num_parameters(param_list, trainable)


def get_num_trainable_parameters(net):
    return _get_num_parameters(net=net, trainable=True)


def get_num_not_trainable_parameters(net):
    return _get_num_parameters(net=net, trainable=False)


def get_num_parameters(net):
    return _get_num_parameters(net=net, trainable=None)


# does NOT distinguish between trainable and NON-trainable layers, for now
def get_num_layers(net):
    return len(
        list(
            filter(
                # lambda p: p.requires_grad and len(p.data.size()) > 1,
                lambda p: p.requires_grad and len(p.data.size()) > 1,
                net.parameters(),
            )
        )
    )


def compute_num_summary(net):
    total_params = get_num_parameters(net)
    total_trainable_params = get_num_trainable_parameters(net)
    total_not_trainable_params = get_num_not_trainable_parameters(net)
    total_layers = get_num_layers(net)

    return (
        f"Total layers                        : {total_layers}\n"
        + f"Total number of params              : {total_params}\n"
        + f"Total number of trainable params    : {total_trainable_params} --> {round(total_trainable_params/total_params * 100, 2)}%\n"
        + f"Total number of NOT trainable params: {total_not_trainable_params} --> {round(total_not_trainable_params/total_params * 100, 2)}%\n"
    )


def print_num_summary(net):
    print(compute_num_summary(net))

import collections

import torch.utils.model_zoo as model_zoo

########################################################################
############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############
########################################################################


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple("GlobalParams", [
    "block", "zero_init_residual",
    "groups", "width_per_group", "replace_stride_with_dilation",
    "norm_layer", "num_classes", "image_size"])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    
    return nn.Conv2d(
        in_planes, 
        out_planes, 
        kernel_size=3, 
        stride=stride,
        padding=dilation, 
        groups=groups, 
        bias=False, 
        dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    
    return nn.Conv2d(
        in_planes, 
        out_planes, 
        kernel_size=1, 
        stride=stride, 
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self, 
        conv_freeze_parameters: bool, 
        lin_freeze_parameters: bool,
        inplanes, 
        planes, 
        stride=1, 
        downsample=None, 
        groups=1,
        base_width=64, 
        dilation=1, 
        norm_layer=None
    ):
        super(BasicBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64"
            )
        
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        
        # Both self.conv1 and self.downsample layers downsample the input when 
        # stride != 1
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
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self, 
        conv_freeze_parameters: bool, 
        lin_freeze_parameters: bool,
        inplanes, 
        planes, 
        stride=1, 
        downsample=None, 
        groups=1,
        base_width=64, 
        dilation=1, 
        norm_layer=None
    ):
        
        super(Bottleneck, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        width = int(planes * (base_width / 64.)) * groups
        
        # Both self.conv2 and self.downsample layers downsample the input when 
        # stride != 1
        
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


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def resnet_params(model_name):
    """ Map resnet_pytorch model name to parameter coefficients. """

    params_dict = {
        # Coefficients:   block, res
        "resnet18": (BasicBlock, 224),
        "resnet34": (BasicBlock, 224),
        "resnet54": (Bottleneck, 224),
        "resnet101": (Bottleneck, 224),
        "resnet152": (Bottleneck, 224),
    }
    return params_dict[model_name]


def resnet(
    arch, 
    block, 
    num_classes=1000, 
    zero_init_residual=False,
    groups=1, 
    width_per_group=64, 
    replace_stride_with_dilation=None,
    norm_layer=None, 
    image_size=224
):
    """ Creates a resnet_pytorch model. """

    global_params = GlobalParams(
        block=block,
        num_classes=num_classes,
        zero_init_residual=zero_init_residual,
        groups=groups,
        width_per_group=width_per_group,
        replace_stride_with_dilation=replace_stride_with_dilation,
        norm_layer=norm_layer,
        image_size=image_size,
    )

    layers_dict = {
        "resnet18": (2, 2, 2, 2),
        "resnet34": (3, 4, 6, 3),
        "resnet54": (3, 4, 6, 3),
        "resnet101": (3, 4, 23, 3),
        "resnet152": (3, 8, 36, 3),
    }
    layers = layers_dict[arch]

    return layers, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith("resnet"):
        b, s = resnet_params(model_name)
        layers, global_params = resnet(arch=model_name, block=b, image_size=s)
    else:
        raise NotImplementedError(
            f"model name is not pre-defined: {model_name}"
        )
    
    if override_params:
        # ValueError will be raised here if override_params has fields not 
        # included in global_params.
        global_params = global_params._replace(**override_params)
    
    return layers, global_params


urls_map = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def load_pretrained_weights(model, model_name, load_fc):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = model_zoo.load_url(urls_map[model_name])
    
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        res = model.load_state_dict(state_dict, strict=False)
        assert set(
            res.missing_keys
        ) == {"fc.weight", "fc.bias"}, "issue loading pretrained weights"
    
    pylogger.info(f"Loaded pretrained weights for {model_name}.")


# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn

# from .utils import BasicBlock
# from .utils import Bottleneck
# from .utils import conv1x1
# from .utils import get_model_params
# from .utils import load_pretrained_weights
# from .utils import resnet_params


class ResNet(nn.Module):

    def __init__(
        self, 
        layers, 
        global_params,
        lin_init_method: bool,
        transfer_learning: bool
    ):
        super(ResNet, self).__init__()
        assert isinstance(layers, tuple), "blocks_args should be a tuple"
        assert len(layers) > 0, "layers must be greater than 0"

        self.num_classes = global_params.num_classes

        if global_params.norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        
        if global_params.replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = global_params.groups
        
        self.base_width = global_params.width_per_group
        
        # --- Begin 1st convolutional layer of Advanced ResNet --- #
        self.conv1 = nn.Conv2d(
            3, 
            self.inplanes, 
            kernel_size=7, 
            stride=2, 
            padding=3,
            bias=False
        )
        # if we are doing Transfer Learning, then freeze the parameters, as per
        # Transfer Learning way of working
        _freeze_layer_params(
            layer=self.conv1, 
            should_freeze_parameters=transfer_learning
        )

        self.bn1 = norm_layer(self.inplanes)
        _freeze_layer_params(
            layer=self.bn1,
            should_freeze_parameters=transfer_learning
        )

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # --- Begin 1st convolutional layer of Advanced ResNet --- #
        
        # --- Begin 2nd convolutional (residual) layer of Advanced ResNet --- #
        self.layer1 = self._make_layer(
            conv_freeze_parameters=transfer_learning, 
            lin_freeze_parameters=transfer_learning, 
            block=global_params.block, 
            planes=64, 
            blocks=layers[0]
        )
        
        self.layer2 = self._make_layer(
            conv_freeze_parameters=transfer_learning, 
            lin_freeze_parameters=transfer_learning,
            block=global_params.block, 
            planes=128, 
            blocks=layers[1], 
            stride=2,
            dilate=replace_stride_with_dilation[0]
        )
        
        self.layer3 = self._make_layer(
            conv_freeze_parameters=transfer_learning, 
            lin_freeze_parameters=transfer_learning,
            block=global_params.block, 
            planes=256, 
            blocks=layers[2], 
            stride=2,
            dilate=replace_stride_with_dilation[1]
        )
        
        self.layer4 = self._make_layer(
            conv_freeze_parameters=transfer_learning, 
            lin_freeze_parameters=transfer_learning,
            block=global_params.block, 
            planes=512, 
            blocks=layers[3], 
            stride=2,
            dilate=replace_stride_with_dilation[2]
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Initializing the Transfer Learning layer directly in the constructor
        # This is safe to do, since the logic to get the NN to perform Transfer 
        # Learning works as follows:
        #    Call the factory method "from_pretrained", which instantiates a 
        #    ResNet-D (D --> ResNet depth).
        #  
        #    Factory method then proceeds to load 
        #    pre-trained weights via the "load_pretrained_weights".
        # 
        #    This method allows to choose whether to load the fc layer or not.
        #    So, choosing to NOT load pre-trained weights for the final fully
        #    connected layer allows us to initialize it for Transfer Learning 
        #    directly here, with no need of performing the usual "layer 
        #    substitution". 
        self.fc = nn.Linear(
            512 * global_params.block.expansion, global_params.num_classes
        )
        if transfer_learning:
            _init_layer_params(
                layer=self.fc, 
                init_method=lin_init_method
            )
            _init_layer_bias(layer=self.fc, init_method="0")

        # This is NOT necessary anymore, since:
        # - Convolutional Residual blocks use pre-trained weights, as per Transfer
        #   Learning modus-operandi
        # - Final Fully Connected Layer is init'd in the previous code lines
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # This is out of the scope of the project AND we're loading pre-trained
        # weights, so it won't have an effect anyways...
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if global_params.zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self, 
        conv_freeze_parameters: bool,
        lin_freeze_parameters: bool,
        block, 
        planes, 
        blocks, 
        stride=1, 
        dilate=False,
    ):
        
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

        layers = [
            block(
                conv_freeze_parameters, 
                lin_freeze_parameters,
                self.inplanes, 
                planes, 
                stride, 
                downsample, 
                self.groups,
                self.base_width, 
                previous_dilation, 
                norm_layer
            )
        ]

        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(
                block(
                    conv_freeze_parameters, 
                    lin_freeze_parameters,
                    self.inplanes, 
                    planes, 
                    groups=self.groups,
                    base_width=self.base_width, 
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def prepare_for_transfer_learning(self):
        # Final layer is already set with correct input and output dimensions
        # Recreating it causes its initialization with the default method, 
        # as per PyTorch source code: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L48, 
        self.fc = nn.Linear(self.fc.in_features, self.num_classes)


    @classmethod
    def from_name(
        cls, 
        model_name, 
        override_params,
        lin_init_method,
        transfer_learning
    ):
        cls._check_model_name_is_valid(model_name)
        
        layers, global_params = get_model_params(model_name, override_params)
        
        return cls(layers, global_params, lin_init_method, transfer_learning)

    @classmethod
    def from_pretrained(
        cls, 
        model_name, 
        num_classes: int,
        lin_init_method: str,
        transfer_learning: bool
    ):
        model = cls.from_name(
            model_name, 
            override_params={
                "num_classes": num_classes
            },
            lin_init_method=lin_init_method,
            transfer_learning=transfer_learning
        )
        
        # Whether to load the weights for the fully connected layer depends on
        # whether we are doing Transfer Learning:
        # - if transfer_learning == True --> do NOT load the weights, since 
        #   we'll re-train the final layer anyways, as per Transfer Learning 
        #   modus operandi
        # - if transfer_learning == False --> load the weights, because we will
        #   NOT replace the last layer, so we need its pre-trained params
        load_pretrained_weights(
            model, 
            model_name, 
            load_fc=not transfer_learning
        )
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, res = resnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (resnet_pytorch{i} for i in 18,34,50,101,152) at the moment. """
        num_models = [18, 34, 50, 101, 152]
        valid_models = ["resnet" + str(i) for i in num_models]
        if model_name not in valid_models:
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))

if __name__ == "__main__":
    
    net = ResNet.from_pretrained(
        model_name="resnet18", 
        num_classes=2, 
        lin_init_method="he_kaiming_uniform",
        transfer_learning=True
    )
