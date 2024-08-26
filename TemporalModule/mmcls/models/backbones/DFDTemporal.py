import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer)
from mmengine.model import caffe2_xavier_init, constant_init, kaiming_init
# from mmcv.utils.parrots_wrapper import _BatchNorm
from mmengine.utils.dl_utils.parrots_wrapper import (SyncBatchNorm, _BatchNorm,
                                                     _InstanceNorm)
from ..builder import BACKBONES
from .base_backbone import BaseBackbone

import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F
from timm.models.layers import DropPath, trunc_normal_
import math



def softmax(x):
    e_x = np.exp((x-torch.max(x)).detach().numpy())# 防溢出
    return e_x/e_x.sum(0)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop(x)
        return x


import numpy as np


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class STAUnit(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DFDUnit(nn.Module):
    def __init__(self, input_size, dim, c_dim, COF, group, qkv_bias=False):
        super().__init__()
        self.segment_dim = group
        self.segment_dim2 = group * 2
        self.norm1 = nn.LayerNorm(c_dim)

        self.mlp_h1 = nn.Linear(dim // 2, dim, bias=qkv_bias)
        self.mlp_w1 = nn.Linear(dim // 2, dim, bias=qkv_bias)

        self.COF = int(COF * dim + 1)

        self.relu = nn.ReLU()

        self.mlp_c = nn.Linear(c_dim, c_dim, bias=qkv_bias)

        self.reweight_h = nn.Linear(c_dim, c_dim, bias=qkv_bias)
        self.reweight_w = nn.Linear(c_dim, c_dim, bias=qkv_bias)

        self.param_h = nn.Parameter(torch.randn(1, input_size * group))
        self.param_w = nn.Parameter(torch.randn(1, input_size * group))

    def forward(self, x):
        B, H, W, C = x.shape
        segment_dim = self.segment_dim
        S = C // segment_dim
        h1 = x.reshape(B, H, W, segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, segment_dim * W, H * S)
        freq_h = torch.fft.rfft(h1, dim=-1)
        low_h1 = freq_h[:, :, :self.COF] # HPF
        high_h1 = freq_h[:, :, self.COF:]
        mask = abs(high_h1).ge((self.param_h).unsqueeze(-1).repeat(1,1,high_h1.size(-1))).to(torch.cfloat)
        high_h1 = high_h1 * mask
        h1_ = torch.fft.irfft(torch.cat((low_h1, high_h1), dim=-1), dim=-1)

        h1_ = F.interpolate(h1_, size=[H * S // 2])
        h1 = (h1 + self.mlp_h1(h1_)).reshape(B, segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)

        segment_dim = self.segment_dim
        S = C // segment_dim
        w1 = x.reshape(B, H, W, segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H * segment_dim, W * S)

        freq_w = torch.fft.rfft(w1, dim=-1)
        low_w1 = freq_w[:, :, :self.COF]
        high_w1 = freq_w[:, :, self.COF:]
        mask = abs(high_w1).ge((self.param_w).unsqueeze(-1).repeat(1,1,high_w1.size(-1))).to(torch.cfloat)
        high_w1 = high_w1 * mask
        w1_ = torch.fft.irfft(torch.cat((low_w1, high_w1), dim=-1), dim=-1)
        w1_ = F.interpolate(w1_, size=[W * S // 2])
        w1 = (w1 + self.mlp_w1(w1_)).reshape(B, H, segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)

        c = (self.mlp_c(x.reshape(B, H * W, C))).reshape(B, H, W, C)

        h = h1
        w = w1

        a_ = h.mean(1).reshape(B, W, C)
        b_ = w.mean(2).reshape(B, H, C)
        a = self.reweight_h(a_)
        b = self.reweight_w(b_)

        a = a.reshape(B, 1, W, C)
        b = b.reshape(B, H, 1, C)

        x = h * (b.expand_as(h)) + w * (a.expand_as(w)) + (c * (a + b))

        return x

class DFDBlock(nn.Module):
    def __init__(self, c_dim, group, COF, input_size):
        super().__init__()

        self.norm1 = nn.LayerNorm(c_dim)

        self.STAU = STAUnit(c_dim)

        c_dim = c_dim

        dim = c_dim * input_size // group

        self.group = group

        self.DFDU = DFDUnit(input_size, dim, c_dim, COF, group)

    def forward(self, x):

        x = x.permute(0, 3, 2, 1)
        B, H, W, C = x.shape
        res = x
        x = self.norm1(x)
        x = self.STAU(x.reshape(B, H * W, C), H, W).reshape(B, H, W, C) + res

        res = x
        x = self.norm1(x)
        x = self.DFDU(x)
        x = (x+res).permute(0, 3, 2, 1)

        return x


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the input/output channels of conv2. Default: 4.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): ``"pytorch"`` or ``"caffe"``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: "pytorch".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 feature_count,
                 layer_count,
                 in_channels,
                 out_channels,
                 COF,
                 num_block,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.num_block = num_block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.feature_count = feature_count

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            padding=0,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        if self.feature_count == 0:
            reduction = 4
            self.input_size = 32
            self.atten = DFDBlock(self.mid_channels, reduction, COF, self.input_size)

            reduction = 8
            self.atten1 = DFDBlock(self.mid_channels, reduction, COF, self.input_size)

        elif self.feature_count == 1:
            reduction = 4
            self.input_size = 32
            self.atten = DFDBlock(self.mid_channels, reduction, COF, self.input_size)

            reduction = 8
            self.atten1 = DFDBlock(self.mid_channels, reduction, COF, self.input_size)

        elif self.feature_count == 2:
            reduction = 4
            self.input_size = 32
            self.atten = DFDBlock(self.mid_channels, reduction, COF, self.input_size)

            reduction = 8
            self.atten1 = DFDBlock(self.mid_channels, reduction, COF, self.input_size)

        else:

            reduction = 4
            self.input_size = 16
            self.atten = DFDBlock(self.mid_channels, reduction, COF, self.input_size)

            reduction = 8
            self.atten1 = DFDBlock(self.mid_channels, reduction, COF, self.input_size)

        self.dropout = nn.Dropout(p=0.5)
        # *******************************************************************
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels * 2,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)


        out1 = self.atten(out)

        out2 = self.atten1(out)

        out = self.conv3(torch.cat((out2, out1), dim=1))
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        if self.num_block >=4:
            out = out + identity

        out = self.relu(out)
        return out



def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError('expansion is not specified for {}'.format(block.__name__))
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 feature_count,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 COF,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)
        self.feature_count = feature_count
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                feature_count=self.feature_count,
                layer_count=0,
                in_channels=in_channels,
                out_channels=out_channels,
                COF=COF,
                num_block = num_blocks,
                expansion=self.expansion,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    feature_count=self.feature_count,
                    layer_count=i,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    COF=COF,
                    num_block = num_blocks,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        #layers.append(BottleneckBlock(in_channels=in_channels, fmap_size=(16, 16), out_channels= in_channels,heads=4))
        super(ResLayer, self).__init__(*layers)


@BACKBONES.register_module()
class DFDT(BaseBackbone):

    def __init__(self,
                 COF,
                 layer_config,
                 in_channels=640,
                 stem_channels=640,
                 base_channels=160,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True):
        super(DFDT, self).__init__()
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.style = style
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block = Bottleneck
        stage_blocks = layer_config
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self._make_stem_layer(in_channels , stem_channels )
        self._make_stem_layer1(in_channels, stem_channels)
        self.in_channels = in_channels
        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]

            if num_blocks == 0:
                res_layer = nn.Identity()
            else:
                res_layer = self.make_res_layer(
                    feature_count = i,
                    block=self.block,
                    num_blocks=num_blocks,
                    in_channels=_in_channels,
                    out_channels=_out_channels,
                    COF=COF,
                    expansion=self.expansion,
                    stride=stride,
                    dilation=dilation,
                    style=self.style,
                    avg_down=self.avg_down,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg)

            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()


    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.stem = nn.Sequential(

            ConvModule(
                in_channels,
                stem_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True),
            nn.BatchNorm2d(stem_channels // 2),
            nn.ReLU(inplace=True),
            ConvModule(
                stem_channels // 2,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def _make_stem_layer1(self, in_channels, stem_channels):
        self.stem1 = nn.Sequential(
            ConvModule(
                in_channels,
                stem_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True),
            nn.BatchNorm2d(stem_channels // 2),
            nn.ReLU(inplace=True),
            ConvModule(
                stem_channels // 2,
                stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=True))

        self.Unstem1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=stem_channels//2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               output_padding=0),
            nn.BatchNorm2d(stem_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=stem_channels//2,
                               out_channels=stem_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1))

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0/fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x[-1] = self.stem(x[-1])
        x[-1] = self.maxpool(x[-1])
        x[0] = self.stem1(x[0])
        outs = []

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            outs.append(res_layer(x[i]))
        outs[0] = self.Unstem1(outs[0])
        return outs

    def train(self, mode=True):
        super(DFDT, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()