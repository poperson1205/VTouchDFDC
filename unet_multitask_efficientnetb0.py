""" Full assembly of the parts to form the complete network """

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from efficientnet_pytorch import EfficientNet

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)

def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class MBConvBlockWithExpandLayer(nn.Module):
    def __init__(self, mbconv_block):
        super().__init__()
        self.mbconv_block = mbconv_block
    
    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """
        # Expansion
        x = inputs
        if self.mbconv_block._block_args.expand_ratio != 1:
            x = self.mbconv_block._swish(self.mbconv_block._bn0(self.mbconv_block._expand_conv(inputs)))
        x_expand = x

        # Depthwise Convolution
        x = self.mbconv_block._swish(self.mbconv_block._bn1(self.mbconv_block._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.mbconv_block.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self.mbconv_block._se_expand(self.mbconv_block._swish(self.mbconv_block._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self.mbconv_block._bn2(self.mbconv_block._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self.mbconv_block._block_args.input_filters, self.mbconv_block._block_args.output_filters
        if self.mbconv_block.id_skip and self.mbconv_block._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.mbconv_block.training)
            x = x + inputs  # skip connection
        return x_expand, x        

class UNetMultiTaskEfficientNetB0(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetMultiTaskEfficientNetB0, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0', num_classes=n_classes)
        self.list_efficientnet_children = list(self.efficientnet.children())
        self.inc = nn.Sequential(*self.list_efficientnet_children[:2])
        self.blocks = self.list_efficientnet_children[2]
        self.block1 = MBConvBlockWithExpandLayer(self.blocks[1])
        self.block3 = MBConvBlockWithExpandLayer(self.blocks[3])
        self.block5 = MBConvBlockWithExpandLayer(self.blocks[5])
        self.block11 = MBConvBlockWithExpandLayer(self.blocks[11])

        # Segmentation task
        self.up1 = Up(320 + 6*112, 256, bilinear)
        self.up2 = Up(256 + 6*40, 128, bilinear)
        self.up3 = Up(128 + 6*24, 64, bilinear)
        self.up4 = Up(64 + 6*16, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # Classification task
        Conv2d = get_same_padding_conv2d(image_size=224)
        self.conv_head = Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=1280, momentum=0.01, eps=0.001)
        self.swish = MemoryEfficientSwish()

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, n_classes)

    def forward(self, x):
        batch_size = x.size(0)

        x1 = self.inc(x)
        x1 = self.blocks[0](x1)
        x1_expand, x1 = self.block1(x1)

        x2 = self.blocks[2](x1)
        x2_expand, x2 = self.block3(x2)
        
        x3 = self.blocks[4](x2)
        x3_expand, x3 = self.block5(x3)

        x4 = nn.Sequential(self.blocks[6], self.blocks[7], self.blocks[8], self.blocks[9], self.blocks[10])(x3)
        x4_expand, x4 = self.block11(x4)

        x5 = nn.Sequential(self.blocks[12], self.blocks[13], self.blocks[14], self.blocks[15])(x4)

        # Segmentation task
        x = self.up1(x5, x4_expand)
        x = self.up2(x, x3_expand)
        x = self.up3(x, x2_expand)
        x = self.up4(x, x1_expand)
        logits = self.outc(x)

        # Classification task
        y = self.swish(self.bn(self.conv_head(x5)))
        y = self.avg_pooling(y)
        y = y.view(batch_size, -1)
        y = self.dropout(y)
        y = self.fc(y)

        return logits, y

    def forward_classifier(self, x):
        batch_size = x.size(0)

        x1 = self.inc(x)
        x1 = self.blocks[0](x1)
        x1_expand, x1 = self.block1(x1)

        x2 = self.blocks[2](x1)
        x2_expand, x2 = self.block3(x2)
        
        x3 = self.blocks[4](x2)
        x3_expand, x3 = self.block5(x3)

        x4 = nn.Sequential(self.blocks[6], self.blocks[7], self.blocks[8], self.blocks[9], self.blocks[10])(x3)
        x4_expand, x4 = self.block11(x4)

        x5 = nn.Sequential(self.blocks[12], self.blocks[13], self.blocks[14], self.blocks[15])(x4)

        # Classification task
        y = self.swish(self.bn(self.conv_head(x5)))
        y = self.avg_pooling(y)
        y = y.view(batch_size, -1)
        y = self.dropout(y)
        y = self.fc(y)

        return y