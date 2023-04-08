import torch
import torch.nn as nn


# 3x3 convolution
"""
Function that returns a 3x3 convolutional layer.

Args:
- in_channels: integer number representing the number of input channels.
- out_channels: integer number representing the number of output channels.
- stride: integer number representing the stride value. Default is 1.

Returns:
- A 2D convolutional layer.
"""
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
         """
        Class that defines a residual block.

        Args:
        - in_channels: integer number representing the number of input channels.
        - out_channels: integer number representing the number of output channels.
        - stride: integer number representing the stride value. Default is 1.
        - downsample: a downsampling layer. Default is None.

        Returns:
        - None.
        """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    """
    Function that defines the forward pass of a residual block.

    Args:
    - x: input tensor.

    Returns:
    - A tensor.
    """
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    """
    A Residual Neural Network (ResNet) implementation for image classification.
    Args:
        block (nn.Module): the residual block to use (e.g. BasicBlock or Bottleneck)
        num_blocks (list of int): number of residual blocks in each layer
        num_classes (int): number of classes to predict
    Inputs:
        x (torch.Tensor): a tensor of shape (batch_size, 3, 32, 32), where 3 is the number of
                           input channels, 32 is the height and width of the input image.
    Outputs:
        out (torch.Tensor): a tensor of shape (batch_size, num_classes), containing the
                            predicted class scores for each input image.
    """
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
    """
    Helper function to create a sequence of residual blocks.
    Args:
        block (nn.Module): the residual block to use (e.g. BasicBlock or Bottleneck)
        planes (int): number of output channels for each residual block
        num_blocks (int): number of residual blocks in the sequence
        stride (int): stride of the first residual block in the sequence
    Returns:
        layers (nn.Sequential): a sequence of residual blocks
    """
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    """
    Forward pass of the Resnet module.
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_planes, height, width).
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, planes, height, width).
    """
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
