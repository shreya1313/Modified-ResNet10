import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic Block module used in ResNet.

    Args:
        in_planes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for the first convolution layer. Default: 1.

    Attributes:
        expansion (int): Multiplier for the number of output channels.

        conv1 (nn.Conv2d): First convolution layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer after the first convolution layer.
        conv2 (nn.Conv2d): Second convolution layer.
        bn2 (nn.BatchNorm2d): Batch normalization layer after the second convolution layer.
        shortcut (nn.Sequential): Shortcut connection if the input/output channels or resolution is different.
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """Forward pass of the BasicBlock module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_planes, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, planes, height, width).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    
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
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
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
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    """
    Forward pass of the Resnet module.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_planes, height, width).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, planes, height, width).

    """
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
