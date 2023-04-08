import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    """
    Residual Neural Network implementation using PyTorch.

    Args:
        block (nn.Module): Block module to use in the network.
        num_blocks (list of int): List of integers that specifies the number of blocks in each layer.
        num_classes (int): Number of classes for the classification task.

    Attributes:
        conv1 (nn.Conv2d): 2D convolutional layer with 3 input channels and 64 output channels.
        bn1 (nn.BatchNorm2d): Batch normalization layer after the first convolutional layer.
        layer1 (nn.Sequential): Sequence of block layers in the first stage of the network.
        layer2 (nn.Sequential): Sequence of block layers in the second stage of the network.
        layer3 (nn.Sequential): Sequence of block layers in the third stage of the network.
        layer4 (nn.Sequential): Sequence of block layers in the fourth stage of the network.
        linear (nn.Linear): Fully connected linear layer for classification.

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain = nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias.data, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Helper function to create block layers of the ResNet.

        Args:
            block (nn.Module): Block module to use in the network.
            planes (int): Number of output channels for each block.
            num_blocks (int): Number of blocks in the layer.
            stride (int): Stride value for the first block.

        Returns:
            nn.Sequential: Sequential of block layers for the given layer.

        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNet.

        Args:
            x (torch.Tensor): Input tensor of size [batch_size, 3, 32, 32].

        Returns:
            torch.Tensor: Output tensor of size [batch_size, num_classes].

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

class BasicBlock(nn.Module):
    """
    A basic residual block for ResNet.

    Args:
        in_planes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Stride of the first convolutional layer. Default is 1.

    Attributes:
        expansion (int): The multiplier for the number of output channels.

    Returns:
        torch.Tensor: The output tensor with the shape of (batch_size, planes, H, W).

    Examples:
        >>> block = BasicBlock(64, 128, stride=2)
        >>> x = torch.randn(32, 64, 32, 32)
        >>> out = block(x)
        >>> out.shape
        torch.Size([32, 128, 16, 16])
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
        """
        The forward pass of a basic residual block.

        Args:
            x (torch.Tensor): The input tensor with the shape of (batch_size, in_planes, H, W).

        Returns:
            torch.Tensor: The output tensor with the shape of (batch_size, planes, H, W).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
