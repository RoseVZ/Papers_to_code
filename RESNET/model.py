import torch.nn as nn
import torch.nn.functional as F
import torch

class BasicBlock(nn.Module):
    #expansion in ResNet is used to adjust the number of output channels
    expand = 1
    def __init__(self,in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        #C1: 3x3 convolution layer:
        # No bias used because BatchNorm(learns γ (gamma): a learnable scale and β (beta): a learnable shift) is used
        # ouputs:`out_channels` feature maps
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)

        # Batch Normalization
        #outputs = γ * x_norm + β
        self.bn1=nn.BatchNorm2d(out_channels) 

        #C2: 3x3 convolution layer
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)

        # Batch Normalization
        self.bn2=nn.BatchNorm2d(out_channels)

        # Shortcut connection (residual path)
        # If stride != 1 (downsampling) or channel dimensions change,
        # apply 1x1 conv + BatchNorm to match shape for addition
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )   

    def forward(self, x):
        """
        Forward pass of the BasicBlock.
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, in_channels, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [Batch, out_channels, H', W'].
        """
        # C1 + BatchNorm + ReLU
        out=self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # C2 + BatchNorm
        out=self.conv2(out)
        out = self.bn2(out)

        #Add skip connection F(x) + x
        # If the dimensions match, we can directly add the input x
        # If the dimensions don't match, apply the shortcut transformation
        out += self.shortcut(x)

        #relu
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.num_classes = 10 #CIFAR 10 has 10 classes

        #C1: Initial 7x7 convolution layer 
        #BatchNorm is applied after the convolution
        #Relu activation is applied after the BatchNorm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # Fully connected layer for classification
        self.fc = nn.Linear(512 * block.expand, self.num_classes)


    def _make_layer(self, block, out_channels, blocks, stride=1):
            """ 
            Create a layer of ResNet blocks.
            Args:
                block (nn.Module): The block type to use (BasicBlock).
                out_channels (int): Number of output channels for the block.
                blocks (int): Number of blocks in the layer.
                stride (int): Stride for the first block.   
            Returns:
                nn.Sequential: A sequential container of ResNet blocks.
            """
            layers = [] 
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expand  # Update in_channels for the next block 
            for _ in range(1, blocks):
                layers.append(block(self.in_channels, out_channels))
            return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNet model.
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, 3, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [Batch, num_classes].
        """
        # Initial convolution + BatchNorm + ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Max pooling
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))

        #flatten before the fully connected layer
        x = torch.flatten(x, 1)
        # Fully connected layer
        x = self.fc(x)
        return x

def ResNet18():
    """
    Create a ResNet-18 model.
    Returns:
        ResNet: An instance of the ResNet model with 18 layers.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])





