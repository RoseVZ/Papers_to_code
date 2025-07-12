import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        """
        LeNet-5 model as described in the original paper by Yann LeCun et al.
        "Gradient-Based Learning Applied to Document Recognition"
        (https://ieeexplore.ieee.org/document/726791)
        The model consists of:
        - 2 convolutional layers with ReLU activations and max pooling
        - 3 fully connected layers with ReLU activations
        - Output layer with 10 classes (for digit recognition)
        The input is expected to be a grayscale image of size 32x32.
        The model is designed for digit classification tasks, such as MNIST.
        """
        #Layers:
        # Input Shape: [Batch, 1, 32, 32] (Grayscale image)
        #1. C1: Convolutional layer with 6 filters of size 5x5 (Input Channels: 1 (GreyScale), Output Channels: 6)
        # Output shape: [Batch, 6, 28, 28]
        self.conv1= nn.Conv2d(1,6,kernel_size=5)

        #Input Shape: [Batch, 6, 28, 28]
        #2. S2: Subsampling Layer (Average Pooling layer) with kernel size 2x2 and stride 2
        #Output shape: [Batch, 6, 14, 14]
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        #3. C3: Convolutional layer with 16 filters of size 5x5 (Input Channels: 6, Output Channels: 16)
        # Output shape: [Batch, 16, 10, 10]
        self.conv2=nn.Conv2d(6,16,kernel_size=5)

        #Input Shape: [Batch, 16, 10, 10]
        #4. S4: Subsampling Layer (Average Pooling layer) with kernel size 2x2 and stride 2
        #Output shape: [Batch, 16, 5, 5]
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        #after flattening the output of the last pooling layer, we get a tensor of shape [Batch, 16*5*5]
        #5. C5: Fully connected layer with 120 units
        #Input Shape: [Batch, 16*5*5]
        #Output shape: [Batch, 120]
        self.fc1 =nn.Linear(16*5*5,120)

        #6. F6: Fully connected layer with 84 units
        #Input Shape: [Batch, 120]
        #Output shape: [Batch, 84]
        self.fc2 = nn.Linear(120,84)

        #7. Output layer with 10 units (for digit classification)
        #Input Shape: [Batch, 84]
        #Output shape: [Batch, 10]
        self.fc3 = nn.Linear(84,10)

    
    def forward(self,x):
        """
        Forward pass of the LeNet-5 model.
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, 1, 32, 32].
        Returns:
            torch.Tensor: Output tensor of shape [Batch, 10] (class scores).
        """
        #C1 and Tanh activation
        x= F.tanh(self.conv1(x))
        #S2 
        x=self.pool1(x)

        #c3 and Tanh activation
        x=F.nn(self.conv2(x))
        #S4
        x=self.pool2(x)

        #flatten the tensor before passing to fully connected layers
        x=x.view(-1,16*5*5)

        #fc1 and Tanh activation
        x=F.tanh(self.fc1(x))
        #fc2 and Tanh activation
        x=F.tanh(self.tanh(x))
        #fc3 (output layer)
        x=self.fc3(x)

        return x



   

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16 * 5 * 5)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x