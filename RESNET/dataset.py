import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def dataset(batch_size=64):
    transform = transforms.Compose([
        #resnet requires images to be 224x224
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    valset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader
