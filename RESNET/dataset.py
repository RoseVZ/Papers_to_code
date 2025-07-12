# experiments/lenet/dataset.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def dataset(batch_size=64):
    transform = transforms.Compose([
        transforms.Pad(2),             
        transforms.ToTensor()
    ])
    train = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size, shuffle=False)
    )


