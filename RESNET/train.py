import torch
import torch.nn as nn
import torch.optim as optim
from model import ResNet18 
from dataset import dataset


def train( batch_size=64):
    model=ResNet18()
    train_loader, val_loader = dataset(batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 5  
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images, labels

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}")

    # Evaluate after training
    evaluate(model, val_loader)

    # Save model
    model_name = 'resnet18.pth'
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}")

    return model

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train()
