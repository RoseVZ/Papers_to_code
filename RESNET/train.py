from model import LeNet5
from dataset import dataset
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

def train():
    #get the data:
    train_loader,_= dataset(batch_size=64)
    # get the model:
    model=LeNet5()
    #set the loss(multiclass) and optimizer (w lr)
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)  
    epochs = 5  # Number of epochs to train

    #training loop
    for i in range(epochs):
        model.train()  # Set the model to training mode
        loss = 0.0

        for batch_idx, (imgs,label)  in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()  

            #forward pass
            outputs = model(imgs)

            #calculate loss
            loss = criterion(outputs, label)

            #backward pass
            loss.backward()
            optimizer.step()
            loss += loss.item()
        print(f'Epoch [{i+1}/{epochs}], Loss: {loss/len(train_loader):.4f}')

    #save the model
    torch.save(model.state_dict(), 'lenet5.pth')
    print("Model saved as lenet5.pth")
    return model

if __name__=="__main__":
    train()



