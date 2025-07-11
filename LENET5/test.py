import torch
import torch.nn as nn
import torch.nn.functional as F
from model import LetNet5
from dataset import dataset


def test_model(model_path='lenet5.pth'):
    #load the test dataset
    _,test_data_loader=dataset(batch_size =64)

    #model initialization
    model=LetNet5()
    model.load_state_dict(torch.load(model_path))


    #set to evaluation mode
    model.eval()

    correct=0
    total=len(test_data_loader.dataset)

    with torch.no_grad():
        for imgs,labels in test_data_loader:
            #forward pass
            outputs=model(imgs)
            
            #get the predicted class
            _,predictions= torch.max(outputs.data, 1)

            #count correct predictions
            correct+=(predictions==labels).sum().item()
    
    accuracy =100 *correct/total
    print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')

if __name__ == "__main__":
    test_model()