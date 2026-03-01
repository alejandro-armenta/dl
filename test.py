from tqdm import tqdm

import torch

from utils import data_loader

from resnet import ResNet, ResidualBlock

import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_loader = data_loader('./data', batch_size=64, test=True)

chech = torch.load('best_model.pt', map_location=device)

print(chech['accuracy'])

model = ResNet(ResidualBlock, [3,4,6,3]).to(device=device)

model.load_state_dict(chech['model_state_dict'])

with torch.no_grad():
    
    correct = 0
    
    total = 0

    loader = tqdm(test_loader, desc='valid epoch')

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        del images, labels, outputs
        gc.collect()
        torch.cuda.empty_cache()
    
    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))