from tqdm import tqdm

import torch

from utils import data_loader

from resnet import ResNet, ResidualBlock

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


test_dataset = datasets.CIFAR10(root='./data', download=True, train=False, transform=t)

chech = torch.load('best_model.pt', map_location=device)

#print(chech['accuracy'])

model = ResNet(ResidualBlock, [3,4,6,3]).to(device=device)

model.load_state_dict(chech['model_state_dict'])

#print(model)

train_mode, eval_mode = get_graph_node_names(model)

#print(train_mode)

return_nodes = {
    'conv1.2':'feature_map_0'
    }

other_model = create_feature_extractor(model=model, return_nodes=return_nodes)

print(other_model)


"""
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
"""
