from tqdm import tqdm

import numpy as np

import torch

import torch.nn as nn

from torchvision import datasets

#transforms the data images 
from torchvision import transforms

from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader

from resnet import ResNet, ResidualBlock

import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#del images, labels, outputs
#del model
#del optimizer
torch.cuda.empty_cache()
gc.collect()

print(device)

def data_loader(data_dir, batch_size, test=False, shuffle=True, valid_size=0.1):

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])

    t = transforms.Compose(
        [
            #este si puede ser pil
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            normalize
        ]

    )

    if test:
        test_dataset = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=t)
        #print(test_dataset)
        data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    train_dataset = datasets.CIFAR10(root=data_dir, download=True, train=True, transform=t)
    valid_dataset = datasets.CIFAR10(root=data_dir, download=True, train=True, transform=t)

    num_train = len(train_dataset)

    indices = list(range(num_train))

    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_indices = indices[split:]
    val_indices = indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    #print(train_dataset is valid_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, valid_loader


train_loader, valid_loader = data_loader('./data', batch_size=64)
test_loader = data_loader('./data', batch_size=64, test=True)

model = ResNet(ResidualBlock, [3,4,6,3]).to(device=device)

criterion = nn.CrossEntropyLoss()

num_epochs = 20
learnin_rate = 0.01

#ay un weight decay en sgd
optimizer = torch.optim.SGD(model.parameters(), lr=learnin_rate, weight_decay=0.001, momentum=0.9)

print(len(train_loader) * num_epochs)

best_accuracy = 0

for epoch in range(num_epochs):

    loader = tqdm(train_loader, desc='training epoch')
    
    
    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)

        #print(pred.shape)
        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loader.set_postfix(loss=f"{loss.item():.4f}")

        del images, labels, pred
        gc.collect()
        torch.cuda.empty_cache()
    

    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    

    with torch.no_grad():
        
        correct = 0
        total = 0

        loader = tqdm(valid_loader, desc='valid epoch')

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            #print(outputs.shape)

            _, predicted = torch.max(outputs.data, 1)
            
            #print("predicted shape", predicted.shape)
            #print("labels shape", labels.shape)

            total += labels.size(0)
            
            correct += (predicted == labels).sum().item()

            del images, labels, outputs
            gc.collect()
            torch.cuda.empty_cache()
        
        accuracy = 100 * correct / total
        print('Accuracy of the network on the {} validation images: {} %'.format(5000, accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'accuracy':accuracy
            }, f='best_model.pt')


        