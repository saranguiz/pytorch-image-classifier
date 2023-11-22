import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import argparse
import os
import logging
import sys
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion):
    model.eval()
    running_loss=0
    running_corrects=0
    pos=0
    for inputs, labels in test_loader:
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        pos+=1
    test_loss = running_loss / len(test_loader.dataset)
    print(
        "\nTesting loss: {:.4f} \t Testing Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, running_corrects, len(test_loader.dataset), 100.0 * running_corrects / len(test_loader.dataset)
        )
    )

def train(model, epochs, train_loader, validation_loader, criterion, optimizer):
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for pos,(inputs, labels) in enumerate(train_loader):
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        model.eval()
        valid_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            running_loss = 0.0
            for pos,(inputs, labels) in enumerate(validation_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        valid_loss = running_loss / len(validation_loader.dataset)
        valid_acc = running_corrects / len(validation_loader.dataset)
        model_saved=False
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), 'model.pth')
            model_saved=True
        print(
            "EPOCH {}: \t Training Loss: {:.4f} \t Training Accuracy: {:.2f}% \t Validation Loss: {:.4f} \t Validation Accuracy: {}/{} ({:.0f}%) \t Model saved? {}".format(
                epoch+1, epoch_loss, 100*epoch_acc, valid_loss, running_corrects, len(validation_loader.dataset), 100*valid_acc, model_saved
            )
        )
    return model
    
def net():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    num_features = 5
    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, num_features))
    return model

def create_data_loaders(data, batch_size):
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')

    img_size = 224
    train_transform = transforms.Compose([
        #transforms.RandomResizedCrop((224, 224)),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    tic = time.time()
    train_loader, test_loader, validation_loader = create_data_loaders(args.data_dir, args.batch_size)
    model = net() # Initialize a model by calling the net function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    logger.info(f'Starting Model Training')
    model=train(model, args.epochs, train_loader, validation_loader, criterion, optimizer)

    model.load_state_dict(torch.load('model.pth'))
    test(model, test_loader, criterion)

    elapsed_time = time.time() - tic
    logger.info(f'Finished! Elapsed time: {round(float(elapsed_time),2)} secs.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=str, help="directory with data for training, validation and testing"
        )
    parser.add_argument(
        "--batch_size", type=int, default=5, metavar="N", help="input batch size for training, validation and testing (default: 5)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    args=parser.parse_args()    
    main(args)
