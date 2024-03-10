from einops import rearrange
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from tqdm.notebook import tqdm
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from VitModel import ViT
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MNISTDataset(Dataset):
    def __init__(self, data_df:pd.DataFrame, transform=None, is_test=False):
        super(MNISTDataset, self).__init__()
        dataset = []
            
        for i, row in tqdm(data_df.iterrows(), total=len(data_df)):
            data = row.to_numpy()
            if is_test:
                label = -1
                image = data.reshape(28, 28).astype(np.uint8)
            else:
                label = data[0]
                image = data[1:].reshape(28, 28).astype(np.uint8)
            
            if transform is not None:
                image = transform(image)
                    
            dataset.append((image, label))
        self.dataset = dataset
        self.transform = transform
        self.is_test = is_test
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        return self.dataset[i]
    
def train_epoch(model, optimizer, data_loader, loss_history):
        total_samples = len(data_loader.dataset)
        model.train()

        for i, (data, target) in enumerate(data_loader):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            optimizer.zero_grad()
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                    ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                    '{:6.4f}'.format(loss.item()))
                loss_history.append(loss.item())

def evaluate(model, data_loader, loss_history):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
        '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
        '{:5}'.format(total_samples) + ' (' +
        '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

if __name__ == '__main__':

    #transform = ToTensor()
    train_transform = transform=transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    val_transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = MNIST(root='./../datasets', train=True, download=True, transform=train_transform)
    test_set = MNIST(root='./../datasets', train=False, download=True, transform=val_transform)
    
    BATCH_SIZE_TRAIN = 100
    BATCH_SIZE_TEST = 1000
    train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE_TRAIN)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE_TEST)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=6, heads=8, mlp_dim=128)
    model = model.to(DEVICE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    train_loss_history, test_loss_history = [], []

    N_EPOCHS = 100

    # lambda1 = lambda epoch: epoch // 30
    # lambda2 = lambda epoch: 0.95 ** epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    start_time = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch,'LR:', scheduler.get_last_lr())
        train_epoch(model, optimizer, train_loader, train_loss_history)
        evaluate(model, test_loader, test_loss_history)
        scheduler.step()

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
