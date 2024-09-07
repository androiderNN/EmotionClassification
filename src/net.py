import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

import config, dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class flatNet(nn.Module):
    def __init__(self):
        super(flatNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(128*120, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        y = self.linear_stack(x)
        return y

class simpleConvNet(nn.Module):
    def __init__(self):
        super(simpleConvNet, self).__init__()
        
        channels = {
            'channel1': 16,
            'channel2': 32,
            'channel3': 64
        }

        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, channels['channel1'], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels['channel1']),
            nn.ReLU(),
            nn.Conv2d(channels['channel1'], channels['channel2'], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels['channel2']),
            nn.ReLU(),
            nn.Conv2d(channels['channel2'], channels['channel3'], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels['channel3']),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(int((channels['channel3']*128*120)/(8**2)), 4)
        )
    
    def forward(self, x):
        return self.conv_stack(x)

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%10 == 0:
            loss, current = loss.item(), batch*len(x)
            print(f'loss: {loss} [{current}/{size}]')

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
        
    test_loss /= size
    correct /= size
    print(f'test error: {test_loss}\naccuracy: {correct}\n')

if __name__=='__main__':
    train_params = {
        'lr': 1e-2,
        'batch_size': 100,
        'epochs':10
    }

    model = simpleConvNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=train_params['lr'])
    
    train_dataloader, valid_dataloader, _ = dataloader.get_dataloaders(batch_size=train_params['batch_size'])

    for e in range(train_params['epochs']):
        print(f'epoch {e}')
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(valid_dataloader, model, loss_fn)
