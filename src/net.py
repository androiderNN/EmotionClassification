import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt

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
    '''
    1エポック学習を行う
    returnは正解率'''
    model.train()
    size = len(dataloader.dataset)
    correct = 0

    for batch, (x, y) in enumerate(dataloader):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        size = len(dataloader.dataset)
        correct += (pred.argmax(1)==y).type(torch.float).sum().item()

    loss = loss.item()
    correct /= size
    print(f'train loss: {loss} \ntrain accuracy: {correct}')

    return correct  # accuracy

def test_loop(dataloader, model, loss_fn):
    '''
    バリデーションデータによる評価を行う'''
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
        
    test_loss /= size
    correct /= size
    print(f'test loss: {test_loss}\ntest accuracy: {correct}\n')

    return correct

def main(lr=1e-3, batch_size=10, epoch=100):
    train_params = {
        'lr': lr,
        'batch_size': batch_size,
        'epochs':epoch
    }

    train_accuracy, valid_accuracy = list(), list()

    model = simpleConvNet()
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=train_params['lr'])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_params['lr'])
    
    train_dataloader, valid_dataloader, _ = dataloader.get_dataloaders(batch_size=train_params['batch_size'])

    for e in range(train_params['epochs']):
        print(f'epoch {e}')
        tr_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
        va_acc = test_loop(valid_dataloader, model, loss_fn)

        train_accuracy.append(tr_acc)
        valid_accuracy.append(va_acc)
    
    return [train_accuracy, valid_accuracy]

if __name__=='__main__':
    log = main(epoch=200)
    plt.plot(log[0], label='train accuracy')
    plt.plot(log[1], label='valid accuracy')
    plt.show()
