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
        'epochs':5
    }

    model = flatNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=train_params['lr'])
    
    train_dataloader, valid_dataloader, _ = dataloader.get_dataloaders(batch_size=train_params['batch_size'])

    for e in range(train_params['epochs']):
        print(f'\nepoch {e}')
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(valid_dataloader, model, loss_fn)
