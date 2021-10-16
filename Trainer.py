# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from DataLoader import getDataset, Trainset


##################################################
# define Net class
##################################################
class Net(nn.Module):
    # This is a simple train net, you can use resnet18 or more deep net
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=7)
        self.conv3_drop = nn.Dropout2d()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(6400, 1000)
        self.fc2 = nn.Linear(1000, 48)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(x)), 2))
        x = x.view(-1, 64 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


##################################################
# function train
##################################################
def train(args, model, device, trainloader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = target.long()
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))


##################################################
# function test
##################################################
def test(model, device, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = target.long()
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))


##################################################
# train process
##################################################
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DL Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=300, metavar='N',
                        help='how many batches to wait before logging training status')
    # default setting to train
    args = parser.parse_args(['--epochs', '10', '--log-interval', '100'])

    # path = r'D:\image_file' #change you own dataset path
    path = r'D:\T'
    x, y, label_dict = getDataset(path)
    # 80% of data to trian left 20% to test
    train_data = x[:int(len(x) * 0.8)]
    test_data = x[int(len(x) * 0.8):]
    train_label = y[:int(len(x) * 0.8)]
    test_label = y[int(len(x) * 0.8):]
    # load data
    train_data = Trainset(train_data, train_label)
    test_data = Trainset(test_data, test_label)
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=100, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # start train epoch
    for epoch in range(1, args.epochs + 1):
        # train
        train(args, model, device, trainloader, optimizer, epoch)
        # test
        test(model, device, testloader)

    # save model in output path
    torch.save(model.state_dict(), 'output/mymodel.pth')
