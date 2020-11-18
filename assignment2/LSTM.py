import argparse

import torch
import torchvision
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100, help="The batch size of the LSTM")
parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate of the LSTM")
parser.add_argument('--epoch', type=int, default=20, help="number of epoch to be trained")
args = parser.parse_args()

BATCH_SIZE = args.batch_size
LR = args.learning_rate
EPOCH = args.epoch

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

trainsets = torchvision.datasets.CIFAR10(root='./cifar10/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainsets, batch_size=BATCH_SIZE, shuffle=True)

testsets = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testsets, batch_size=BATCH_SIZE, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.LSTM = nn.LSTM(32 * 3, 128, batch_first=True, num_layers=3)  # 将彩色图片输入给LSTM怎么办
        self.line = nn.Linear(128, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        out, (h_n, c_n) = self.LSTM(x)
        out = self.line(out[:, -1, :])
        return self.output(out)


if __name__ == '__main__':
    net = Net()
    print('parameters:', sum(param.numel() for param in net.parameters()))  # 计算网络需要计算的参数量
    net.cuda()
    Loss = nn.CrossEntropyLoss()
    Opt = optim.Adam(net.parameters(), lr=LR)
    for epoch in range(EPOCH):
        for step, (data, target) in enumerate(trainloader):
            data = Variable(data)
            target = Variable(target)
            data = data.view(-1, 32, 32 * 3)
            data = data.cuda()
            target = target.cuda()
            out = net(data)
            loss = Loss(out, target)
            Opt.zero_grad()
            loss.backward()
            Opt.step()
            total_accu, total_num = 0, 0
            if step % 50 == 0:
                for d in testloader:
                    test_x, test_y = d
                    test_x, test_y = test_x.cuda().data, test_y.cuda().data
                    test_x = test_x.view(-1, 32, 32 * 3)
                    test_out = net(test_x)
                    pred_y = torch.max(test_out, 1)[1].cuda().data
                    accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
                    total_accu += accuracy
                    total_num += 1
                print("Epoch: {} | number of batches: {} | train loss: {} | test accuracy: {}".format(epoch, step,
                                                                                                      loss.data.cpu().numpy(),
                                                                                                      total_accu / total_num))
