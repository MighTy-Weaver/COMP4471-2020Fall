import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

print(torch.cuda.is_available())

transform = transforms.Compose([transforms.ToTensor()])
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          transform=transform)
print(cifar_train)
print(cifar_test)

trainloader = torch.utils.data.DataLoader(cifar_train, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(cifar_test, batch_size=32, shuffle=True)


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # Input channels, Output channels, kernel size
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Output channels in conv2 * kernel size 1 * kernel size 2
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # We have ten classes in total
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)

    # Function for forward propagation
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # Dimension reduction
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda:0")
cnn_net = NeuralNet().to(device)
evaluation = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_net.parameters(), lr=1e-3, momentum=0.8)

print("Training Started------------------------")

for epoch in tqdm(range(100)):
    loss_per_hundred = 0.0
    for i, data in enumerate(trainloader):
        # Read the inputs and labels of the dataset
        inputs, labels = data
        # Push the data to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Clear the gradient so each time the gradient starts from 0
        optimizer.zero_grad()
        # Calculate the output of the forward propagation
        outputs = cnn_net(inputs)
        # Compute the loss between the output of FP and labels
        epoch_loss = evaluation(outputs, labels)
        # Use the loss to do the back propagation
        epoch_loss.backward()
        # Optimize the CNN by the gradient calculated
        optimizer.step()
        # Add the loss to the loss per hundred
        loss_per_hundred += epoch_loss.item()
        # Report the training loss once per hundred batch
        if i % 100 == 99:
            print("Epoch {}, Batch {} loss: {}".format(epoch + 1, i + 1, loss_per_hundred / 100))
            loss_per_hundred = 0.0

print("Training Done---------------------------")

# Construct the testing dataloader
test_data = iter(testloader)
# Store the correct classifications among all the classifications
correct, total = 0, 0
# No need to calculate gradient in the forward propagation when we do the testing
with torch.no_grad():
    for data in testloader:
        # Read the inputs and labels of the dataset
        inputs, labels = data
        # Push the data to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = cnn_net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
