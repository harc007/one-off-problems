import torch
import torchvision
import torchvision.transforms as transforms
import traceback
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def transform_images_to_tensors():
    try:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5), (0.5, 0.5))])
        trainset = torchvision.datasets.FashionMNIST(root='./fashion_data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./fashion_data/', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
        classes = ("tshirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankleboot")
        return trainloader, testloader, classes
    except Exception as e:
        print(traceback.format_exc())
        raise e

def imshow(img):
    img = img/2 + 0.5
    npimg = np.transpose(img.numpy())
    print(npimg.shape)
    cv2.namedWindow("w1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("w1", 100, 100)
    cv2.imshow("w1", npimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def showimages(trainloader):
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_optimizer(net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    return criterion, optimizer

def train(trainloader, net, criterion, optimizer, is_gpu, n_epochs=2):
    try:
        for epoch in range(n_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                if is_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                    print(1, inputs.shape, labels.shape)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]
                if i%2000 == 1999:
                    print("[%d, %5d] loss: %3f" % (epoch+1, i+1, running_loss/2000))
                    running_loss = 0.0
        return outputs
    except Exception as e:
        print(traceback.format_exc())
        raise e

def test_single(testloader, net):
    try:
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        imshow(torchvision.utils.make_grid(images))
        print("Ground truth: ", " ".join('%5s' % classes[labels[j]] for j in range(4)))
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    except Exception as e:
        print(traceback.format_exc())
        raise e


def test_multiple(testloader, net, is_gpu):
    try:
        correct = 0
        total = 0
        for data in testloader:
            test_images, labels = data
            if is_gpu:
                outputs = net(Variable(test_images.cuda()))
                _, predicted = torch.max(outputs.cuda().data, 1)
            else:
                outputs = net(Variable(test_images))

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        return correct, total
    except Exception as e:
        print(traceback.format_exc())
        raise e

if __name__=='__main__':
    is_gpu = False
    trainloader, testloader, classes = transform_images_to_tensors()
    showimages(trainloader)
    net = Net()
    print("Done1")
    if is_gpu:
        criterion, optimizer = get_optimizer(net.cuda())
    else:
        criterion, optimizer = get_optimizer(net)
    print("Done2")
    outputs = train(trainloader, net, criterion, optimizer, is_gpu, 2)
    print("Finished training")
    correct, total = test_multiple(testloader, net, is_gpu)
    print("accuracy of network on the 1000 test images: %d %%" % (100*correct/total))
