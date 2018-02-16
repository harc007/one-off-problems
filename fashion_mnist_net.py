import torch
import torchvision
import torchvision.transforms as transforms
import traceback
import numpy as np
import cv2
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import argparse
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

writer = SummaryWriter('board')

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

        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        nn.init.xavier_uniform(self.conv1.weight)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        nn.init.xavier_uniform(self.conv2.weight)

        self.fc1 = nn.Linear(32*7*7, 160)
        self.fc2 = nn.Linear(160, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        #x = F.relu(self.conv3(x))
        x = x.view(-1, 32*7*7)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_optimizer(net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    return criterion, optimizer

def train(trainloader, net, criterion, optimizer, is_gpu, n_epochs=2, is_accuracy_comparison=True):
    try:
        images_seen = 0
        for epoch in range(n_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                if is_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]
                if i%2000 == 0:
                    imgs = vutils.make_grid(inputs.cpu().data, normalize=True, scale_each=True)
                    writer.add_image('f_image', imgs, i+1+images_seen)
                    print("[%d, %5d] loss: %3f" % (epoch+1, i+1+images_seen, running_loss/2000))
                    writer.add_scalar('f_data/running_loss', running_loss/2000.0, i+1+images_seen)
                    running_loss = 0.0
                    if is_accuracy_comparison:
                        test_correct, test_total = test_multiple(testloader, net, is_gpu)
                        test_accuracy = 100*test_correct/test_total
                        train_correct, train_total = test_multiple(trainloader, net, is_gpu)
                        train_accuracy = 100*train_correct/train_total
                        writer.add_scalar('f_data/train_accuracy', train_accuracy, i+1+images_seen)
                        writer.add_scalar('f_data/test_accuracy', test_accuracy, i+1+images_seen)
                    images_seen += i
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
                test_images = test_images.cuda()
                labels = labels.cuda()
              
            outputs = net(Variable(test_images))

            if is_gpu:
                outputs = outputs.cuda()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        return correct, total
    except Exception as e:
        print(traceback.format_exc())
        raise e

def save_model(net, path='fashion_mnist_01'):
    try:
        torch.save(net.state_dict(), path)
        return True
    except Exception as e:
        print(traceback.format_exc())
        return False

def load_model(is_gpu, path='fashion_mnist_01'):
    try:
        net = Net()
        net.load_state_dict(torch.load(path))
        if is_gpu:
            net = net.cuda()
        return net
    except Exception as e:
        print(traceback.format_exc())
        raise e


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="fashion mnist")
    parser.add_argument('-p', '--path', default="fashion_mnist_01")
    parser.add_argument('-e', '--epochs', default=2, type=int)
    args = parser.parse_args()
    path_to_file = args.path
    n_epochs = args.epochs
    trainloader, testloader, classes = transform_images_to_tensors()
    is_gpu = torch.cuda.is_available()
    if not os.path.isfile(path_to_file):
        showimages(trainloader)
        net = Net()
        if is_gpu:
            net = net.cuda()
        criterion, optimizer = get_optimizer(net)
        print("Training....")
        outputs = train(trainloader, net, criterion, optimizer, is_gpu, n_epochs)
        print("Finished training")
        save_model(net, path_to_file)
        print("Saved file in location {0}".format(path_to_file))
    print("Loading model {0}".format(path_to_file))
    net = load_model(is_gpu, path_to_file)
    print("Testing...") 
    correct, total = test_multiple(testloader, net, is_gpu)
    print("accuracy of network on the 1000 test images: %d %%" % (100*correct/total))
