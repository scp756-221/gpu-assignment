#!/usr/bin/env python3
'''
  Train CIFAR10 with PyTorch.
  Original code from https://github.com/kuangliu/pytorch-cifar.git

  With small modifications to refactor code, add new arguments, and make
  the file directly executable.

  This program allows you to test 15 published research models on the CIFAR10
  dataset (https://www.cs.toronto.edu/~kriz/cifar.html). We have added the
  naive model (model/naivemodel.py) from the PyTorch Tutorial
  (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)
  for comparison.  The naive model has substantially lower acccuracy than
  the research models.

  See the originating GitHub repository for links to the papers for all
  the research models.
'''

# Standard libraries
import argparse
import os
import sys
import time

# Installed libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

# Local modules
from models import *
from utils import progress_bar

# Constants
TRAIN_BATCH_SIZE=128
TEST_BATCH_SIZE=100

# Defaults for command-line options
LR_DEF=0.1
ECOUNT_DEF=10
DATA_ROOT_DEF='./data'
NET_DEF='RegNetX_200MF'
WIDTH_DEF=6

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 training example using several net designs ')
    parser.add_argument('net', default=NET_DEF, nargs='?',
            help=('Net to use; "NaiveNet" is the naive alternative. Many of '
                'the other net names are difficult to type---check the code. '
                '(Default "{}")'.format(NET_DEF)))
    parser.add_argument('--cpu', action='store_true',
            help='Use CPU even when GPU present')
    parser.add_argument('--data_root', default=DATA_ROOT_DEF,
            help='Root directory for data (default {})'.format(DATA_ROOT_DEF))
    parser.add_argument('--ecount', type=int, default=ECOUNT_DEF,
            help='Number of further epochs to run (default {})'.format(ECOUNT_DEF))
    parser.add_argument('--lr', default=LR_DEF, type=float,
            help='Learning rate (default {})'.format(LR_DEF))
    parser.add_argument('--print_net', action='store_true',
            help='Print the net structure')
    parser.add_argument('--resume', '-r', action='store_true',
            help='Resume from checkpoint. Separate checkpoints are stored for each net type.')
    parser.add_argument('--test_only', action='store_true',
            help='Test the most recent saved net of the specified type and print its accuracy for each class. Forces --ecount 1 and --resume')
    parser.add_argument('--width', type=int, default=WIDTH_DEF,
            help=('Width parameter (only for NaiveNet---'
                'default {})'.format(WIDTH_DEF)))
    args = parser.parse_args()
    if args.test_only:
        args.ecount = 1
        args.resume = True
    return args

def prepare_data(args):
    print('==> Preparing data from root directory "{}" ..'.format(args.data_root))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)

    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes

def build_model(args, device):
    print('==> Building model for {} net ..'.format(args.net))
    if args.net == 'VGG19':
        net = VGG('VGG19')
    elif args.net == 'ResNet18':
        net = ResNet18()
    elif args.net == 'PreActResNet18':
        net = PreActResNet18()
    elif args.net == 'GoogLeNet':
        net = GoogLeNet()
    elif args.net == 'DenseNet121':
        net = DenseNet121()
    elif args.net == 'ResNeXt29_2x64d':
        net = ResNeXt29_2x64d()
    elif args.net == 'MobileNet':
        net = MobileNet()
    elif args.net == 'MobileNetV2':
        net = MobileNetV2()
    elif args.net == 'DPN92':
        net = DPN92()
    elif args.net == 'ShuffleNetG2':
        net = ShuffleNetG2()
    elif args.net == 'SENet18':
        net = SENet18()
    elif args.net == 'ShuffleNetV2':
        net = ShuffleNetV2(1)
    elif args.net == 'EfficientNetB0':
        net = EfficientNetB0()
    elif args.net == 'NaiveNet':
        net = NaiveNet(args.width)
    elif args.net == 'RegNetX_200MF':
        net = RegNetX_200MF()
    else:
        print('Unsupported net type: {}'.format(args.net))
        sys.exit(1)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    start_epoch = 0
    best_acc = 0
    if args.resume:
        # Load checkpoint.
        print('==> Loading checkpoint..')
        if not os.path.isdir('checkpoint'):
            print('Error: No checkpoint directory found!')
            sys.exit(1)
        chkpt_name = './checkpoint/{}.pth'.format(args.net)
        if not os.path.isfile(chkpt_name):
            print('Error: No checkpoint saved for {} net!'.format(args.net))
            sys.exit(1)
        checkpoint = torch.load('./checkpoint/{}.pth'.format(args.net))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        print('==> Resuming from epoch {}'.format(checkpoint['epoch']))
        start_epoch = checkpoint['epoch'] + 1

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
    return net, start_epoch, best_acc, criterion, optimizer

def train(net, epoch, device, trainloader, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(net, net_name, epoch, best_acc, device, testloader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}.pth'.format(net_name))
        best_acc = acc
    return best_acc

def test_only(net, device, testloader, classes):
    """Run only the test data and print the net's accuracy for each class"""
    net.eval()
    n_classes = len(classes)
    class_correct = list(0. for i in range(n_classes))
    class_total = list(0. for i in range(n_classes))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            images, labels = inputs.to(device), targets.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(TEST_BATCH_SIZE):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            progress_bar(batch_idx, len(testloader), 'Acc: {:3.0%}'.format(
                sum(class_correct) / sum(class_total)))

    for i in range(n_classes):
        print('Accuracy of {:<10s} : {:>3.0%}'.format(
            classes[i], class_correct[i] / class_total[i]))

def main(args, device):
    trainloader, testloader, classes = prepare_data(args)
    net, start_epoch, best_acc, criterion, optimizer = build_model(args, device)
    if args.print_net:
        print('\n==> Structure of net "{}"'.format(args.net)) 
        print(net)
    if args.test_only:
        test_only(net, device, testloader, classes)
        return
    start = time.monotonic()
    epoch = -1
    try:
        for epoch in range(start_epoch, start_epoch + args.ecount):
            train(net, epoch, device, trainloader, criterion, optimizer)
            best_acc = test(net, args.net, epoch, best_acc, device, testloader, criterion)
            end = time.monotonic()
    except KeyboardInterrupt as ki:
        total_epochs = epoch - start_epoch
        print()
    else:
        total_epochs = epoch - start_epoch + 1
    if epoch != -1:
        print('Total time for {} completed epochs {:7.2f} s, per epoch '
            '{:6.2f} s'.format(total_epochs, end - start, (end-start)/total_epochs))
        print('Best test accuracy {:5.2f} %'.format(best_acc))

if __name__ == '__main__':
    args = parse_args()
    device = 'cuda' if (not args.cpu) and torch.cuda.is_available() else 'cpu'
    main(args, device)
