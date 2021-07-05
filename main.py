from continualai.colab.scripts import mnist
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import os
import copy

from net import Net
from train import train, test, train_ewc, on_task_update
from permute_mnist import permute_mnist

def make_cuda_device():

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda')
    return device

def make_datasets():
    X_train, y_train, X_test, y_test = mnist.load()

    print('---------------------- mnist ----------------------')
    print("X_train dim and type: ", X_train.shape, X_train.dtype)
    print("y_train dim and type: ", y_train.shape, y_train.dtype)
    print("X_test dim and type: ", X_test.shape, X_test.dtype)
    print("y_test dim and type: ", y_test.shape, y_test.dtype)
    print('---------------------------------------------------')

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    DEVICE = make_cuda_device()

    mnist.init()
    X_train, y_train, X_test, y_test = make_datasets()

    model_normal = Net().to(DEVICE)
    model_ewc = copy.deepcopy(model_normal)

    # optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
    optimizer_normal = optim.Adam(model_normal.parameters(), lr = 0.001)
    optimizer_ewc = optim.Adam(model_ewc.parameters(), lr = 0.001)

    # print(model)

    # Basic training - 98%
    # for epoch in range(1, 11):
    #     train(model, DEVICE, X_train, y_train, optimizer, epoch)
    #     test(model, DEVICE, X_test, y_test)

    # X_train2, X_test2 = permute_mnist([X_train, X_test], 0)

    # print('Testing on the first task : ')
    # test(model, DEVICE, X_test, y_test)
    # print('Testing on the second task : ')
    # test(model, DEVICE, X_test2, y_test)




    # for continual learning
    # task1 
    task_1 = [(X_train, y_train), (X_test, y_test)]
    # task2
    X_train2, X_test2 = permute_mnist([X_train, X_test], 1)
    task_2 = [(X_train2, y_train), (X_test2, y_test)]
    # task3
    X_train3, X_test3 = permute_mnist([X_train, X_test], 2)
    task_3 = [(X_train3, y_train), (X_test3, y_test)]

    tasks = [task_1, task_2, task_3]

    # Elastic Weights Consolidation (EWC)
    fisher_dict = {}
    optpar_dict = {}
    ewc_lambda = 0.2
    ewc_accs = []

    avg_acc = 0
    avg_normal_acc = 0

    for id, task in enumerate(tasks):

        print('Training on task : ', id)

        (X_train, y_train), _ = task

        for epoch in range(1, 11):

            train(model_normal, DEVICE, X_train, y_train, optimizer_normal, epoch)

            fisher_dict, optpar_dict = train_ewc(model_ewc, DEVICE, fisher_dict, optpar_dict, ewc_lambda, id, X_train, y_train, optimizer_ewc, epoch)
        fisher_dict, optpar_dict = on_task_update(model_ewc, DEVICE, optimizer_ewc, fisher_dict, optpar_dict, id, X_train, y_train)

    for id_test, task in enumerate(tasks):
        print('Testing on task : ', id_test)
        _, (X_test, y_test) = task

        acc_normal = test(model_normal, DEVICE, X_test, y_test)
        acc = test(model_ewc, DEVICE, X_test, y_test)

        avg_normal_acc += acc_normal
        avg_acc = avg_acc + acc

    print('Avg normal acc : ', avg_normal_acc / 3)
    print('Avg acc : ', avg_acc / 3)
    ewc_accs.append(avg_acc / 3)