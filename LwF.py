from LwF_net import LwF

from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from dataset import CIFAR100

import numpy as np

from sklearn.model_selection import train_test_split

import math

import utils

import copy

import torch
from torch.autograd import Variable

####Hyper-parameters####
DEVICE = 'cuda'
BATCH_SIZE = 128
CLASSES_BATCH = 10
MEMORY_SIZE = 2000
########################

def incremental_learning(num):

    torch.cuda.empty_cache()

    path='orders/'
    classes_groups, class_map, map_reverse = utils.get_class_maps_from_files(path+'classgroups'+num+'.pickle',
                                                                             path+'map'+ num +'.pickle',
                                                                             path+'revmap'+ num +'.pickle')

    net = LwF(0, class_map)
    net.to(DEVICE)

    new_acc_list = []
    old_acc_list = []
    all_acc_list = []

    for i in range(int(100/CLASSES_BATCH)):

        print('-'*30)
        print(f'**** ITERATION {i+1} ****')
        print('-'*30)

        print('Loading the Datasets ...')
        print('-'*30)

        train_dataset, val_dataset, test_dataset = utils.get_datasets(classes_groups[i])

        print('-'*30)
        print('Updating representation ...')
        print('-'*30)

        net.update_representation(dataset=train_dataset, val_dataset=val_dataset, class_map=class_map, map_reverse=map_reverse)

        net.n_known = net.n_classes

        print('Testing ...')
        print('-'*30)

        print('New classes')
        new_acc = net.classify_all(test_dataset, map_reverse)
        new_acc_list.append(new_acc)
        if i == 0:
            all_acc_list.append(new_acc)


        if i > 0:

            previous_classes = np.array([])
            for j in range(i):
                previous_classes = np.concatenate((previous_classes, classes_groups[j]))

            prev_classes_dataset, all_classes_dataset = utils.get_additional_datasets(previous_classes, np.concatenate((previous_classes, classes_groups[i])))

            print('Old classes')
            old_acc = net.classify_all(prev_classes_dataset, map_reverse)
            print('All classes')
            all_acc = net.classify_all(all_classes_dataset, map_reverse)

            old_acc_list.append(old_acc)
            all_acc_list.append(all_acc)

            print('-'*30)

    return new_acc_list, old_acc_list, all_acc_list
