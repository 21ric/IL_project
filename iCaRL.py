from iCaRL_net import iCaRL

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

    path='orders/'
    classes_groups, class_map, map_reverse = utils.get_class_maps_from_files(path+'classgroups'+num+'.pickle',
                                                                             path+'map'+ num +'.pickle',
                                                                             path+'revmap'+ num +'.pickle')
    print(classes_groups, class_map, map_reverse)

    net = iCaRL(0, class_map)

    acc_list = []


    for i in range(int(100/CLASSES_BATCH)):

        print('-'*30)
        print(f'**** Iteration {i+1} ****')
        print('-'*30)

        torch.cuda.empty_cache()

        net.compute_means = True

        print('Loading the Datasets ...')
        print('-'*30)


        train_dataset, test_dataset = utils.get_train_test(classes_groups[i])

        print('-'*30)
        print('Updating representation ...')
        print('-'*30)

        net.update_representation(dataset=train_dataset, class_map=class_map, map_reverse=map_reverse) #val_dataset=val_dataset


        print('Reducing exemplar sets ...')
        print('-'*30)

        m = MEMORY_SIZE // net.n_classes

        net.reduce_exemplars_set(m)

        print('Constructing exemplar sets ...')
        print('-'*30)

        for y in classes_groups[i]:
           net.construct_exemplars_set(train_dataset.get_class_imgs(y), m, random_flag=False)

        net.n_known = net.n_classes

        print('Testing ...')
        print('-'*30)

        print('New classes')
        acc = net.classify_all(test_dataset, map_reverse)

        if i > 0:

            previous_classes = np.array([])
            for j in range(i):
                previous_classes = np.concatenate((previous_classes, classes_groups[j]))

            prev_classes_dataset, all_classes_dataset = utils.get_additional_datasets(previous_classes, np.concatenate((previous_classes, classes_groups[i])))

            print('Old classes')
            acc = net.classify_all(prev_classes_dataset, map_reverse)
            print('All classes')
            acc = net.classify_all(all_classes_dataset, map_reverse)

            print('-'*30)

        acc_list.append(acc)

    return acc_list
