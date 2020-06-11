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

def incremental_learning(dict_num,loss_config,classifier,lr, new_herding=False, undersample=False, resize_factor=0.5):

    utils.set_seed(0)

    path='orders/'
    classes_groups, class_map, map_reverse = utils.get_class_maps_from_files(path+'classgroups'+dict_num+'.pickle',
                                                                             path+'map'+ dict_num +'.pickle',
                                                                             path+'revmap'+ dict_num +'.pickle')
    print(classes_groups, class_map, map_reverse)

    net = iCaRL(0, class_map, loss_config=loss_config,lr=lr)

    new_acc_list = []
    old_acc_list = []
    all_acc_list = []

    for i in range(int(100/CLASSES_BATCH)):

        print('-'*30)
        print(f'**** Iteration {i+1} / {int(100/CLASSES_BATCH)} ****')
        print('-'*30)

        torch.cuda.empty_cache()

        net.new_means = []
        net.compute_means = True

        print('Loading the Datasets ...')
        print('-'*30)


        train_dataset, test_dataset = utils.get_train_test(classes_groups[i])
        
        if undersample:
            train_indices, _ = train_test_split(range(len(train_dataset)), test_size=resize_factor, stratify=train_dataset.targets)
            train_dataset = Subset(train_dataset, train_indices)
            train_dataset = train_dataset.dataset

        print('-'*30)
        print(f'Known classes: {net.n_known}')
        print('-'*30)
        print('Updating representation ...')
        print('-'*30)

        net.update_representation(dataset=train_dataset, class_map=class_map, map_reverse=map_reverse, iter=i)
        
        m = MEMORY_SIZE // (net.n_classes)
        
        if i != 0:
            print('Reducing exemplar sets ...')
            print('-'*30)

            if not new_herding:
                net.reduce_exemplars_set(m)
            else:
                net.reduce_exemplars_set(m, recompute=True)

        print('len prev ex', len(net.exemplar_sets))
        print('Constructing exemplar sets ...')
        print('-'*30)

        for y in classes_groups[i]:
           net.construct_exemplars_set(train_dataset.get_class_imgs(y), m, random_flag=False)


        print('Testing ...')
        print('-'*30)

        print('New classes')
        new_acc = net.classify_all(test_dataset, map_reverse, classifier=classifier)

        new_acc_list.append(new_acc)
        if i == 0:
            all_acc_list.append(new_acc)

        if i > 0:

            previous_classes = np.array([])
            for j in range(i):
                previous_classes = np.concatenate((previous_classes, classes_groups[j]))

            prev_classes_dataset, all_classes_dataset = utils.get_additional_datasets(previous_classes, np.concatenate((previous_classes, classes_groups[i])))

            print('Old classes')
            old_acc = net.classify_all(prev_classes_dataset, map_reverse, classifier=classifier)
            print('All classes')
            all_acc = net.classify_all(all_classes_dataset, map_reverse, classifier=classifier)


            old_acc_list.append(old_acc)
            all_acc_list.append(all_acc)

            print('-'*30)

        print('lunghezza medie', len(net.exemplar_means))
        print('lunghezza nuove medie', len(net.new_means))

        net.n_known = net.n_classes
        
        if undersample:
            return

    return new_acc_list, old_acc_list, all_acc_list
