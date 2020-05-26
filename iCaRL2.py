from iCaRL_net2 import iCaRL

from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from dataset import CIFAR100

import numpy as np

from sklearn.model_selection import train_test_split

import math

import copy

import torch
from torch.autograd import Variable

####Hyper-parameters####
DEVICE = 'cuda'
BATCH_SIZE = 128
CLASSES_BATCH = 10
MEMORY_SIZE = 2000
########################

def main():

    #define images transformation
    #define images transformation
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                        #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                       ])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                       #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                      ])

    #creo i dataset per ora prendo solo le prime 10 classi per testare, ho esteso la classe cifar 100 con attributo
    #classes che è una lista di labels, il dataset carica solo le foto con quelle labels


    total_classes = 100

    perm_id = np.random.permutation(total_classes)
    all_classes = np.arange(total_classes)

    #mix the classes indexes
    for i in range(len(all_classes)):
      all_classes[i] = perm_id[all_classes[i]]

    #Create groups of 10
    classes_groups = np.array_split(all_classes, 10)
    #print(classes_groups)

    # Create class map
    class_map = {}
    #takes 10 new classes randomly
    for i, cl in enumerate(all_classes):
        class_map[cl] = i
    #print (f"Class map:{class_map}\n")

    # Create class map reversed
    map_reverse = {}
    for cl, map_cl in class_map.items():
        map_reverse[map_cl] = int(cl)
    #print (f"Map Reverse:{map_reverse}\n")


    #range_classes = np.arange(100)
    #classes_groups = np.array_split(range_classes, 10)


    net = iCaRL(0, class_map)

    for i in range(int(100/CLASSES_BATCH)):

        print('Loading the Datasets ...')
        print('-'*30)

        train_dataset = CIFAR100(root='data/', classes=classes_groups[i], train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(root='data/', classes=classes_groups[i],  train=False, download=True, transform=test_transform)

        train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.1, stratify=train_dataset.targets)

        val_dataset = Subset(copy.deepcopy(train_dataset), val_indices)
        train_dataset = Subset(train_dataset, train_indices)

        val_dataset.dataset.transform = test_transform

        print('-'*30)
        print('Updating representation ...')
        print('-'*30)

        net.update_representation(dataset=train_dataset, val_dataset=val_dataset, class_map=class_map, map_reverse=map_reverse)


        #print('Dato train sample')
        #print(train_dataset.data[:1])
        #print('tipo')
        #print(type(train_dataset.data[:1]))
        print('Reducing exemplar sets ...')
        print('-'*30)

        m = int(math.ceil(MEMORY_SIZE/net.n_classes))

        net.reduce_exemplars_set(m)

        print('Constructing exemplar sets ...')
        print('-'*30)

        for y in classes_groups[i]:
            net.construct_exemplars_set(train_dataset.dataset.get_class_imgs(y), m)

        net.n_known = net.n_classes

        print('Testing ...')
        print('-'*30)

        if i == 0:

            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

            running_corrects = 0

            for imgs, labels, _ in  test_dataloader:
                imgs = Variable(imgs).cuda()
                preds = net.classify(imgs, compute_means=True)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                running_corrects += (preds == labels.numpy()).sum()
                #running_corrects += torch.sum(preds == labels.data).data.item()
            accuracy = running_corrects / float(len(test_dataloader.dataset))
            print('Test Accuracy: {}'.format(accuracy))
        else:
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


            previous_classes = np.array([])
            for j in range(i):
              previous_classes = np.concatenate((previous_classes, classes_groups[j]))
            test_prev_dataset = CIFAR100(root='data/', classes=previous_classes,  train=False, download=True, transform=test_transform)

            # Creating dataset for all classes
            all_classes = np.concatenate((previous_classes, classes_groups[i]))
            test_all_dataset = CIFAR100(root='data/', classes=all_classes,  train=False, download=True, transform=test_transform)

            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            all_dataloader = DataLoader(test_all_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            prev_dataloader = DataLoader(test_prev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

            running_corrects = 0

            for imgs, labels, _ in  test_dataloader:
                imgs = Variable(imgs).cuda()
                preds = net.classify(imgs, compute_means = True)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                running_corrects += (preds == labels.numpy()).sum()
                #running_corrects += torch.sum(preds == labels.data).data.item()
            accuracy = running_corrects / float(len(test_dataloader.dataset))
            print('Test Accuracy: {}'.format(accuracy))

            running_corrects = 0

            for imgs, labels, _ in  prev_dataloader:
                imgs = Variable(imgs).cuda()
                preds = net.classify(imgs, compute_means = False)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                running_corrects += (preds == labels.numpy()).sum()
                #running_corrects += torch.sum(preds == labels.data).data.item()
            accuracy = running_corrects / float(len(prev_dataloader.dataset))
            print('Test Accuracy old classes: {}'.format(accuracy))

            running_corrects = 0

            for imgs, labels, _ in  all_dataloader:
                imgs = Variable(imgs).cuda()
                preds = net.classify(imgs, compute_means = False)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                running_corrects += (preds == labels.numpy()).sum()
                #running_corrects += torch.sum(preds == labels.data).data.item()
            accuracy = running_corrects / float(len(all_dataloader.dataset))
            print('Test Accuracy all classes: {}'.format(accuracy))

            print('-'*30)

        if i == 1:
            return

if __name__ == '__main__':
    main()
