from iCaRL_net import iCaRL

from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import CIFAR100

import numpy as np

import math

import torch

####Hyper-parameters####
DEVICE = 'cuda'
BATCH_SIZE = 128
ClASSES_BATCH = 10
MEMORY_SIZE = 40
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

    range_classes = np.arange(100)
    classes_groups = np.array_split(range_classes, 10)


    net = iCaRL(0)

    for i in range(int(100/ClASSES_BATCH)):

        train_dataset = CIFAR100(root='data/', classes=classes_groups[i], train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(root='data/', classes=classes_groups[i],  train=False, download=True, transform=test_transform)

        net.update_representation(dataset = train_dataset)

        #print('Dato train sample')
        #print(train_dataset.data[:1])
        #print('tipo')
        #print(type(train_dataset.data[:1]))

        print('done upd')

        m = int(math.ceil(MEMORY_SIZE/net.n_classes))

        net.reduce_exemplars_set(m)


        for y in range(net.n_known, net.n_classes):
            net.construct_exemplars_set(train_dataset.get_class_imgs(y), m)

        net.n_known = net.n_classes

        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        running_corrects = 0
        
        for imgs, labels, _ in  test_dataloader:
            imgs = imgs
            preds = net.classify(imgs)
            running_corrects += torch.sum(preds == labels.data).data.item()
        accuracy = running_corrects / float(len(test_dataloader.dataset))
        print('Test Accuracy: {}'.format(accuracy))

        if i == 1:
            return

if __name__ == '__main__':
    main()
