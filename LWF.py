from LWF_net import LwF

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import CIFAR100

import numpy as np
from numpy import random


####Hyper-parameters####
DEVICE = 'cuda'
BATCH_SIZE = 128
CLASSES_BATCH = 10
MEMORY_SIZE = 2000
########################


def main():
    #  Define images transformation
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

    print("\n")

    #range_classes = np.arange(100)

    total_classes = 20   #try with 2 iterations 

    perm_id = np.random.permutation(total_classes)
    all_classes = np.arange(total_classes)
    
    for i in range(len(all_classes)):
      all_classes[i] = perm_id[all_classes[i]]

    # Create groups of 10
    #classes_groups = np.array_split(all_classes, 10)
    #print(classes_groups)

    num_iters = total_classes//CLASSES_BATCH        #con 20 classi e classes_batch = 10 fa 2 iterazioni

    # Create class map
    class_map = {}
    for i, cl in enumerate(all_classes):
        class_map[cl] = i
    print (f"Class map:{class_map}\n")
    
    # Create class map reversed
    map_reverse = {}
    for cl, map_cl in class_map.items():
        map_reverse[map_cl] = int(cl)
    print ("Map Reverse:{map_reverse}\n")

    # Create Network
    net = LwF(CLASSES_BATCH,class_map)

    #for i in range(int(total_classes//CLASSES_BATCH)):

    for s in range(0, total_classes, CLASSES_BATCH):  #c'era (0, num_iter,CLASSES_BATCH), modificato perchè altrimenti avevamo num_iter=10
                                                      #CLASSES_BATCH= 10 quindi s andava da 0 a 10 e si fermava
   

        print(f"ITERATION: {s//CLASSES_BATCH}\n")
      
        print("\n")
        
        
        # Load Datasets
                                                                       
        train_dataset = CIFAR100(root='data',train=True,classes=all_classes[s:s+CLASSES_BATCH],download=True,transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True, num_workers=4)

        test_dataset = CIFAR100(root='data',train=False,classes=all_classes[:s+CLASSES_BATCH],download=True, transform=test_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=False, num_workers=4)
        
        print("\n")                                       

        #for i in range(1):

        #train_dataset = CIFAR100(root='data/', classes=classes_groups[i], train=True, download=True, transform=train_transform)
        #test_dataset = CIFAR100(root='data/', classes=classes_groups[i],  train=False, download=True, transform=test_transform)

        #train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
        #test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)'''

        # UPDATE STEP on train set
        print("Update step\n")
        net.update(train_dataset, class_map)
        # net.update(dataset = train_dataset)

        print("evalutation step on training and test set\n")
        # EVALUATION STEP on training set and test set   
        net.eval()

        net.n_known = net.n_classes
        
        print ("model classes : %d\n " % net.n_known)

        total = 0.0
        correct = 0.0

        for images, labels, indices in train_dataloader:

            images = Variable(images).cuda()
            preds = net.classify(images)
            preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
            total += labels.size(0)
            correct += (preds == labels.numpy()).sum()

        # Train Accuracy
        #print ('%.2f ,' % (100.0 * correct / total), file=file, end="")
        print ('Train Accuracy : %.2f\n' % (100.0 * correct / total))



        # net.classify(...)
        total = 0.0
        correct = 0.0
        for images, labels, indices in test_dataloader:

            images = Variable(images).cuda()
            preds = net.classify(images)
            preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
            total += labels.size(0)
            correct += (preds == labels.numpy()).sum()

        # Test Accuracy
        #print ('%.2f' % (100.0 * correct / total), file=file)
        print ('Test Accuracy : %.2f\n' % (100.0 * correct / total))

        net.train()


if __name__ == '__main__':
    main()

