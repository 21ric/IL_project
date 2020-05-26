from LWF_net import LwF

from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import copy

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


    print("\n")


    total_classes = 100    

    perm_id = np.random.permutation(total_classes)
    all_classes = np.arange(total_classes)
    
    # Mix the classes indexes
    for i in range(len(all_classes)):
      all_classes[i] = perm_id[all_classes[i]]

    # Create class map
    class_map = {}
    #takes 10 new classes randomly
    for i, cl in enumerate(all_classes):
        class_map[cl] = i
    print (f"Class map:{class_map}\n")     
    
    # Create class map reversed
    map_reverse = {}
    for cl, map_cl in class_map.items():
        map_reverse[map_cl] = int(cl)
    print (f"Map Reverse:{map_reverse}\n")


    # Create Network
    net = LwF(CLASSES_BATCH, class_map)
 
      
    #iterating until the net knows total_classes with 10 by 10 steps 
    for s in range(0, total_classes, CLASSES_BATCH):   
        print(f"ITERATION: {(s//CLASSES_BATCH)+1} / {total_classes//CLASSES_BATCH}\n")
        print("\n")
        
        # Creating dataset for current iteration        
        train_dataset = CIFAR100(root='data',train=True,classes=all_classes[s:s+CLASSES_BATCH],download=True,transform=train_transform)
        test_dataset = CIFAR100(root='data',train=False,classes=all_classes[:s+CLASSES_BATCH],download=True, transform=test_transform)
        
        # Create indices for train and validation
        train_indexes, val_indexes = train_test_split(range(len(train_dataset)), test_size=0.1, stratify=train_dataset.targets)
        
        val_dataset = Subset(train_dataset, val_indexes)   
        train_dataset = Subset(train_dataset, train_indexes)
   
        # Prepare dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=False, num_workers=4, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=False, num_workers=4, drop_last=False)
        
        print("\n")                                       

        # UPDATE STEP on training set
        print("Update step\n")
      
        # returns the history of val loss and accuracy and the net providing the lower val loss
        results = net.update(train_dataset, val_dataset, class_map, map_reverse)
        
        """
        # takes the dictionary {num_epoch : [val_acc, avg_val_loss]}
        scores = results[0]
        sorted_scores = sorted(scores.items(), key=lambda x: x[1][1]) # sorted according to the lower val loss

        print(f"lower validation loss: {sorted_scores[0][1][1]} at epoch:{sorted_scores[0][0]}:\n")
 
        # takes the best net
        to_test = results[1]
        """
        # EVALUATION STEP
        print("\nevalutation step on training and test set\n")  
        net.eval()
        net.n_known = net.n_classes
        
        print ("the model knows %d classes:\n " % net.n_known)
        print(f"Ready to evaluate train set, len= {len(train_dataset)}")
  
        #Evaluating on training set
        total = 0.0
        correct = 0.0

        for images, labels, indices in train_dataloader:

            images = Variable(images).cuda()
            preds = net.classify(images)
            preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
            total += labels.size(0)
            correct += (preds == labels.numpy()).sum()

        # Train Accuracy
        print ('Train Accuracy : %.2f\n' % (100.0 * correct / total))


        #EValuating on test set
        print(f"Ready to evaluate test set, len= {len(test_dataset)}")
       
        total = 0.0
        correct = 0.0
        for images, labels, indices in test_dataloader:

            images = Variable(images).cuda()
            preds = to_test.classify(images)
            preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
            total += labels.size(0)
            correct += (preds == labels.numpy()).sum()

        # Test Accuracy
        print ('Test Accuracy : %.2f\n' % (100.0 * correct / total))


        #set the net to train for the next iteration 
        net.train(True)


if __name__ == '__main__':
    main()

