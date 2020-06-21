from prova_net import iCaRL
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



def incremental_learning(dict_num, loss_config, classifier, lr, loss1=False, undersample=False, resize_factor=0.5, random_flag=False, proportional_loss=False):
    utils.set_seed(0)
    path='orders/'
    classes_groups, class_map, map_reverse = utils.get_class_maps_from_files(path+'classgroups'+dict_num+'.pickle',
                                                                             path+'map'+ dict_num +'.pickle',
                                                                             path+'revmap'+ dict_num +'.pickle')
    print(classes_groups, class_map, map_reverse)

    #net = iCaRL(0, class_map, loss_config=loss_config,lr=lr, class_balanced_loss=class_balanced_loss, proportional_loss=proportional_loss)
    net = iCaRL(0, class_map, map_reverse=map_reverse, loss1=loss1, loss_config=loss_config,lr=lr, proportional_loss=proportional_loss)

    new_acc_list = []
    old_acc_list = []
    all_acc_list = []
    
    # Perform 10 iterations
    for i in range(int(100/CLASSES_BATCH)):
        
        print('-'*30)
        print(f'**** Iteration {i+1} / {int(100/CLASSES_BATCH)} ****')
        print('-'*30)
        
        torch.cuda.empty_cache()
        
        net.new_means = []
        net.compute_means = True
        net.train_model = True
        
        print('Loading the Datasets ...')
        print('-'*30)
        
        # Load the dataset
        train_dataset, test_dataset = utils.get_train_test(classes_groups[i])
        
        if undersample and i != 0: # Undersampling the dataset (experiment)       
            train_dataset.resample(resize_factor = (net.n_known)/5000)
            print('Resamplig to size', len(train_dataset)) 
            
        print('-'*30)
        print(f'Known classes: {net.n_known}')
        print('-'*30)
        print('Updating representation ...')
        print('-'*30)
        
        # Perform the training as described in iCaRL Paper
        net.update_representation(dataset=train_dataset, class_map=class_map, map_reverse=map_reverse, iter=i)
        
        m = MEMORY_SIZE // (net.n_classes)
        net.exemplars_per_class = m
            
        # Reduce exemplar sets only if not first iteration, selecting only first m elements of each class
        if i != 0:
            print('Reducing exemplar sets ...')
            print('-'*30)
            net.reduce_exemplars_set(m)
        
        print('len prev ex', len(net.exemplar_sets))
        print('Constructing exemplar sets ...')
        print('-'*30)
        
        # Construct, at each iteration, new exemplars for the new classes
        for y in classes_groups[i]:
           net.construct_exemplars_set(train_dataset.get_class_imgs(y), m, random_flag)
        
        print('len exemplars of previous classes', len(net.exemplar_sets))
        
        # If not first iteration, perform a second training, only on the exemplars (old+new classes)
        if i !=0:
            print('Training on exemplars...')
            print('-'*30)

            #net.new_means=[]
            net.train_on_exemplars(class_map, map_reverse, iter = i)

            net.new_means=[]
            print('Recomputing new means ...')
            print('-'*30)

            for y in classes_groups[i]:
               net.compute_new_means(train_dataset.get_class_imgs(y))
       
      
        print('-'*30)
        
        print('Testing ...')
        print('-'*30)
        print('New classes')
        
        # Classify on new classes
        new_acc = net.classify_all(test_dataset, map_reverse, classifier=classifier, train_dataset=train_dataset)
        new_acc_list.append(new_acc)
        
        if i == 0:
            all_acc_list.append(new_acc)
        
        if i > 0:
            previous_classes = np.array([])
            
            for j in range(i):
                previous_classes = np.concatenate((previous_classes, classes_groups[j]))
            
            prev_classes_dataset, all_classes_dataset = utils.get_additional_datasets(previous_classes, np.concatenate((previous_classes, classes_groups[i])))
            
            print('Old classes')
            # Classify the old learned classes
            old_acc = net.classify_all(prev_classes_dataset, map_reverse, classifier=classifier)
            
            print('All classes')
            # Classify all the classes seen so far
            all_acc = net.classify_all(all_classes_dataset, map_reverse, classifier=classifier)
            
            old_acc_list.append(old_acc)
            all_acc_list.append(all_acc)
            print('-'*30)
        
        print('lunghezza medie', len(net.exemplar_means))
        print('lunghezza nuove medie', len(net.new_means))
       
        net.n_known = net.n_classes
        
        #if undersample:
            #return new_acc_list, old_acc_list, all_acc_list
    
    return new_acc_list, old_acc_list, all_acc_list
