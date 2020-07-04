from iCaRL_net import iCaRL
import numpy as np
import utils
import torch




####Hyper-parameters####
DEVICE = 'cuda'
BATCH_SIZE = 128
CLASSES_BATCH = 10
MEMORY_SIZE = 2000
########################



def incremental_learning(dict_num, loss_config, classifier, lr, undersample=False, oversample=False, resize_factor=0.5, random_flag=False, mix_up=False, second_training=False, double_ex=False, dbscan=False):
    utils.set_seed(0)
    
    
    path='orders/'
    classes_groups, class_map, map_reverse = utils.get_class_maps_from_files(path+'classgroups'+dict_num+'.pickle',
                                                                             path+'map'+ dict_num +'.pickle',
                                                                             path+'revmap'+ dict_num +'.pickle')
    
    #print classes mapped to fake class
    print(classes_groups, class_map, map_reverse)

    #net = iCaRL(0, class_map, loss_config=loss_config,lr=lr, class_balanced_loss=class_balanced_loss, proportional_loss=proportional_loss)
    net = iCaRL(0, class_map, map_reverse=map_reverse, loss_config=loss_config,lr=lr, mix_up=mix_up, dbscan=dbscan)

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
        
            print('known', net.n_known)
            
            resize_factor = MEMORY_SIZE/(10*i*500)
            train_dataset.resample(resize_factor = resize_factor)
            print('Resamplig to size', len(train_dataset)) 
            
        
        if oversample and i !=0:
            net.oversample_exemplars(2000)
            print('Oversampling exemplars')
            
        print('-'*30)
        print(f'Known classes: {net.n_known}')
        print('-'*30)
        print('Updating representation ...')
        print('-'*30)
        
        # Perform the training as described in iCaRL Paper
        net.update_representation(dataset=train_dataset, class_map=class_map, map_reverse=map_reverse, iter=i, double_ex=double_ex)
        
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
