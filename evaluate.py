'''
    This is the file from which you call the training+test functions performed from the three different methods:
    - Finetuning
    - Learning without Forgetting (implementation described in iCaRL paper)
    - iCaRL

'''

import fine_tuning as ft
import LwF as lwf
import iCaRL as icarl

import matplotlib.pyplot as plt

import utils


'''
def plot_accuracy(classes_split,num):

    plt.figure(figsize=(8, 6))
    for key in classes_split:
       plt.plot([i for i in range(10,110,10)],classes_split[key],marker = 'o', markersize=7, label=key)

    plt.legend()
    plt.title("Test accuracy with split: %s" %str(num))
    plt.xlabel('Known classes')
    plt.ylabel('Accuracy')
    plt.show()

    return
'''

def main(i):   # The parameter i can be set to ['1','2','3'] depending on the random split of the dataset you want to load.
               # Change this value if you want to perform calculations with other random splits

    new_dict_acc = {}
    old_dict_acc = {}
    all_dict_acc = {}
    
    # Iterate over different methods in order to calculate the different accuracies
    for learner in [ft, lwf, icarl]:

        print(f"Incremental learning: {learner.__name__}\n")
        print(f"Classes group {i}\n")
        
        # Call the incremental_learning function to perform train+test on 10 iterations
        if learner == icarl:
            # iCaRL's incremental learning function has additional parameters: 
            # loss_config = ['bce', 'mse', 'l1', 'kl']  is the distillation loss function we want to use. 
            #                                           classification loss is always bce.
            # classifier = ['nme', 'knn', 'svc']        is the classifier we want to use.
            # lr = int                                  is the learning rate.
            # new_herding = [True, False]               is automatically set to False. If new_herding = True, the custom 
            #                                           exemplar selection will be performed.
            new_acc_, old_acc_, all_acc_ = learner.incremental_learning(i, loss_config='bce', classifier='nme',lr=2.0)
        else:
            new_acc_, old_acc_, all_acc_ = learner.incremental_learning(i)
        print('new_acc', new_acc_)
        print('old_acc', old_acc_)
        print('all_acc', all_acc_)

        learner_name = learner.__name__
        new_dict_acc[learner_name] = new_acc_
        old_dict_acc[learner_name] = old_acc_
        all_dict_acc[learner_name] = all_acc_
        #print(dict_acc)

    return new_dict_acc, old_dict_acc, all_dict_acc



if __name__ == '__main__':
    main()
