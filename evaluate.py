'''
    This is the file from which you call the training+test functions performed from the three different methods: 
    - Finetuning
    - Learning without Forgetting (implementation described in iCaRL paper)
    - iCaRL
    
    The training phase is commented because it implies the serialization of the accuracy results which are stored
    in a praticular file. Uncomment that section if you want to perform the training and see the results
'''

import fine_tuning as ft
import LwF2 as lwf
import iCaRL2 as icarl

import matplotlib.pyplot as plt

import utils

########################
i = '1'     # This parameter can be set to ['1','2','3'] depending on the random split of the dataset you want to load.
            # Change this value if you want to perform calculations with other random splits.
########################

def plot_accuracy():
    pass

def main():
    
    '''
    #for i in ['1','2','3']:
    dict_acc = {}
    for learner in [icarl, lwf, ft]:

        print(f"Incremental learning: {learner.__name__}\n")
        print(f"Classes group {i}\n")
        # Call the incremental_learning function to perform train+test on 10 iterations
        acc_ = learner.incremental_learning(i)
        print(acc_)
        learner_name = learner.__name__
        dict_acc[learner_name] = [accuracy for accuracy in acc_]
        #print(dict_acc)
    
    # Write accuracy lists on file
    utils.dumb_dict(i, dict_acc)
    
    '''
        #COMMENT FROM NOW ON IF YOU JUST WANT TO STORE RESULTS TO FILE
    '''
    # Load accuracy lists from file
    dict_1,dict_2,dict_3 = utils.get_dict_from_file()
    '''
    # This is an example dict to see if the plot works: IT DOESN'T SO PLEASE MODIFY THE CODE
    dict_1 = {'fine_tuning': [0.295, 0.203, 0.13533333333333333, 0.093, 0.1054, 0.07733333333333334, 0.05785714285714286, 0.0635, 0.034444444444444444, 0.051],
              'LwF2': [0.158, 0.1165, 0.07366666666666667, 0.04575, 0.0376, 0.029833333333333333, 0.037142857142857144, 0.03325, 0.03133333333333333, 0.0285],
              'iCaRL2': [0.147, 0.09, 0.058333333333333334, 0.043, 0.0344, 0.0315, 0.024428571428571428, 0.024625, 0.02033333333333333, 0.0197]
             }
    '''
    acc_ft = dict_1['fine_tuning']
    acc_lwf = dict_1['LwF2']
    acc_icarl = dict_1['iCaRL2']'''
    
    # Plot it !
    plt.figure(figsize=(8, 6))
    for key in dict_1:
       plt.plot(dict_1[key], '-ok', markersize=7, label=key)

    #plt.style.use('seaborn-whitegrid')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()






if __name__ == '__main__':
    main()
