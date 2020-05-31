'''
    This is the file from which you call the training+test functions performed from the three different methods:
    - Finetuning
    - Learning without Forgetting (implementation described in iCaRL paper)
    - iCaRL

    The training phase is commented because it implies the serialization of the accuracy results which are stored
    in a praticular file. Uncomment that section if you want to perform the training and see the results
'''

import fine_tuning as ft
import LwF as lwf
import iCaRL as icarl

import matplotlib.pyplot as plt

import utils



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

def main(i):   # The parameter i can be set to ['1','2','3'] depending on the random split of the dataset you want to load.
               # Change this value if you want to perform calculations with other random splits

    new_dict_acc = {}
    old_dict_acc = {}
    all_dict_acc = {}
    for learner in [ft, lwf, icarl]:

        print(f"Incremental learning: {learner.__name__}\n")
        print(f"Classes group {i}\n")
        # Call the incremental_learning function to perform train+test on 10 iterations
        if learner == icarl:
            new_acc_, old_acc_, all_acc_ = learner.incremental_learning(i, loss_config=0, classifier='nme',lr=2.0)
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

    '''
    # Write accuracy lists on file
    utils.dumb_dict(i, dict_acc)

    '''
        #COMMENT FROM NOW ON IF YOU JUST WANT TO STORE RESULTS TO FILE
    '''
    # Load accuracy lists from file
    dict_1,dict_2,dict_3 = utils.get_dict_from_file()

    # This is an example dict to see if the plot works: IT DOESN'T SO PLEASE MODIFY THE CODE .... NOW IT WORKS!

    dict_1 = {'fine_tuning': [0.295, 0.203, 0.13533333333333333, 0.093, 0.1054, 0.07733333333333334, 0.05785714285714286, 0.0635, 0.034444444444444444, 0.051],
              'LwF': [0.158, 0.1165, 0.07366666666666667, 0.04575, 0.0376, 0.029833333333333333, 0.037142857142857144, 0.03325, 0.03133333333333333, 0.0285],
              'iCaRL': [0.147, 0.09, 0.058333333333333334, 0.043, 0.0344, 0.0315, 0.024428571428571428, 0.024625, 0.02033333333333333, 0.0197]
             }

    plot_accuracy(dict_1,1)
    '''



if __name__ == '__main__':
    main()
