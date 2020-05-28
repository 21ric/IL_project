import fine_tuning as ft
import LwF2 as lwf
import iCaRL2 as icarl

import utils


def main():
    
    for i in ['1','2','3']:
        dict_acc = {}
        for learner in [ft, lwf, icarl]:
            
            print(f"Incremental learning: {learner.__name__}\n")
            print(f"Classes group {i}\n")
            acc_ = learner.incremental_learning(i)
            print(acc_)
            learner_name = learner.__name__
            dict_acc[learner_name] = [accuracy for accuracy in acc_]
            #print(dict_acc)

        utils.dumb_dict(dict_acc,i)

    dict_1,dict_2,dict_3 = utils.get_dict_from_file()

         '''
            if i == '1': 
                dict_1[learner_name] = [accuracy for accuracy in acc_]
                print(dict_1)
            if i == '2':
                dict_2[learner_name] = [accuracy for accuracy in acc_]
            if i == '3':
                dict_3[learner_name] = [accuracy for accuracy in acc_]
            '''

















if __name__ == '__main__':
    main()
