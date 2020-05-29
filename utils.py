from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from dataset import CIFAR100
from torch.utils.data import Subset

import matplotlib.pyplot as plt

import copy
import numpy as np
import pickle


train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ])

test_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                   #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                   #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ])


def get_train_test(classes):
    train_dataset = CIFAR100(root='data/', classes=classes, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR100(root='data/', classes=classes,  train=False, download=True, transform=test_transform)

    return train_dataset, test_dataset


def get_datasets(classes):
    #define images transformation
    #define images transformation

    train_dataset = CIFAR100(root='data/', classes=classes, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR100(root='data/', classes=classes,  train=False, download=True, transform=test_transform)

    train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.1, stratify=train_dataset.targets)

    val_dataset = Subset(copy.deepcopy(train_dataset), val_indices)
    train_dataset = Subset(train_dataset, train_indices)

    val_dataset.dataset.transform = test_transform

    return train_dataset, val_dataset, test_dataset


def get_additional_datasets(prev_classes, all_classes):

    test_prev_dataset = CIFAR100(root='data/', classes=prev_classes,  train=False, download=True, transform=test_transform)
    test_all_dataset = CIFAR100(root='data/', classes=all_classes,  train=False, download=True, transform=test_transform)

    return test_prev_dataset, test_all_dataset



def get_class_maps_from_file(map_filename, revmap_filename):

    with open(map_filename, 'rb') as handle:
        class_map = pickle.load(handle)

    with open(revmap_filename, 'rb') as handle:
        map_reverse = pickle.load(handle)


    return class_map, map_reverse


def dump_dict(num,dictionary):

    with open('accuracy'+num+'.pickle', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_dict_from_file(file_name):

    with open(file_name, 'rb') as handle:
        dict = pickle.load(handle)

    return dict


def get_class_maps():

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

    return classes_groups, class_map, map_reverse

def dump_class_maps():

    classes_groups, class_map, map_reverse = get_class_maps()

    group_dict = {}

    for i in range(len(classes_groups)):
      group_dict[i] = classes_groups[i]

    with open('map3.pickle', 'wb') as handle:
        pickle.dump(class_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('revmap3.pickle', 'wb') as handle:
        pickle.dump(map_reverse, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('classgroups3.pickle', 'wb') as handle:
        pickle.dump(map_reverse, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_class_maps_from_files(classgroup_filename, map_filename, revmap_filename):

    with open(classgroup_filename, 'rb') as handle:
        class_groups_dict = pickle.load(handle)

    with open(map_filename, 'rb') as handle:
        class_map = pickle.load(handle)

    with open(revmap_filename, 'rb') as handle:
        map_reverse = pickle.load(handle)

    class_groups = [class_groups_dict[i] for i in range(len(class_groups_dict.keys()))]
    class_groups = np.array_split(class_groups, 10)

    return class_groups, class_map, map_reverse


def elaborate_results(dict_files):
    ft = []
    lwf = []
    icarl = []

    for file in dict_files:
        dict = get_dict_from_file(file)
        ft.append(dict['fine_tuning'])
        lwf.append(dict['LwF'])
        icarl.append(dict['iCaRL'])

    ft = np.array(ft)
    lwf = np.array(lwf)
    icarl = np.array(icarl)

    #print(ft)
    #print(lwf)
    #print(icarl)

    ft_mean, ft_std = np.mean(ft, axis=0), np.std(ft, axis=0)
    lwf_mean, lwf_std = np.mean(lwf, axis=0), np.std(lwf, axis=0)
    icarl_mean, icarl_std = np.mean(icarl, axis=0), np.std(icarl, axis=0)

    print(ft_mean)
    print(ft_std)

    xs = [list(range(10,110,10)),list(range(10,110,10)), list(range(10,110,10))]
    ys = [ft_mean, lwf_mean, icarl_mean]
    errs = [ft_std, lwf_std, icarl_std]

    create_plot(xs, ys, errs, 'Mean accuracies over 3 different splits')


def create_plot(xs, ys, errs, title):

    print(xs[0], ys[0])

    fig, ax = plt.subplots()
    for i, label in zip([0,1,2], ['fine_tuning', 'LwF', 'iCaRL']):
        ax.errorbar(xs[i], ys[i], yerr=errs[i], fmt='o', label=label)
    ax.set_yticks(np.arange(0, 1., 0.1))
    ax.set_ylim(bottom=0.)
    ax.legend(loc='upper right')
    fig.suptitle(title)

    ax.xlabel('Known classes')
    ax.ylabel('Accuracy')

    fig.savefig('results.png')
    fig.show()

    return
