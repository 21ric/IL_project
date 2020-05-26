from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import numpy as np


train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                   ])

test_transform = transforms.Compose([transforms.ToTensor(),
                                   #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                   transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                  ])


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

    test_prev_dataset = CIFAR100(root='data/', classes=previous_classes,  train=False, download=True, transform=test_transform)
    test_all_dataset = CIFAR100(root='data/', classes=all_classes,  train=False, download=True, transform=test_transform)

    return test_prev_dataset, test_all_datasets


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
