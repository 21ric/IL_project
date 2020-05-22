from LWF_net import LwF

from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import CIFAR100

import numpy as np


####Hyper-parameters####
DEVICE = 'cuda'
NUM_CLASSES = 10
BATCH_SIZE = 128
ClASSES_BATCH = 10
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

  #range_classes = np.arange(100)
  total_classes = 100
  perm_id = np.random.permutation(total_classes)
  all_classes = np.arange(total_classes)
  for i in range(len(all_classes)):
	  all_classes[i] = perm_id[all_classes[i]]
  
  # Create groups of 10
  classes_groups = np.array_split(all_classes, 10)
  print(classes_groups)


  net = LwF(0)
  
  for i in range(1): #range(int(100/ClASSES_BATCH)):

      train_dataset = CIFAR100(root='data/', classes=classes_groups[i], train=True, download=True, transform=train_transform)
      test_dataset = CIFAR100(root='data/', classes=classes_groups[i],  train=False, download=True, transform=test_transform)

      train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
      test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
      
      # UPDATE STEP on train set
      # net.update(dataset = train_dataset)

      # EVALUATION STEP on training set and test set     
      # net.classify(...)




if __name__ == '__main__':
    main()
