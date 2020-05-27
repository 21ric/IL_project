import numpy as np

import torch

import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
import copy

from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from dataset import CIFAR100
from resnet import resnet32

from torchvision.models import resnet18

####Hyper-parameters####
DEVICE = 'cuda'
NUM_CLASSES = 10
BATCH_SIZE = 128
CLASSES_BATCH =10
STEPDOWN_EPOCHS = [49, 63]
STEPDOWN_FACTOR = 5
LR = 2
MOMENTUM = 0.9
WEIGHT_DECAY = 0.00001
NUM_EPOCHS = 70
########################


#train function
def train(net, train_dataloader, val_dataloader, n_classes):

  #criterion = nn.CrossEntropyLoss() # for classification, we use Cross Entropy
  criterion = nn.BCEWithLogitsLoss() #binary CrossEntropyLoss
  parameters_to_optimize = net.parameters() # In this case we optimize over all the parameters of AlexNet
  optimizer = optim.SGD(parameters_to_optimize, lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
  #scheduler = optim.lr_scheduler.StepLR(optimizer)#, step_size=STEP_SIZE, gamma=GAMMA)
  net.to(DEVICE)

  best_acc = 0 #this is the validation accuracy for model selection

  for epoch in range(70):
      if(epoch%5 == 0 ):
        print('-' * 30)
        print('Epoch {}/{}'.format(epoch+1, 70))
        for param_group in optimizer.param_groups:
          print('Learning rate:{}'.format(param_group['lr']))


      #divide learning rate by 5 after 49 63 epochs
      if epoch in STEPDOWN_EPOCHS:
        for param_group in optimizer.param_groups:
          param_group['lr'] = param_group['lr']/STEPDOWN_FACTOR

      running_loss = 0.0
      valid_loss = 0.0
      running_corrects_train = 0
      running_corrects_val = 0

      #
      # TRAINING
      #
      # Iterate over data.
      for inputs, labels, index in train_dataloader:
          inputs = inputs.to(DEVICE)
          labels = labels.to(DEVICE)
          labels_hot = torch.eye(n_classes)[labels] #one hot
          labels_hot = labels_hot.to(DEVICE)

          net.train(True)
          # zero the parameter gradients
          optimizer.zero_grad()
          # forward
          outputs = net(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels_hot)
          loss.backward()
          optimizer.step()

          # statistics
          running_loss += loss.item() * inputs.size(0)
          running_corrects_train += torch.sum(preds == labels.data)

      #
      # VALIDATION
      #
      for inputs, labels, index in val_dataloader:
          inputs = inputs.to(DEVICE)
          labels = labels.to(DEVICE)
          labels_hot = torch.eye(n_classes)[labels] #one hot
          labels_hot = labels_hot.to(DEVICE)

          net.train(False)
          # forward
          outputs = net(inputs)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels_hot)

          # statistics
          valid_loss += loss.item() * inputs.size(0)
          running_corrects_val += torch.sum(preds == labels.data)

      # Calculate average losses
      epoch_loss = running_loss / len(train_dataloader.dataset)
      valid_loss = valid_loss / len(val_dataloader.dataset)
      # Calculate accuracy
      epoch_acc = running_corrects_train.double() / len(train_dataloader.dataset)
      valid_acc = running_corrects_val / float(len(val_dataloader.dataset))

      #Save the model with the best validation accuracy
      if (valid_acc > best_acc):
        best_acc = valid_acc
        best_net = copy.deepcopy(net.state_dict())

      #scheduler.step()

      if(epoch%5 == 0 ):
        print('Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        print('Val Loss: {:.4f} Val Acc: {:.4f}'.format(valid_loss, valid_acc))

  #at the end, load best model weights
  net.load_state_dict(best_net)

  return net





#test function
def test(net, test_dataloader):
  net.to(DEVICE)
  net.train(False)

  running_corrects = 0
  for images, labels, _ in test_dataloader:
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    # Forward Pass
    outputs = net(images)
    # Get predictions
    _, preds = torch.max(outputs.data, 1)
    # Update Corrects
    running_corrects += torch.sum(preds == labels.data).data.item()

  # Calculate Accuracy
  accuracy = running_corrects / float(len(test_dataloader.dataset))
  print('Test Accuracy: {}'.format(accuracy))



def main():

#define images transformation
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

  range_classes = np.arange(100)
  classes_groups = np.array_split(range_classes, 10)


  #net = resnet18()
  net = resnet32(num_classes=0)
  print(net)

  for i in range(int(100/CLASSES_BATCH)):

    print(f"ITERATION {i+1} / {100/CLASSES_BATCH}")
    #cambio il numero di classi di output
    net.fc = nn.Linear(64, 10+i*10)
    n_classes = 10+i*10
    
    if i != 0:

      # Creating dataset for current iteration
      train_dataset = CIFAR100(root='data/', classes=classes_groups[i], train=True, download=True, transform=train_transform)
      test_dataset = CIFAR100(root='data/', classes=classes_groups[i],  train=False, download=True, transform=test_transform)

      # Create indices for train and validation
      train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.1, stratify=train_dataset.targets)

      val_dataset = Subset(train_dataset, val_indices)
      train_dataset = Subset(train_dataset, train_indices)

      # Creating dataset for test on previous classes
      previous_classes = np.array([])
      for j in range(i):
        previous_classes = np.concatenate((previous_classes, classes_groups[j]))
      test_prev_dataset = CIFAR100(root='data/', classes=previous_classes,  train=False, download=True, transform=test_transform)

      # Creating dataset for all classes
      all_classes = np.concatenate((previous_classes, classes_groups[i]))
      test_all_dataset = CIFAR100(root='data/', classes=all_classes,  train=False, download=True, transform=test_transform)

      # Prepare Dataloaders
      train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
      val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
      test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
      test_prev_dataloader = DataLoader(test_prev_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
      test_all_dataloader = DataLoader(test_all_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)


      # Check dataset sizes
      print('Train Dataset: {}'.format(len(train_dataset)))
      print('Valid Dataset: {}'.format(len(val_dataset)))
      print('Test Dataset (new classes): {}'.format(len(test_dataset)))

      net = train(net, train_dataloader, val_dataloader, n_classes)
      print('Test on new classes')
      test(net, test_dataloader)
      print('Test on old classes')
      test(net, test_prev_dataloader)
      print('Test on all classes')
      test(net, test_all_dataloader)

    else: # First iteration

      # Create train and test dataset
      train_dataset = CIFAR100(root='data/', classes=classes_groups[i], train=True, download=True, transform=train_transform)
      test_dataset = CIFAR100(root='data/', classes=classes_groups[i],  train=False, download=True, transform=test_transform)

      # Create indices for train and validation
      train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.1, stratify=train_dataset.targets)

      val_dataset = Subset(train_dataset, val_indices)
      train_dataset = Subset(train_dataset, train_indices)

      # Check dataset sizes
      print('Train Dataset: {}'.format(len(train_dataset)))
      print('Valid Dataset: {}'.format(len(val_dataset)))
      print('Test Dataset: {}'.format(len(test_dataset)))

      # Prepare Dataloaders
      train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
      test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
      val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

      net = train(net, train_dataloader, val_dataloader, n_classes)
      print('Test on first 10 classes')
      test(net, test_dataloader)

    #if i==1:
        #return #per fare solo la prima iterazione (10 classi) fin quando non si replicano i risultati

if __name__ == '__main__':
    main()
