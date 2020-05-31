import numpy as np

import torch

import utils

import torch.nn as nn
import torch.optim as optim

import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from resnet import resnet32


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
def train(net, train_dataloader, n_classes, class_map):

  criterion = nn.BCEWithLogitsLoss() #binary CrossEntropyLoss
  parameters_to_optimize = net.parameters() # In this case we optimize over all the parameters of AlexNet
  optimizer = optim.SGD(parameters_to_optimize, lr=LR, weight_decay=WEIGHT_DECAY)

  net.to(DEVICE)

  for epoch in range(NUM_EPOCHS):

    if epoch in STEPDOWN_EPOCHS:
      for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']/STEPDOWN_FACTOR

    running_loss = 0.0
    running_corrects_train = 0

    for inputs, labels, index in train_dataloader:
      inputs = inputs.to(DEVICE)
      # We need to save labels in this way because classes are randomly shuffled at the beginning
      seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
      labels = Variable(seen_labels).to(DEVICE)
      labels_hot=torch.eye(n_classes)[labels]
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

    # Calculate average losses
    epoch_loss = running_loss / len(train_dataloader.dataset)
    # Calculate accuracy
    epoch_acc = running_corrects_train.double() / len(train_dataloader.dataset)

    if epoch % 10 == 0 or epoch == (NUM_EPOCHS-1):
      print('Epoch {} Avg Loss:{:.4f}'.format(epoch, epoch_loss))
      for param_group in optimizer.param_groups:
        print('Learning rate:{}'.format(param_group['lr']))
      print('-'*30)

  return net


def test(net, test_dataloader, map_reverse):
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

    preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
    running_corrects += (preds == labels.cpu().numpy()).sum()

  # Calculate Accuracy
  accuracy = running_corrects / float(len(test_dataloader.dataset))

  return accuracy


def incremental_learning(num):

  path='orders/'
  classes_groups, class_map, map_reverse = utils.get_class_maps_from_files(path+'classgroups'+ num +'.pickle',
                                                                             path+'map'+ num +'.pickle',
                                                                             path+'revmap'+ num +'.pickle')
  print(classes_groups, class_map, map_reverse)

  net = resnet32(num_classes=0)
  #print(net)

  new_acc_list = []
  old_acc_list = []
  all_acc_list = []

  for i in range(int(100/CLASSES_BATCH)):

    print('-'*30)
    print(f'**** ITERATION {i+1} ****')
    print('-'*30)

    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, 10+i*10) # Change output nodes
    n_classes = 10+i*10

    print('Loading the Datasets ...')
    print('-'*30)

    train_dataset, val_dataset, test_dataset = utils.get_datasets(classes_groups[i])

    print('-'*30)
    print('Training ...')
    print('-'*30)

    # Prepare Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

    net = train(net, train_dataloader, n_classes, class_map)

    print('Testing ...')
    print('-'*30)

    print('New classes')
    new_acc = test(net, test_dataloader, map_reverse)
    print(new_acc)

    new_acc_list.append(new_acc)

    if i == 0:
        all_acc_list.append(new_acc)

    if i > 0:

      # Creating dataset for test on previous classes
      previous_classes = np.array([])
      for j in range(i):
        previous_classes = np.concatenate((previous_classes, classes_groups[j]))

      prev_classes_dataset, all_classes_dataset = utils.get_additional_datasets(previous_classes, np.concatenate((previous_classes, classes_groups[i])))

      # Prepare Dataloaders
      test_prev_dataloader = DataLoader(prev_classes_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
      test_all_dataloader = DataLoader(all_classes_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

      print('Old classes')
      old_acc = test(net, test_prev_dataloader, map_reverse)
      print(old_acc)
      print('All classes')
      all_acc = test(net, test_all_dataloader, map_reverse)
      print(all_acc)

      old_acc_list.append(old_acc)
      all_acc_list.append(all_acc)
      print('-'*30)


  return new_acc_list, old_acc_list, all_acc_list
