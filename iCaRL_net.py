import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from resnet import resnet32

import math

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

####Hyper-parameters####
LR = 2
WEIGHT_DECAY = 0.00001
BATCH_SIZE = 128
STEPDOWN_EPOCHS = [49, 63]
STEPDOWN_FACTOR = 5
NUM_EPOCHS = 4
DEVICE = 'cuda'
########################


class iCaRL(nn.Module):
  def __init__(self, num_classes):
    super(iCaRL,self).__init__()
    self.feature_extractor = resnet32(iCaRL=True)

    self.loss = nn.CrossEntropyLoss()
    self.dist_loss = nn.BCELoss()

    self.optimizer   = optim.SGD(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    self.num_classes = num_classes
    self.num_known   = 0
    self.exemplars   = []
    self.means = None


  def forward(self, x):
    x = self.feature_extractor(x)
    return(x)

  def update_representation(self, dataset):

    targets = list(set(dataset.targets))
    n = len(targets)
    self.cuda()
    print('{} new classes'.format(len(targets)))

    #merge new data and exemplars
    for y, exemplars in enumerate(self.exemplars):
        dataset.append(exemplars, [y]*len(exemplars))

    print('Len dataset+exemplar', len(dataset))

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    #Store network outputs with pre-updated parameters
    if self.num_classes !=0:
        q = torch.zeros(len(dataset), self.num_classes).cuda()
        self.eval()
        for images, labels, indexes in dataloader:
            images = images.cuda()
            indexes = indexes.cuda()
            g =  torch.sigmoid(self(images))
            q[indexes] = g.data
        self.train()
        q.cuda()


    #Increment classes
    in_features = self.feature_extractor.fc.in_features
    out_features = self.feature_extractor.fc.out_features
    weight = self.feature_extractor.fc.weight.data

    self.feature_extractor.fc = nn.Linear(in_features, out_features+n, bias=False)
    self.feature_extractor.fc.weight.data[:out_features] = weight
    self.num_known = self.num_classes
    self.num_classes += n

    self.optimizer   = optim.SGD(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer = self.optimizer


    self.cuda()
    i=0
    for epoch in range(NUM_EPOCHS):

        if epoch in STEPDOWN_EPOCHS:
            for param_group in optimizer.param_groups:
              param_group['lr'] = param_group['lr']/STEPDOWN_FACTOR

        if i%5 == 0:
            print('-'*30)
            print('Epoch {}/{}'.format(i+1, NUM_EPOCHS))
            for param_group in optimizer.param_groups:
                print('Learning rate:{}'.format(param_group['lr']))

        for images, labels, indexes in dataloader:
            images = images.cuda()
            labels = labels.cuda()
            indexes = indexes.cuda()

            #zero-ing the gradients
            optimizer.zero_grad()
            #hidden.detach_()
            out = self(images)

            #classification Loss
            #loss = sum(self.loss(out[:,y], 1 if y==labels else 0) for y in range(self.num_known, self.num_classes))
            loss = self.loss(out, labels)
            #distillation Loss
            if self.num_known > 0:
                q_i = q[indexes]
                dist_loss = sum(self.dist_loss(out[:, y], q_i[:, y]) for y in range(self.num_known))
                loss += dist_loss

            loss.backward()
            optimizer.step()

        if i%5 == 0:
            print("Loss: {:.4f}".format(loss.item()))
        i+=1
    print('end epoch')

  def reduce_exemplars_set(self, m):
    for y, exemplars in enumerate(self.exemplars):
        self.exemplars[y] = exemplars[:m]


  def construct_exemplars_set(self, images, m, transform):

    feature_extractor = self.feature_extractor.to(DEVICE)
    features = []
    self.eval()
    for img in images:
        img = Image.fromarray(img)
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(DEVICE)
        feature = feature_extractor.extract_features(img).data.cpu().numpy().squeeze()
        features.append(feature)

    #print('caricato immagini per costruzione')
    #print(features)
    #features = features / np.linalg.norm(features)
    #class_mean = np.mean(np.array(features))
    class_mean = None
    for feature in features:
        if class_mean is None:
            class_mean = feature
        else:
            class_mean += feature
    class_means = class_mean / len(features)

    #print('Costruzione exemp---class_mean:{}'.format(class_mean))
    #class_mean = class_mean / np.linalg.norm(class_mean)


    exemplar_set = []
    exemplar_features = []
    for k in range(m):

        S = np.sum(exemplar_features, axis=0)
        phi = features
        mu = class_mean
        mu_p = 1.0/(k+1) * (phi + S)
        #mu_p = mu_p / np.linalg.norm(mu_p)
        #mu_p = mu_p / np.linalg.norm(mu_p)
        i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))

        exemplar_set.append(images[i])
        exemplar_features.append(features[i])
        #print('Indice scelto:{}'.format(i))

        images = np.concatenate((images[:i], images[i+1:]))

        if i == 0:
            features = features[i+1:]

        elif i == len(features):
            features = features[:i-1]
        else:
            features = np.concatenate((features[:i], features[i+1:]))
        #images = np.concatenate((images[:i], images[i+1:]))
        #features = np.concatenate((features[:i], features[i+1:]))
        #print('F1',features[i-2:i])
        #print('F2',features[i+1: i+3])

    #print(exemplar_set[:3])
    self.exemplars.append(exemplar_set)
    self.train()



  #da cambiare completamente

  def classify(self, dataloader, transform):

        #compute the mean for each examplars
        cond = False

        class_means = None

        exemplar_means=[]
        for exemplars in self.exemplars:

            feature_extractor = self.feature_extractor.to(DEVICE)
            features = []


            for ex in exemplars:
                ex = Image.fromarray(ex)
                ex = transform(ex)
                ex = ex.unsqueeze(0)
                ex = ex.to(DEVICE)
                feature = feature_extractor.extract_features(ex).data.cpu().numpy().squeeze()
                #feature = feature / np.linalg.norm(feature)
                features.append(feature)



            for feature in features:
                if class_means is None:
                    class_means = feature
                else:
                    class_means += feature
            class_means = class_means / len(exemplars)
            #class_means = np.mean(features, axis=1)

            exemplar_means.append(class_means)

        #print('Numero di classi in classify:{}'.format(len(class_means)))
        #print('Medie per classi')
        #print(exemplar_means)



        """

        if exemplar_means is None:
            raise ValueError(
                "Cannot classify without built examplar means,"
            )

        if exemplar_means.shape[0] != self.num_classes:
            raise ValueError(
                "The number of examplar means ({}) is inconsistent".format(exemplar_means.shape[0])
            )

        """
        #if exemplar_means is None:
         #   raise ValueError(
          #      "Cannot classify without built examplar means,"
           # )

        #if exemplar_means.shape[0] != self.num_classes:
         #   raise ValueError(
          #      "The number of examplar means ({}) is inconsistent".format(exemplar_means.shape[0])
           # )


        ypred = []
        ytrue = []


        running_corrects = 0
        for inputs, targets, _ in dataloader:
            imputs = inputs

            inputs = inputs.to(DEVICE)
            #compute the feature map of the input
            features = self.feature_extractor.extract_features(inputs).data.cpu().numpy().squeeze()

            dist =[]
            for mean in exemplar_means:
                dist.append(np.sqrt(np.sum((mean - features) ** 2, axis=1)))#(batch_size, n_classes)
            preds = np.argmin(np.array(dist), axis=1)
            print('len preds',len(preds))
            #_, preds = dists.min(1)

            ypred.extend(preds)
            ytrue.extend(targets)

            running_corrects += torch.sum(torch.from_numpy(preds) == targets.data).data.item()
            print("qua")

            pred_labels = []
            a = 0

            for feature in features:
              #feature = feature / np.linalg.norm(feature)
              #computing L2 distance
              distances = torch.pow(exemplar_means - feature, 2).sum(-1)



              for mean in exemplar_means:
                  if a > 0:
                      print('sottrazione')
                      print(mean - feature)
                  distances.append(np.sqrt(np.sum((mean - features) ** 2, axis=1)))
              if a > 0:
                  print('DISTANCES')
                  print(distances)
                  a = a-1
              pred_labels.append(np.argmin(distances))

            preds = np.array(pred_labels)

            running_corrects += torch.sum(torch.from_numpy(preds) == targets.data).data.item()
            print('Running corrects')
            print(running_corrects)

            ypred.extend(preds)
            ytrue.extend(targets)


        print(ypred)

        accuracy = running_corrects / float(len(dataloader.dataset))
        print(f"Test accuracy: {accuracy}")

        return ypred, ytrue
