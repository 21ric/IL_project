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
    self.feature_extractor = resnet32()
    self.feature_extractor.fc = nn.Linear(64, num_classes)

    self.loss = nn.CrossEntropyLoss()
    self.dist_loss = nn.BCELoss()

    self.optimizer   = optim.SGD(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    self.num_classes = num_classes
    self.num_known   = 0
    self.exemplars   = []



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
    q = torch.zeros(len(dataset), self.num_classes).cuda()
    for images, labels, indexes in dataloader:
        images = images.cuda()
        indexes = indexes.cuda()
        g =  torch.sigmoid(self(images))
        q[indexes] = g.data
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


  def dist(a, b):
        """Computes L2 distance between two tensors.
        :param a: A tensor.
        :param b: A tensor.
        :return: A tensor of distance being of the shape of the "biggest" input
                 tensor.
        """
        return torch.pow(a - b, 2).sum(-1)


  def get_the_closest(centers, features):

        """Returns the center index being the closest to each feature.
        :param centers: Centers to compare, in this case the class means.
        :param features: A tensor of features extracted by the convnet.
        :return: A numpy array of the closest centers indexes.
        """
        pred_labels = []

        features = features
        for feature in features:
            distances = dist(centers, feature)
            pred_labels.append(distances.argmin().item())

        return np.array(pred_labels)


  def classify(self, dataloader):


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
     
        ypred = []
        ytrue = []

        for inputs, targets, _ in dataloader:
            inputs = inputs.to(DEVICE)

            features = self.feature_extractor.extract_features(inputs).detach()
            #preds = get_the_closest(exemplar_means,features)

            pred_labels = []

      
            for feature in features:
                #distances = dist(centers, feature)
                  distances = torch.pow(exemplar_means - feature, 2).sum(-1)
                  pred_labels.append(distances.argmin().item())

            preds = np.array(pred_labels)

            ypred.extend(preds)
            ytrue.extend(targets)

        return np.array(ypred), np.array(ytrue)







 
'''
        compute_means = True
        if compute_means:
            print("Computing mean of exemplars...")
            exemplar_means = []
            for P_y in self.exemplars:
                features = []
                # Extract feature for each exemplar set in P_y
                for ex in P_y:
                    ex = Variable(transform(Image.fromarray(ex)), volatile=True).cuda()
                    feature = self.feature_extractor.extract_features(ex.unsqueeze(0))
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm() # Normalize
                    features.append(feature)
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm() # Normalize
                exemplar_means.append(mu_y)
            #self.exemplar_means = exemplar_means
            #self.compute_means = False
            print("Done")

        #exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means) # (n_classes, feature_size)
        means = torch.stack([means] * BATCH_SIZE) # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2) # (batch_size, feature_size, n_classes)

        preds = []
        running_corrects = 0
        for inputs, targets, _ in dataloader:
            print('qui?')
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            feature = self.feature_extractor.extract_features(inputs) # (batch_size, feature_size)
            #for i in xrange(feature.size(0)): # Normalize
            #    feature.data[i] = feature.data[i] / feature.data[i].norm()
            feature = feature.unsqueeze(2) # (batch_size, feature_size, 1)
            feature = feature.expand_as(means) # (batch_size, feature_size, n_classes)

            dists = (feature - means).pow(2).sum(1).squeeze() #(batch_size, n_classes)
            _, predictions = dists.min(1)

            running_corrects += torch.sum(predictions == targets.data).data.item()  #torch.from_numpy(
            preds.extend(predictions)

        accuracy = running_corrects/len(dataloader.dataset)
        print(f"Test accuracy: {accuracy}")
        print(preds)
        return preds
'''
