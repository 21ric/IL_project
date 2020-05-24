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
NUM_EPOCHS = 70
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
    features = features / np.linalg.norm(features)
    class_mean = np.mean(np.array(features))
    class_mean = class_mean / np.linalg.norm(class_mean)


    exemplar_set = []
    exemplar_features = []
    for k in range(m):

        S = np.sum(exemplar_features, axis=0)
        phi = features
        mu = class_mean
        mu_p = 1.0/(k+1) * (phi + S)
        mu_p = mu_p / np.linalg.norm(mu_p)
        #mu_p = mu_p / np.linalg.norm(mu_p)
        i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))

        exemplar_set.append(images[i])
        exemplar_features.append(features[i])
        #print('Indice scelto:{}'.format(i))
        images = np.concatenate((images[:i], images[i+1:]))
        features = np.concatenate((features[:i], features[i+1:]))

    #print(exemplar_set[:3])
    self.exemplars.append(exemplar_set)

  """
  #da cambiare completamente
  def classify(self, dataloader, transform):

        #compute the mean for each examplars
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
                features.append(feature)
            exemplar_means.append(np.mean(features))




        if exemplar_means is None:
            raise ValueError(
                "Cannot classify without built examplar means,"
            )

        if exemplar_means.shape[0] != self.num_classes:
            raise ValueError(
                "The number of examplar means ({}) is inconsistent".format(exemplar_means.shape[0])
            )


        ypred = []
        ytrue = []


        running_corrects = 0
        for inputs, targets, _ in dataloader:
            imputs = inputs

            inputs = inputs.to(DEVICE)
            #compute the feature map of the input
            features = self.feature_extractor.extract_features(inputs).data.cpu().numpy().squeeze()

            pred_labels = []

            for feature in features:
              #computing L2 distance
              #distances = torch.pow(exemplar_means - feature, 2).sum(-1)
              distances = []
              for mean in exemplar_means:
                  distances.append(np.sqrt(np.sum((mean - feature) ** 2)))
              pred_labels.append(np.argmin(distances))

            preds = np.array(pred_labels)

            running_corrects += torch.sum(torch.from_numpy(preds) == targets.data).data.item()
            print('Running corrects')
            print(running_corrects)

            ypred.extends(preds)
            ytrue.extends(targets)


        print(ypred)

        accuracy = running_corrects / float(len(dataloader.dataset))
        print(f"Test accuracy: {accuracy}")

        return np.array(ypred), np.array(ytrue)
  """
  def classify(self, x, transform):
        """Classify images by neares-means-of-exemplars
        Args:
            x: input image batch
        Returns:
            preds: Tensor of size (batch_size,)
        """
        batch_size = x.size(0)

        if self.compute_means:
            print("Computing mean of exemplars...")
            exemplar_means = []
            for P_y in self.exemplar_sets:
                features = []
                # Extract feature for each exemplar in P_y
                for ex in P_y:
                    ex = Variable(transform(Image.fromarray(ex)), volatile=True).cuda()
                    feature = self.feature_extractor(ex.unsqueeze(0))
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm() # Normalize
                    features.append(feature)
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm() # Normalize
                exemplar_means.append(mu_y)
            self.exemplar_means = exemplar_means
            self.compute_means = False
            print("Done")

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means) # (n_classes, feature_size)
        means = torch.stack([means] * batch_size) # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2) # (batch_size, feature_size, n_classes)

        feature = self.feature_extractor(x) # (batch_size, feature_size)
        for i in xrange(feature.size(0)): # Normalize
            feature.data[i] = feature.data[i] / feature.data[i].norm()
        feature = feature.unsqueeze(2) # (batch_size, feature_size, 1)
        feature = feature.expand_as(means) # (batch_size, feature_size, n_classes)

        dists = (feature - means).pow(2).sum(1).squeeze() #(batch_size, n_classes)
        _, preds = dists.min(1)

        return preds
