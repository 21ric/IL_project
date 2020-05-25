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
def to_onehot(targets, n_classes):
    one_hots = []
    #print('len targets', len(targets))
    #print('targets', targets)
    for t in targets:
        temp = np.zeros(n_classes)
        temp[t] = 1
        one_hots.append(temp)
    #print(one_hots)
    one_hots = torch.FloatTensor(one_hots)
    #print(one_hots.size())
    return one_hots


class iCaRL(nn.Module):
    def __init__(self, n_classes):
        super(iCaRL, self).__init__()
        self.features_extractor = resnet32(num_classes=0)

        self.n_classes = n_classes
        self.n_known = 0
        self.exemplar_sets = []

        self.clf_loss = nn.BCEWithLogitsLoss()
        self.dist_loss = nn.BCEWithLogitsLoss()

        self.exemplar_means = []
        self.exemplars_sets = []


    def forward(self, x):
        x = self.features_extractor(x)
        return x

    def add_classes(self, n):
        in_features = self.features_extractor.fc.in_features
        out_features = self.features_extractor.fc.out_features
        weight = self.features_extractor.fc.weight.data
        bias = self.features_extractor.fc.bias.data

        self.features_extractor.fc = nn.Linear(in_features, out_features+n)
        self.features_extractor.fc.weight.data[:out_features] = weight
        self.features_extractor.fc.bias.data[:out_features] = bias

        self.n_classes += n

    def add_exemplars(self, dataset):
        for y, exemplars in enumerate(self.exemplars_sets):
            dataset.append(exemplars, [y]*len(exemplars))

    def update_representation(self, dataset):
        targets = list(set(dataset.targets))
        n = len(targets)
        print('{} new classes'.format(n))

        self.add_exemplars(dataset)

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        if self.n_known > 0:
            self.to(DEVICE)
            self.train(False)
            q = torch.zeros(len(dataset), self.n_classes).cuda()
            for images, labels, indexes in loader:
                images = Variable(images).cuda()
                indexes = indexes.cuda()
                #g = torch.sigmoid(self.forward(images))
                g = self.forward(images)
                q[indexes] = g.data
            q = Variable(q).cuda()
            self.train(True)

        self.add_classes(n)

        optimizer = optim.SGD(self.parameters(), lr=2.0, weight_decay=0.00001)

        self.to(DEVICE)
        i = 0
        for epoch in range(NUM_EPOCHS):

            if epoch in STEPDOWN_EPOCHS:
              for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/STEPDOWN_FACTOR


            for imgs, labels, indexes in loader:
                imgs = imgs.to(DEVICE)

                labels = to_onehot(labels, self.n_classes)
                labels = labels.to(DEVICE)
                indexes = indexes.to(DEVICE)

                optimizer.zero_grad()
                out = self(imgs)

                #print('out', out[0], 'labels', labels[0])

                loss = self.clf_loss(out, labels)

                if self.n_known > 0:
                    #out = torch.sigmoid(out)
                    q_i = q[indexes]
                    #print('g', g[:,1])
                    #print('q_i', q_i[:,1])
                    #controllare dist loss
                    print('here?')
                    print(out[:, self.n_known])
                    print(q_i)
                    #dist_loss = sum(self.dist_loss(g[:,y], q_i[:,y]) for y in range(self.n_known))
                    dist_loss = self.dist_loss(out[:,self.n_known], q_i)

                    loss += dist_loss

                loss.backward()
                optimizer.step()

            if i % 10 == 0:
                print('Epoch {} Loss:{:.4f}'.format(i, loss.item()))
                for param_group in optimizer.param_groups:
                  print('Learning rate:{}'.format(param_group['lr']))
            i+=1

    def reduce_exemplars_set(self, m):
        for y, exemplars in enumerate(self.exemplars_sets):
            self.exemplars_sets[y] = exemplars[:m]


    def construct_exemplars_set(self, images, m):

        features = []
        self.to(DEVICE)
        self.train(False)
        for img in images:
            x = Variable(transform(Image.fromarray(img))).to(DEVICE)
            feature = self.features_extractor.extract_features(x.unsqueeze(0)).data.cpu().numpy()
            feature = feature / np.linalg.norm(feature)
            features.append(feature[0])

        #print('features shape', features[0])
        #features = np.array(features)
        #print('num_features',len(features))
        class_mean = np.mean(features, axis=0)
        #print('class_mean', class_mean)
        class_mean = class_mean / np.linalg.norm(class_mean)

        exemplar_set = []
        exemplar_features = []
        for k in range(m):
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = 1.0 / (k+1)*(phi+S)
            mu_p = mu_p / np.linalg.norm(mu_p)
            i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis =1)))

            exemplar_set.append(images[i])
            exemplar_features.append(features[i])

            #print('chosen i:{}'.format(i))

            images = np.concatenate((images[:i], images[i+1:]))
            features = np.concatenate((features[:i], features[i+1:]))

        self.exemplar_sets.append(np.array(exemplar_set))


    def classify(self, x):

        batch_size = x.size(0)

        exemplar_means = []

        self.to(DEVICE)
        #print('exset', self.exemplar_sets)
        for exemplars in self.exemplar_sets:
            #print('in')
            features = []
            for ex in  exemplars:
                ex = Variable(transform(Image.fromarray(ex))).to(DEVICE)
                feature = self.features_extractor.extract_features(ex.unsqueeze(0))
                feature = feature.squeeze()
                feature.data = feature.data / feature.data.norm()
                features.append(feature)

            features = torch.stack(features)
            mu_y = features.mean(0).squeeze()
            mu_y.data = mu_y.data / mu_y.data.norm()
            exemplar_means.append(mu_y)
            #print('mu_y', mu_y)

        self.exemplar_means = exemplar_means
        #print(self.exemplar_means)
        exemplar_means = self.exemplar_means

        means = torch.stack(exemplar_means)
        means = torch.stack([means]*batch_size)
        means = means.transpose(1,2)

        self.to(DEVICE)
        x = x.to(DEVICE)

        feature = self.features_extractor.extract_features(x)
        for i in range(feature.size(0)):
            feature.data[i] = feature.data[i]/ feature.data[i].norm()
        feature = feature.unsqueeze(2)
        feature = feature.expand_as(means)


        dists = (feature - means).pow(2).sum(1).squeeze()
        _, preds = dists.min(1)

        return preds
