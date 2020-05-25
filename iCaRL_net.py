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
    def __init__(self, n_classes):
        super(iCaRL, self).__init__()
        self.featureture_extractor = resnet32()

        self.n_classes = n_classes
        self.n_known = 0
        self.exemplar_sets = []

        self.clf_loss = nn.CrossEntropyLoss()
        self.dist_loss = nn.BCELoss()

        self.exemplar_means = []
        self.exemplars_sets = []


    def forward(self, x):
        x = self.featureture_extractor(x)
        return x

    def add_classes(self, n):
        in_features = self.featureture_extractor.fc.in_features
        out_features = self.feature_extractor.fc.out_features
        weight = self.feature_extractor.fc.weight.data
        bias = self.feature_extractor.fc.bias.data

        self.feature_extractor.fc = nn.Linear(in_features, out_features+n)
        self.fc.weight.data[:out_features] = weight
        self.fc.bias.data[:out_features] = bias

        self.n_classes += n

    def add_exemplars(self, dataset):
        for y, exemplars in enumerate(self.exemplars_sets):
            dataset.append(exemplars, [y]*len(exemplars))

    def update_representation(self, dataset):
        targets = list(set(dataset.targets))
        n = len(targets)
        print('{} new classes'.format(n))

        self.add_exemplars(dataset)

        self.to(DEVICE)

        loader = Dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        if self.n_known > 0:
            self.train(False)
            q = torch.zeros(len(dataset), self.n_classes).cuda()
            for images, labels, indexes in loader:
                images = Variable(images).cuda()
                indexes = indexes.cuda()
                g = F.sigmoid(self.forward(images))
                q[indexes] = g.data
            q = Variable(q).cuda()
            self.train(True)

        self.add_classes(n)

        optimizer = optim.SGD(self.parameters(), lr=2.0, weight_decay=0.00001)

        for epoch in range(NUM_EPOCHS):
            i = 0
            for imgs, labels, indexes in loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                indexes = indexes.to(DEVICE)

                optimizer.zero_grad()
                out = self(imgs)

                loss = self.clf_loss(g, labels)

                if self.n_known > 0:
                    out = torch.sigmoid(out)
                    q_i = q[indexes]
                    dist_loss = sum(self.dist_loss(g[:,y], q_i[:,y]) for y in range(self.num_known))

                    loss += dist_loss

                loss.backward()
                optimizer.step()

                if i % 10 == 0:
                    print('Epoch {} Loss:{:.4f}'.format(i, loss.data[0]))
                i+=1
