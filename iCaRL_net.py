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
import copy

import random

import utils

import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity

####Hyper-parameters####
LR = 2
WEIGHT_DECAY = 0.00001
BATCH_SIZE = 128
STEPDOWN_EPOCHS = [49, 63]
STEPDOWN_FACTOR = 5
NUM_EPOCHS = 70
DEVICE = 'cuda'
MOMENTUM = 0.9
########################

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

bce = nn.BCEWithLogitsLoss()
mlsm = nn.MultiLabelSoftMarginLoss()
l1 = nn.L1Loss()
mse = nn.MSELoss()

losses = {'bce': bce, 'mlsm': mlsm,'l1': l1, 'mse': mse}

class iCaRL(nn.Module):
    def __init__(self, n_classes, class_map, loss_config,lr):
        super(iCaRL, self).__init__()
        self.features_extractor = resnet32(num_classes=0)

        self.n_classes = 0
        self.n_known = 0
        self.exemplar_sets = []
        self.loss_config = loss_config
        self.lr = lr

        self.clf_loss = losses[loss_config]
        self.dist_loss = losses[loss_config]

        self.exemplar_means = []
        self.compute_means = True
        self.new_means = []
        self.class_map = class_map


    def forward(self, x):
        x = self.features_extractor(x)
        return x

    def add_classes(self, n):
        in_features = self.features_extractor.fc.in_features
        out_features = self.features_extractor.fc.out_features

        weight = copy.deepcopy(self.features_extractor.fc.weight.data)
        bias = copy.deepcopy(self.features_extractor.fc.bias.data)

        self.features_extractor.fc = nn.Linear(in_features, out_features+n)
        self.features_extractor.fc.weight.data[:out_features] = copy.deepcopy(weight)
        self.features_extractor.fc.bias.data[:out_features] = copy.deepcopy(bias)

        self.n_classes += n

    def add_exemplars(self, dataset, map_reverse):
        for y, exemplars in enumerate(self.exemplar_sets):
            dataset.append(exemplars, [map_reverse[y]]*len(exemplars))

    def update_representation(self, dataset, class_map, map_reverse, iter):

        targets = list(set(dataset.targets))
        n = len(targets)

        print('New classes:{}'.format(n))
        print('-'*30)

        self.add_exemplars(dataset, map_reverse)

        print('Datset extended to {} elements'.format(len(dataset)))

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


        self.features_extractor.to(DEVICE)

        #prev_features_ex = copy.deepcopy(self.features_extractor)
        f_ex = copy.deepcopy(self.features_extractor)
        f_ex.to(DEVICE)


        #compute previous output for training
        q = torch.zeros(len(dataset), self.n_known).to(DEVICE)
        for images, labels, indexes in loader:
            f_ex.train(False)
            images = Variable(images).to(DEVICE)
            indexes = indexes.to(DEVICE)
            g = torch.sigmoid(f_ex.forward(images))
            q[indexes] = g.data
        q = Variable(q).to(DEVICE)

        self.add_classes(n)
        
        self.features_extractor.train(True)


        optimizer = optim.SGD(self.features_extractor.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)

        i = 0

        self.features_extractor.to(DEVICE)
        for epoch in range(NUM_EPOCHS):

            if epoch in STEPDOWN_EPOCHS:
              for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/STEPDOWN_FACTOR


            self.features_extractor.train(True)
            for imgs, labels, indexes in loader:
                imgs = imgs.to(DEVICE)
                indexes = indexes.to(DEVICE)
                # We need to save labels in this way because classes are randomly shuffled at the beginning
                seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
                labels = Variable(seen_labels).to(DEVICE)
                #labels = torch.Tensor(seen_labels).to(DEVICE)
                labels_hot=torch.eye(self.n_classes)[labels]
                labels_hot = labels_hot.to(DEVICE)


                optimizer.zero_grad()
                out = self(imgs)

                if self.loss_config == 'l1' or self.loss_config == 'mse':
                    #print(out)
                    out = torch.softmax(out,dim=1)

                loss = self.clf_loss(out[:, self.n_known:], labels_hot[:, self.n_known:])

                if self.n_known > 0:

                    q_i = q[indexes]
                    dist_loss = self.dist_loss(out[:, :self.n_known], q_i[:, :self.n_known])
                    loss = (1/(iter+1))*loss + (iter/(iter+1))*dist_loss    

                loss.backward()
                optimizer.step()


            if i % 10 == 0 or i == (NUM_EPOCHS-1):
                print('Epoch {} Loss:{:.4f}'.format(i, loss.item()))
                for param_group in optimizer.param_groups:
                  print('Learning rate:{}'.format(param_group['lr']))
                print('-'*30)
            i+=1

        return


    def reduce_exemplars_set(self, m):
        for y, exemplars in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = exemplars[:m]


    @torch.no_grad()
    def construct_exemplars_set(self, images, m, random_flag=False):

        if random:
            exemplar_set = []
            indexes = random.sample(range(len(images)), m)
            for i in indexes:
                exemplar_set.append(images[i])
            self.exemplar_sets.append(exemplar_set)

        else:
            features = []

            self.features_extractor.to(DEVICE)


            self.features_extractor.train(False)
            for img in images:
                x = Variable(transform(Image.fromarray(img))).to(DEVICE)
                feature = self.features_extractor.extract_features(x.unsqueeze(0)).data.cpu().numpy()
                feature = feature / np.linalg.norm(feature)
                features.append(feature[0])

            class_mean = np.mean(features, axis=0)
            class_mean = class_mean / np.linalg.norm(class_mean)

            self.new_means.append(class_mean)

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

                if i == 0:
                    images = images[1:]
                    features = features[1:]

                elif i == len(features):
                    images = images[:-1]
                    features = features[:-1]
                else:
                    try:
                        images = np.concatenate((images[:i], images[i+1:]))
                        features = np.concatenate((features[:i], features[i+1:]))
                    except:
                        print('chosen i:{}'.format(i))


            self.exemplar_sets.append(np.array(exemplar_set))
            self.features_extractor.train(True)


    @torch.no_grad()
    def classify(self, x, classifier):

        #NME
        if classifier == 'nme' or classifier == 'nme-cosine':

            batch_size = x.size(0)
            if self.compute_means:

                exemplar_means = []

                self.features_extractor.train(False)
                for exemplars in self.exemplar_sets[:self.n_known]:
                    features = []
                    for ex in  exemplars:

                        ex = Variable(transform(Image.fromarray(ex))).to(DEVICE)
                        feature = self.features_extractor.extract_features(ex.unsqueeze(0))
                        feature = feature.squeeze()
                        feature.data = feature.data / torch.norm(feature.data, p=2)
                        features.append(feature)

                    features = torch.stack(features)
                    mu_y = features.mean(0).squeeze()
                    mu_y.data = mu_y.data / torch.norm(mu_y.data, p=2)
                    exemplar_means.append(mu_y)

                self.exemplar_means = exemplar_means
                self.exemplar_means.extend(self.new_means)
                self.compute_means = False

            exemplar_means = self.exemplar_means

            print('numero medie', len(exemplar_means))
            
            x = x.to(DEVICE)
            self.features_extractor.train(False)
            feature = self.features_extractor.extract_features(x)


            preds = []

            for feat in feature:
                measures = []
                feat = feat / torch.norm(feat, p=2)

                for mean in exemplar_means:

                    if classifier =='nme':
                        measures.append((feat - mean).pow(2).sum().squeeze().item())
                    elif classifier =='nme-cosine':
                        measures.append(cosine_similarity(feat.unsqueeze(0).cpu().numpy(), mean.unsqueeze(0).cpu().numpy()))

                if classifier =='nme':
                    preds.append(np.argmin(np.array(measures)))
                elif classifier =='nme-cosine':
                    preds.append(np.argmax(np.array(measures)))

            return preds

        #KNN
        elif classifier =='knn' or classifier ==  'svc':

            X_train, y_train = [], []

            self.features_extractor.train(False)
            for i, exemplars in enumerate(self.exemplar_sets):
                for ex in  exemplars:
                    ex = Variable(transform(Image.fromarray(ex))).to(DEVICE)
                    feature = self.features_extractor.extract_features(ex.unsqueeze(0))
                    feature = feature.squeeze()
                    feature.data = feature.data / torch.norm(feature.data, p=2)
                    X_train.append(feature.cpu().numpy())
                    y_train.append(i)

            if classifier == 'knn':
                model = KNeighborsClassifier(n_neighbors=3)
            elif classifier == 'svc':
                model = LinearSVC()

            model.fit(X_train, y_train)

            x = x.to(DEVICE)
            self.features_extractor.train(False)
            feature = self.features_extractor.extract_features(x)

            X = []

            for feat in feature:
                feat = feat / torch.norm(feat, p=2)
                X.append(feat.cpu().numpy())

            preds = model.predict(X)

            return preds


    def classify_all(self, test_dataset, map_reverse, classifier):

        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        running_corrects = 0

        for imgs, labels, _ in  test_dataloader:
            imgs = Variable(imgs).cuda()
            preds = self.classify(imgs, classifier)
            preds = [map_reverse[pred] for pred in preds]
            running_corrects += (preds == labels.numpy()).sum()
        accuracy = running_corrects / float(len(test_dataloader.dataset))
        print('Test Accuracy: {}'.format(accuracy))

        return accuracy
