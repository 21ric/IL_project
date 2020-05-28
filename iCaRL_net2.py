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

import math

#transform = transforms.Compose([transforms.ToTensor(),
#                               transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
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

@torch.no_grad()
def validate(net, val_dataloader, map_reverse):
    running_corrects_val = 0
    net.train(False)
    for inputs, labels, index in val_dataloader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        # forward
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
        running_corrects_val += (preds == labels.cpu().numpy()).sum()
        #running_corrects_val += torch.sum(preds == labels.data)

    valid_acc = running_corrects_val / float(len(val_dataloader.dataset))
    net.train(True)
    return valid_acc


class iCaRL(nn.Module):
    def __init__(self, n_classes, class_map):
        super(iCaRL, self).__init__()
        self.features_extractor = resnet32(num_classes=n_classes)

        #orch.nn.init.xavier_uniform_(self.features_extractor.fc.weight)
        #self.features_extractor.fc.bias.data.fill_(0.01)


        self.n_classes = n_classes
        self.n_known = 0
        self.exemplar_sets = []

        self.clf_loss = nn.BCEWithLogitsLoss()
        self.dist_loss = nn.BCEWithLogitsLoss()

        self.exemplar_means = []
        self.compute_means = True

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

    def add_exemplars(self, dataset, class_map, map_reverse):
        for y, exemplars in enumerate(self.exemplar_sets):
            #print(map_reverse[y])
            dataset.append(exemplars, [map_reverse[y]]*len(exemplars))

    def update_representation(self, dataset, class_map, map_reverse):
        #dataset = dataset.dataset
        targets = list(set(dataset.targets))
        n = len(targets)

        print('New classes:{}'.format(n))
        print('-'*30)

        self.add_exemplars(dataset, class_map, map_reverse)

        print('Datset extended to {} elements'.format(len(dataset)))

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        #val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)




        self.add_classes(n)

        self.features_extractor.to(DEVICE)
        f_ex = copy.deepcopy(self.features_extractor)
        f_ex.to(DEVICE)
        q = torch.zeros(len(dataset), self.n_classes).cuda()
        for images, labels, indexes in loader:
            f_ex.train(False)
            images = Variable(images).cuda()
            indexes = indexes.cuda()
            g = torch.sigmoid(f_ex.forward(images))
            #g = self.forward(images)
            q[indexes] = g.data

            #self.features_extractor.to(DEVICE)

        q = Variable(q).cuda()
        self.features_extractor.train(True)



        #self.n_classes += n
        optimizer = optim.SGD(self.features_extractor.parameters(), lr=2.0, weight_decay=0.00001, momentum=0.9)
        #optimizer = optim.Adam(self.features_extractor.parameters(), lr=0.2, weight_decay=0.00001)

        i = 0

        best_acc = -1
        best_epoch = 0

        self.features_extractor.to(DEVICE)
        cond = True
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
                if cond:
                    #print('labels', list(set(labels.tolist())))
                    cond =False
                labels_hot=torch.eye(self.n_classes)[labels]
                labels_hot = labels_hot.to(DEVICE)


                optimizer.zero_grad()
                #out = torch.sigmoid(self(imgs))
                out = self(imgs)

                #print(out[0])

                #print('out', out[0], 'labels', labels[0])

                if self.n_known <= 0:
                    loss = self.clf_loss(out, labels_hot)

                else:
                    #out = torch.sigmoid(out)
                    q_i = q[indexes]

                    target = torch.cat((q_i[:, :self.n_known], labels_hot[:, self.n_known:self.n_classes]), dim=1)
                    loss = self.dist_loss(out, target)
                    #loss += dist_loss

                loss.backward()
                optimizer.step()

            #accuracy = validate(self.features_extractor, val_loader, map_reverse)

            """
            if accuracy > best_acc:
                best_acc = accuracy
                best_epoch = epoch
                best_net = copy.deepcopy(self.state_dict())
            """
            if i % 10 == 0 or i == (NUM_EPOCHS-1):
                print('Epoch {} Loss:{:.4f}'.format(i, loss.item()))
                for param_group in optimizer.param_groups:
                  print('Learning rate:{}'.format(param_group['lr']))
                #print('Max Accuracy:{:.4f} (Epoch {})'.format(best_acc, best_epoch))
                print('-'*30)
            i+=1

        #self.load_state_dict(best_net)
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
            #del features
            #self.features_extractor.train(True)


    @torch.no_grad()
    def classify(self, x):

        batch_size = x.size(0)

        if self.compute_means:

            exemplar_means = []

            #self.features_extractor.train(False)
            #print('exset', self.exemplar_sets)
            self.features_extractor.train(False)
            for exemplars in self.exemplar_sets:
                #print('in')
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
                #print('mu_y', mu_y)

            self.exemplar_means = exemplar_means
            self.compute_means = False
        #print(self.exemplar_means)


        exemplar_means = self.exemplar_means

        """
        means = torch.stack(exemplar_means)
        means = torch.stack([means]*batch_size)
        means = means.transpose(1,2)
        """

        #self.features_extractor(DEVICE)
        x = x.to(DEVICE)
        self.features_extractor.train(False)
        #self.features_extractor.train(False)
        feature = self.features_extractor.extract_features(x)

        """
        for i in range(feature.size(0)):
            feature.data[i] = feature.data[i]/ torch.norm(feature.data[i], p=2)
        """
        """
        feature = feature.unsqueeze(2)
        feature = feature.expand_as(means)
        """

        preds = []
        a = 3
        cond = True

        for feat in feature:
            dists = []
            feat = feat / torch.norm(feat, p=2)

            if cond:
                #print(len(exemplar_means))
                cond = False


            for mean in exemplar_means:

                """
                if a>0:
                    print('mean')
                    print(mean)
                    print('feat')
                    print(feat)
                    print('dist')
                    print((feat - mean).pow(2).sum().squeeze().item())
                """

                dists.append((feat - mean).pow(2).sum().squeeze().item())

            """
            if a > 0:
                print('all dists')
                print(dists)
                a = a-1
            """


            preds.append(np.argmin(np.array(dists)))

        """
        dists = (feature - means).pow(2).sum(1).squeeze()
        _, preds = dists.min(1)
        """
        #self.features_extractor.train(True)

        #print(preds)

        return preds



    def classify_all(self, test_dataset, map_reverse):

        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        running_corrects = 0
        #self.features_extractor(DEVICE)

        for imgs, labels, _ in  test_dataloader:
            imgs = Variable(imgs).cuda()
            preds = self.classify(imgs)
            #preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
            preds = [map_reverse[pred] for pred in preds]
            running_corrects += (preds == labels.numpy()).sum()
            #running_corrects += torch.sum(preds == labels.data).data.item()
        accuracy = running_corrects / float(len(test_dataloader.dataset))
        print('Test Accuracy: {}'.format(accuracy))
