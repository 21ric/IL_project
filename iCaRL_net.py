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
from sklearn.neural_network import MLPClassifier

####Hyper-parameters####
LR = 2
WEIGHT_DECAY = 0.00001
BATCH_SIZE = 128
STEPDOWN_EPOCHS = [49, 63]
STEPDOWN_FACTOR = 5
NUM_EPOCHS = 2
DEVICE = 'cuda'
MOMENTUM = 0.9
########################

#transofrmation for exemplars
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


#different combination of classification/distillation losses
bce = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()
mse = nn.MSELoss()
bce_sum = nn.BCEWithLogitsLoss(reduction='sum')
kl = nn.KLDivLoss(reduction='batchmean')

losses = {'bce': [bce, bce], 'kl': [bce,kl],'l1': [bce, l1], 'mse': [bce,mse]}



#define function to apply to network outputs
def modify_output_for_loss(loss_name, output):        
    #BCEWithLogits doesn't need to apply sigmoid func
    if loss_name == "bce":
        return output

    # L1 loss and MSE loss need input to be softmax
    if loss_name in ["mse", "l1"]:
        return F.softmax(output, dim=1)

    # KL loss needs input to be log-softmax
    if loss_name == "kl":
        return F.log_softmax(output, dim=1)
    

    
class iCaRL(nn.Module):
    def __init__(self, n_classes, class_map, loss_config, lr, new_extractor=False):
        super(iCaRL, self).__init__()
        self.features_extractor = resnet32(num_classes=n_classes)

        self.n_classes = 0 #number of seen classes
        self.n_known = 0 #number of known classes before new training
        self.exemplar_sets = []
        self.loss_config = loss_config
        self.lr = lr

        self.clf_loss = losses[loss_config][0]
        self.dist_loss = losses[loss_config][1]

        self.exemplar_means = []
        self.compute_means = True
        self.new_means = []
        self.class_map = class_map #needed to map real label to fake label
        
        self.new_extractor = new_extractor
        

    #forward pass
    def forward(self, x):
        x = self.features_extractor(x)
        return x
    
    
    

    #incrementing number of classes
    def add_classes(self, n):
        in_features = self.features_extractor.fc.in_features
        out_features = self.features_extractor.fc.out_features
        
        #copying old weights
        weight = copy.deepcopy(self.features_extractor.fc.weight.data)
        bias = copy.deepcopy(self.features_extractor.fc.bias.data)
        self.features_extractor.fc = nn.Linear(in_features, out_features+n)
        self.features_extractor.fc.weight.data[:out_features] = copy.deepcopy(weight)
        self.features_extractor.fc.bias.data[:out_features] = copy.deepcopy(bias)

        #incrementing number of seen classes
        self.n_classes += n
        
        
        

    #extending dataset with exemplars
    def add_exemplars(self, dataset, map_reverse):
        for y, exemplars in enumerate(self.exemplar_sets):
            dataset.append(exemplars, [map_reverse[y]]*len(exemplars))
            
            
            
         
    
    #updating representation
    def update_representation(self, dataset, class_map, map_reverse, iter):

        #computing number of new classes
        targets = list(set(dataset.targets))
        n = len(targets)

        print('New classes:{}'.format(n))
        print('-'*30)
        
        #training a new network for new classes
        if self.new_extractor and iter != 0:
            print('Training a new network ...')
            print('-'*30)
            new_extractor = resnet32(num_classes=(n))          
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            optimizer = optim.SGD(new_extractor.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
            
            #training phase
            i = 0
            new_extractor.to(DEVICE)
            for epoch in range(NUM_EPOCHS):
                #reducing learning 
                if epoch in STEPDOWN_EPOCHS:
                  for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/STEPDOWN_FACTOR


                new_extractor.train(True)
                for imgs, labels, indexes in loader:
                    imgs = imgs.to(DEVICE)
                    indexes = indexes.to(DEVICE)            
                    seen_labels = torch.LongTensor([class_map[label]-10*iter for label in labels.numpy()])
                    labels = Variable(seen_labels).to(DEVICE)

                    #computing one hots of labels
                    labels_hot=torch.eye(n)[labels]
                    labels_hot = labels_hot.to(DEVICE)

                    #zeroing the gradients
                    optimizer.zero_grad()

                    #computing outputs
                    out = new_extractor(imgs)

                    #computing loss
                    loss = self.clf_loss(out, labels_hot)

                    #backward pass()
                    loss.backward()
                    optimizer.step()


                if i % 10 == 0 or i == (NUM_EPOCHS-1):
                    print('Epoch {} Loss:{:.4f}'.format(i, loss.item()))
                    for param_group in optimizer.param_groups:
                      print('Learning rate:{}'.format(param_group['lr']))
                    print('-'*30)
                i+=1
            
            #storing outputs of this network
            r = torch.zeros(len(dataset), n).to(DEVICE)
            for images, labels, indexes in loader:
                new_extractor.train(False)
                images = Variable(images).to(DEVICE)
                indexes = indexes.to(DEVICE)
                g = new_extractor.forward(images)               
                g = torch.sigmoid(g)
                r[indexes] = g.data
                
            r = Variable(r).to(DEVICE)
            
        
        
        #adding exemplars to dataset
        self.add_exemplars(dataset, map_reverse)

        print('Datset extended to {} elements'.format(len(dataset)))

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        #incrementing number of classes
        self.add_classes(n)

        #storing outputs of previous network
        self.features_extractor.to(DEVICE)
        f_ex = copy.deepcopy(self.features_extractor)
        f_ex.to(DEVICE)
        q = torch.zeros(len(dataset), self.n_classes).to(DEVICE)
        for images, labels, indexes in loader:
            f_ex.train(False)
            images = Variable(images).to(DEVICE)
            indexes = indexes.to(DEVICE)
            g = f_ex.forward(images)
            if self.loss_config == 'bce':
                g = torch.sigmoid(g)
            else: 
                g = F.softmax(g,dim=1)
            q[indexes] = g.data
        q = Variable(q).to(DEVICE)
        self.features_extractor.train(True)

        #defining optimizer and resetting learning rate
        optimizer = optim.SGD(self.features_extractor.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)

        #training phase
        i = 0
        self.features_extractor.to(DEVICE)
        for epoch in range(NUM_EPOCHS):
            
            #reducing learning 
            if epoch in STEPDOWN_EPOCHS:
              for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/STEPDOWN_FACTOR


            self.features_extractor.train(True)
            for imgs, labels, indexes in loader:
                imgs = imgs.to(DEVICE)
                indexes = indexes.to(DEVICE)            
                seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
                labels = Variable(seen_labels).to(DEVICE)
                
                #computing one hots of labels
                labels_hot=torch.eye(self.n_classes)[labels]
                labels_hot = labels_hot.to(DEVICE)

                #zeroing the gradients
                optimizer.zero_grad()
                
                #computing outputs
                out = self(imgs)
                
                #computing loss
                if self.new_extractor:
                    loss = self.clf_loss(out, labels_hot)
                else:
                    loss = self.clf_loss(out[:, self.n_known:], labels_hot[:, self.n_known:])

                #computing distillation loss
                if self.n_known > 0:
                    if self.new_extractor:
                        f_ex.train(False)
                        new_extractor.train(False)
                        self.features_extractor.train(False)
                        
                        f_ex.to(DEVICE)
                        new_extractor.to(DEVICE)
                        self.features_extractor.to(DEVICE)
                        
                        exemplars = imgs[~(labels < self.n_known)]
                        new_samples = imgs[~(labels >= self.n_known)]
                        
                        
                        ex_out = torch.sigmoid(self.features_extractor(exemplars))
                        sample_out = torch.sigmoid(self.features_extractor(new_samples))
                        
                        q_i = q[indexes]
                        q_i = q_i[~(labels < self.n_known)]
                        r_i = r[indexes]
                        r_i = r_i[~(labels >= self.n_known)]
                        
                        ex_loss = bce_sum(ex_out[:, :self.n_known], q_i[:, :self.n_known])
                        sample_loss = bce_sum(sample_out[:, self.n_known:], r_i)
                        
                        tot_loss = (ex_loss + sample_loss)/(len(exemplars)+len(new_features))
                        
                        loss = 0.5*loss + 0.5*tot_loss
                        
                        self.features_extractor.train(True)
                        
                        
                    else:
                        out = modify_output_for_loss(self.loss_config, out) # Change logits for L1, MSE, KL
                        q_i = q[indexes]
                        dist_loss = self.dist_loss(out[:, :self.n_known], q_i[:, :self.n_known])
                        loss = (1/(iter+1))*loss + (iter/(iter+1))*dist_loss
                

                #backward pass()
                loss.backward()
                optimizer.step()


            if i % 10 == 0 or i == (NUM_EPOCHS-1):
                print('Epoch {} Loss:{:.4f}'.format(i, loss.item()))
                for param_group in optimizer.param_groups:
                  print('Learning rate:{}'.format(param_group['lr']))
                print('-'*30)
            i+=1

        return
    
    
    

    
    #reduce exemplars lists
    @torch.no_grad()
    def reduce_exemplars_set(self, m):        
        #reducing by discarding last elements
        for y, exemplars in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = exemplars[:m]
                
                      

    #construct exemplars set. if recompute=True we are creating a new exemplar set strating from a previous one
    @torch.no_grad()
    def construct_exemplars_set(self, images, m, random_flag=False):
        
        #computing features from images and computing mean of features
        features, class_mean = self.get_features_and_mean(images)
        
        #facing new classes, use as class mean, mean on all data
        self.new_means.append(class_mean)
        
        #construct exemeplars by random selection
        if random_flag:
            self.construct_random_exemplars(images, m)
        
        #construct exemplar set
        else:
            self.construct_exemplars(images, m, features, class_mean)
            
            
           
            
    #method for constructin exemplars with herding  
    @torch.no_grad()
    def construct_exemplars(self, images, m, features, class_mean):
        
        self.features_extractor.train(False)
        exemplar_set = []
        exemplar_features = []
        for k in range(m):
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = 1.0 / (k+1)*(phi+S)
            mu_p = mu_p / np.linalg.norm(mu_p) #l2 norm
            i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis =1)))

            exemplar_set.append(images[i])
            exemplar_features.append(features[i])

            #removing chosen image from candidates, avoiding duplicates
            if i == 0:
                images = images[1:]
                features = features[1:]

            elif i == (len(features)-1):
                images = images[:-1]
                features = features[:-1]
            else:
                try:
                    images = np.concatenate((images[:i], images[i+1:]))
                    features = np.concatenate((features[:i], features[i+1:]))
                except:
                    print('chosen i:{}'.format(i))

        #adding exemplars set
        self.exemplar_sets.append(np.array(exemplar_set))
        self.features_extractor.train(True)
            
            
    
    #method to construct random exemplars
    def construct_random_exemplars(self, images, m):
        exemplar_set = []
        indexes = random.sample(range(len(images)), m)
        for i in indexes:
            exemplar_set.append(images[i])
        self.exemplar_sets.append(exemplar_set)
        
        
    
    #method to extract features from images and computing mean on feature
    @torch.no_grad()
    def get_features_and_mean(self, images):
        features = []
        self.features_extractor.to(DEVICE)
        self.features_extractor.train(False)
        for img in images:
            x = Variable(transform(Image.fromarray(img))).to(DEVICE)
            feature = self.features_extractor.extract_features(x.unsqueeze(0)).data.cpu().numpy()
            feature = feature / np.linalg.norm(feature) #l2 norm
            features.append(feature[0])

        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean) #l2 norm
        
        return features, class_mean
    
    #method to compute means of exemplars
    @torch.no_grad()
    def compute_exemplars_mean(self):
        
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
            mu_y.data = mu_y.data / torch.norm(mu_y.data, p=2) #l2 norm
            exemplar_means.append(mu_y.cpu())

        self.exemplar_means = exemplar_means
        self.exemplar_means.extend(self.new_means)
        

    #classification method
    @torch.no_grad()
    def classify(self, x, classifier):

        #Using NME as classifier
        if classifier == 'nme':
            
            #computing mean only if first iteration
            if self.compute_means:
                self.compute_exemplars_mean()
            self.compute_means = False 

            exemplar_means = self.exemplar_means
            
            preds = []
            
            #computing features of images to be classified
            x = x.to(DEVICE)
            self.features_extractor.train(False)
            feature = self.features_extractor.extract_features(x)
            for feat in feature:
                measures = []
                feat = feat / torch.norm(feat, p=2) #l2 norm

                #computing l2 distance with all class means
                for mean in exemplar_means:
                    measures.append((feat.cpu() - mean).pow(2).sum().squeeze().item())

                #chosing closest mean label as prediction
                preds.append(np.argmin(np.array(measures)))
                
            return preds

        # Using KNN, SVC, 3-layers MLP as classifier
        elif classifier == 'knn' or classifier == 'svc' or classifier == 'mlp':

            X_train, y_train = [], []

            #computing features on exemplars to create X_train, y_train
            self.features_extractor.train(False)
            for i, exemplars in enumerate(self.exemplar_sets):
                for ex in  exemplars:
                    ex = Variable(transform(Image.fromarray(ex))).to(DEVICE)
                    feature = self.features_extractor.extract_features(ex.unsqueeze(0))
                    feature = feature.squeeze()
                    feature.data = feature.data / torch.norm(feature.data, p=2)
                    X_train.append(feature.cpu().numpy())
                    y_train.append(i)

            #choice of the model
            if classifier == 'knn':
                model = KNeighborsClassifier(n_neighbors=3)
            elif classifier == 'svc':
                model = LinearSVC()
            elif classifier == 'mlp':
                model = MLPClassifier(hidden_layer_sizes=(100,50,25), random_state=1)
            
            #fitting the model
            model.fit(X_train, y_train)

            #computing features of images to be classified
            x = x.to(DEVICE)
            self.features_extractor.train(False)
            feature = self.features_extractor.extract_features(x)
            X = []
            
            #l2 normalization
            for feat in feature:
                feat = feat / torch.norm(feat, p=2)
                X.append(feat.cpu().numpy())
            
            #getting predictions
            preds = model.predict(X)

            return preds

    #method to classify all batches of the test dataloader
    def classify_all(self, test_dataset, map_reverse, classifier):

        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        running_corrects = 0

        for imgs, labels, _ in  test_dataloader:
            imgs = Variable(imgs).cuda()
            preds = self.classify(imgs, classifier)
            
            #mapping back fake lable to true label
            preds = [map_reverse[pred] for pred in preds]
            
            #computing accuracy
            running_corrects += (preds == labels.numpy()).sum()
            
        accuracy = running_corrects / float(len(test_dataloader.dataset))
        print('Test Accuracy: {}'.format(accuracy))

        return accuracy
