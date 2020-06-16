import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

import numpy as np
from PIL import Image
from resnet import resnet32
import copy
import random
import utils
import math

import time

####Hyper-parameters####
LR = 2
WEIGHT_DECAY = 0.00001
BATCH_SIZE = 128
STEPDOWN_EPOCHS = [49, 63]
STEPDOWN_FACTOR = 5
NUM_EPOCHS = 70
DEVICE = 'cuda'
MOMENTUM = 0.9
BETA = 0.8
########################

#transofrmation for exemplars
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

#possible losses
losses = { 'bce': nn.BCEWithLogitsLoss(),
           'l1': nn.L1Loss(),
           'mse': nn.MSELoss(),
           'kl': nn.KLDivLoss(reduction='batchmean'),
           'bce_sum': nn.BCEWithLogitsLoss(reduction='sum')}


#possible classifier
models = {'svc': LinearSVC(),
          'knn': KNeighborsClassifier(n_neighbors=3),
          'svc-rbf': SVC(),
          'nme': None } 
    

    
#ICARL MODEL
class iCaRL(nn.Module):
    def __init__(self, n_classes, class_map, map_reverse, clf_loss, dist_loss, classifier, lr, add_samples=False):
        super(iCaRL, self).__init__()
        self.features_extractor = resnet32(num_classes=n_classes)

        self.n_classes = 0 #number of classes at step t
        self.n_known = 0 #number of classes at step t-1
        self.exemplar_sets = []
        
        self.lr = lr
        self.loss_config = [clf_loss, dist_loss]
        
        if not add_samples:
            self.clf_loss = losses[clf_loss]
            self.dist_loss = losses[dist_loss]
        
        #reduction sum is needed when adding new samples
        else:
            self.clf_loss = losses['bce_sum']
            self.dist_loss = losses['bce_sum']

        #exemplars
        self.exemplar_means = []
        self.compute_means = True 
        self.new_means = [] #use mean of all data for new samples
        self.exemplars_per_class = 0
              
        self.class_map = class_map #needed to map real label to fake label
        self.map_reverse = map_reverse
        
        #use a different classifier
        self.train_model = True
        self.model = models[classifier]
        
        self.class_balanced_loss =False
        #choose if create new samples
        self.add_samples = add_samples

        
    def forward(self, x):
        x = self.features_extractor(x)
        return x
    
    
    
    #UPDATE REPRESENTATION
    #updating the feature extractor
    """
    def update_representation(self, dataset, iter):

        #computing number of new classes
        targets = list(set(dataset.targets))
        n = len(targets)

        print('New classes:{}'.format(n))
        print('-'*30)
            
        #adding exemplars to dataset
        self.add_exemplars(dataset)

        print('Datset extended to {} elements'.format(len(dataset)))

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        
        #incrementing number of classes
        self.add_classes(n)
        
        #obtaining previous outputs and previous network
        q, previous_net = self.get_previous_outputs(loader)
        
        
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
                #indexes = indexes.to(DEVICE)            
                seen_labels = torch.LongTensor([self.class_map[label] for label in labels.numpy()])
                labels = Variable(seen_labels).to(DEVICE)
                
                #computing one hots of labels
                labels_hot=torch.eye(self.n_classes)[labels]
                labels_hot = labels_hot.to(DEVICE)
                
                #creating new samples
                if self.add_samples and self.n_known > 0:
                      new_samples, new_targets = self.mixed_up_samples(imgs, labels_hot, labels)
                
                #zeroing the gradients
                optimizer.zero_grad()
                
                                                          
                #computing outputs of training data
                out = self(imgs)             
                
                #computing classification loss
                loss = self.clf_loss(out[:, self.n_known:], labels_hot[:, self.n_known:])
                
                if self.add_samples:
                      #reduction is sum, averaging
                      loss = loss/(len(out)*10)
                    
                    
                #DISTILLATION LOSS
                if self.n_known > 0 :
                    
                    q_i = q[indexes]

                    #creating new samples by linear combination of exemplars
                    if self.add_samples:
                     
                        
                        #computing outputs  
                        start = time.time()
                        new_out = self(new_samples)
                        
                                 
                        #classification loss with added samples
                        clf_loss = self.clf_loss(out[:, self.n_known:], labels_hot[:, self.n_known:])
                        clf_loss_new = self.clf_loss(new_out[:, self.n_known:], new_targets[:, self.n_known:])                  
                        #average loss
                        loss = (clf_loss + clf_loss_new)/((len(out)+len(new_out))*10)  
                        
                        
                        with torch.no_grad():   
                             previous_net.to(DEVICE)
                             previous_net.train(False)
                             #q_i = torch.sigmoid(previous_net(imgs))
                             new_samples.to(DEVICE)
                             q_i_new = torch.sigmoid(previous_net(new_samples)) 
                      
                        

                        #computing sum of losses
                        dist_loss = self.dist_loss(out[:, :self.n_known], q_i[:, :self.n_known])
                        dist_loss_new = self.dist_loss(new_out[:, :self.n_known], q_i_new[:, :self.n_known])
                        
                        #average
                        dist_loss = (dist_loss+ dist_loss_new)/((len(out)+len(new_out))*self.n_known)

                    else:
                        #modifying outputs to match losses requireents
                        out = F.softmax(out, dim=1) if self.dist_loss in ['mse', 'l1'] else \
                              F.log_softmax(out, dim = 1) if self.dist_loss == 'kl' \
                              else out
                                 
                        dist_loss = self.dist_loss(out[:, :self.n_known], q_i[:, :self.n_known])

                    #loss = (1/(iter+1))*loss + (iter/(iter+1))*dist_loss
                    loss = (loss + iter * dist_loss)/(iter + 1)

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
    """
    def update_representation(self, dataset, iter):
        #computing number of new classes
        targets = list(set(dataset.targets))
        n = len(targets)
        print('New classes:{}'.format(n))
        print('-'*30)
            
        
        
        #adding exemplars to dataset
        self.add_exemplars(dataset)
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
                seen_labels = torch.LongTensor([self.class_map[label] for label in labels.numpy()])
                labels = Variable(seen_labels).to(DEVICE)
                
                #computing one hots of labels
                labels_hot=torch.eye(self.n_classes)[labels]
                labels_hot = labels_hot.to(DEVICE)
                if self.add_samples:
                    
                    if iter !=0:
                         #mix up augmentation
                         exemplars = imgs[(labels < self.n_known)]
                         ex_labels = labels_hot[(labels < self.n_known)]
                         mixed_up_points = []
                         mixed_up_targets = []

                         
                         for j in range(128 - len(exemplars)):
                             i1, i2 = np.random.randint(0, len(exemplars)), np.random.randint(0, len(exemplars))
                             
                             w = 0.6
                             new_point = w*exemplars[i1]+(1-w)*exemplars[i2]
                             new_target = w*ex_labels[i1]+(1-w)*ex_labels[i2]

                             mixed_up_points.append(new_point)
                             mixed_up_targets.append(new_target)
                                 
                         mixed_up_points = torch.stack(mixed_up_points)
                         mixed_up_targets = torch.stack(mixed_up_targets)
                    
                        
                #zeroing the gradients
                optimizer.zero_grad()
                
                #computing outputs
                out = self(imgs)
                
                if self.add_samples and iter !=0:              
                    mixed_out = self(mixed_up_points)
                
                #computing classification loss
                if self.class_balanced_loss or self.add_samples:
                                                            
                    
                    ex_out = out[(labels < self.n_known)] #masking: taking only outputs of images of new classes
                    sample_out = out[(labels >= self.n_known)] #masking: taking only outputs of images of old classes
                    
                    labels_ex = labels_hot[(labels < self.n_known)] #masking: taking only true labels of images of new classes
                    labels_sample = labels_hot[(labels >= self.n_known)] #masking: taking only true labels of images of old classes
                    
                    
                    
                    
                    if self.class_balanced_loss:
                        #taking coefficients for new images (coeff_new), exemplars (coeff_old)
                        coeff_new, coeff_old = get_balanced_coefficients(BETA, card_new=500,num_new_classes=(self.n_classes-self.n_known),num_old_classes=self.n_known, i=iter, card_old=self.exemplars_per_class)
                        #coeff_new, coeff_old = 1, 1
                        
                    else:
                        #coeff_new , coeff_old = 1, (500/self.exemplars_per_class) * (iter/(iter+1)) if self.exemplars_per_class else 1
                        coeff_new , coeff_old = 1, 1
                        
                    clf_loss_ex =  bce_sum(ex_out[:, self.n_known:], labels_ex[:, self.n_known:]) #calculating clf loss on exemplars
                    clf_loss_sample =  bce_sum(sample_out[:, self.n_known:], labels_sample[:, self.n_known:]) #calculating clf loss on new images
                    if iter !=0:
                        clf_loss_mixedup = bce_sum(mixed_out[:, self.n_known:], mixed_up_targets[:, self.n_known:])
                    
                    #loss_ex = coeff_old * bce_sum(ex_out, labels_ex)
                    #loss_sample = coeff_new * bce_sum(sample_out, labels_sample)
                    
                    #print('loss ex', loss_ex)
                    #print('loss sample', loss_sample)
                    
                    
                    
                    if self.class_balanced_loss:
                        #loss = loss_sample/(len(sample)*10)
                        loss = (clf_loss_ex + clf_loss_sample)/(len(out)*10)
                        
                    else:
                        if iter != 0:
                            loss = (clf_loss_ex + clf_loss_sample + clf_loss_mixedup)/((len(ex_out)+len(sample_out)+len(mixed_out))*10)
                        else:
                            loss = (clf_loss_ex + clf_loss_sample )/((len(ex_out)+len(sample_out))*10)
                        #loss = loss_sample/(len(sample_out)*10)
                        
                else:
                    loss = self.clf_loss(out[:, self.n_known:], labels_hot[:, self.n_known:])
                #computing distillation loss
                if self.n_known > 0 :
                    if self.class_balanced_loss or self.add_samples:
                        
                        f_ex.to(DEVICE)
                        f_ex.train(False)
                        
                        q_i_mixed = torch.sigmoid(f_ex(mixed_up_points))
                        
                        q_i = q[indexes]
                        
                        
                        q_i_ex = q_i[(labels < self.n_known)]
                        q_i_sample = q_i[(labels >= self.n_known)]
                        #q_i_sample = torch.zeros(len(q_i_sample), self.n_known).to(DEVICE)
                        
                        
                        dist_loss_ex =  coeff_old * bce_sum(ex_out[:, :self.n_known], q_i_ex[:, :self.n_known])
                        dist_loss_sample = coeff_new * bce_sum(sample_out[:, :self.n_known], q_i_sample[:, :self.n_known])
                        
                        dist_loss_mixed = bce_sum(mixed_out[:, :self.n_known], q_i_mixed[:, :self.n_known])
                        
                        if self.class_balanced_loss:
                            dist_loss = (dist_loss_ex + dist_loss_sample)/(len(out)*(self.n_known))
                            #dist_loss = loss_ex/(len(ex_out)*(self.n_known))
                            
                        else:
                            #dist_loss = loss_ex/(len(ex_out)*(self.n_known))
                            dist_loss = (dist_loss_ex + dist_loss_sample + dist_loss_mixed)/((len(ex_out)*coeff_old+len(sample_out)+len(mixed_out))*self.n_known)
                        
                        clf_contr, dist_contr = (1/(iter+1))*loss , (iter/(iter+1))*dist_loss
                        loss = (1/(iter+1))*loss + (iter/(iter+1))*dist_loss
                        
                        #loss = loss + 2*dist_loss
                        
                        
                    else:
                        out = modify_output_for_loss(self.loss_config, out) # Change logits for L1, MSE, KL
                        q_i = q[indexes]
                        dist_loss = self.dist_loss(out[:, :self.n_known], q_i[:, :self.n_known])
                        clf_contr, dist_contr = (1/(iter+1))*loss , (iter/(iter+1))*dist_loss
                        loss = (1/(iter+1))*loss + (iter/(iter+1))*dist_loss
                
                #backward pass()
                loss.backward()
                optimizer.step()
                      
            if i % 10 == 0 or i == (NUM_EPOCHS-1):
                print('Epoch {} Loss:{:.4f}'.format(i, loss.item()))
                for param_group in optimizer.param_groups:
                  print('Learning rate:{}'.format(param_group['lr']))
                  
                  if iter!= 0:
                    print('dist loss', dist_contr)
                    print('clf loss', clf_contr)
                print('-'*30)
            i+=1
        return
    

    #INCREMENT NUMBER OF CLASSES
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
        
        
        

    #ADD EXEMPLARS TO DATASET
    def add_exemplars(self, dataset):
        for y, exemplars in enumerate(self.exemplar_sets):
            dataset.append(exemplars, [self.map_reverse[y]]*len(exemplars))
            
            
    
    #GET PREVIOUS OUTPUTS
    #get output of previous step network, and previous network
    @torch.no_grad()
    def get_previous_outputs(self, loader):
        
        previous_net = copy.deepcopy(self.features_extractor)
        previous_net.to(DEVICE)
        q = torch.zeros(len(loader.dataset), self.n_classes).to(DEVICE)
        for images, labels, indexes in loader:
            previous_net.train(False)
            images = Variable(images).to(DEVICE)
            indexes = indexes.to(DEVICE)
            g = previous_net.forward(images)
            if self.loss_config == 'bce':
                g = torch.sigmoid(g)
            else: 
                g = F.softmax(g,dim=1)
            q[indexes] = g.data
        q = Variable(q).to(DEVICE)
        
        return q, previous_net
            
  
        
    #MIXED UP SAMPLES
    #creating samples by combining samples
    def mixed_up_samples(self, imgs, labels_hot, labels):
        #mix up augmentation
                        
        #dividing exemplars from new images      
        exemplars, ex_labels, samples, samples_labels = self.separate_exemplars(imgs, labels_hot, labels)
        
        new_samples = []
        new_targets = []

        #creating 2*BATCH_SIZE new samples
        for _ in range(128):
            #indexes of 2 exemplars
            i1, i2 = np.random.randint(0, len(exemplars)), np.random.randint(0, len(exemplars))
            #j1, j2 = np.random.randint(0, len(samples)), np.random.randint(0, len(exemplars))
            #weight of combination
            w = 0.1
            #creating new sample
            new_sample1, new_target1 = w*exemplars[i1]+(1-w)*exemplars[i2], w*ex_labels[i1]+(1-w)*ex_labels[i2]
            #new_sample2, new_target2 = w*samples[j1]+(1-w)*exemplars[j2], w*samples_labels[j1]+(1-w)*ex_labels[j2]
       
            new_samples.extend([new_sample1])
            new_targets.extend([new_target1])

        #creating tensor from list of tensors
        new_samples = torch.stack(new_samples)
        new_targets = torch.stack(new_targets)
        
        return new_samples, new_targets
    
    
    #SEPARATE EXEMPLARS  
    #separating exemplars from new data
    def separate_exemplars(self, imgs, labels_hot, labels):
        
        exemplars, ex_labels = imgs[(labels < self.n_known)], labels_hot[(labels < self.n_known)]
        samples, samples_labels = imgs[(labels > self.n_known)], labels_hot[(labels >= self.n_known)]
        
        return exemplars, ex_labels, samples, samples_labels
    
    
    
    #REDUCE EXEMPLARS
    def reduce_exemplars_set(self, m):  
        
        #reducing by discarding last elements
        for y, exemplars in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = exemplars[:m]
        
    
                      
    #CONSTRUCT EXEMPLARS
    #construct exemplars set. if recompute=True we are creating a new exemplar set strating from a previous one
    @torch.no_grad()
    def construct_exemplars_set(self, images, m, random_flag=False):
        
        #computing features from images and computing mean of features
        features, class_mean = self.get_features_and_mean(images)
        
        #for new classes use mean on all data available
        self.new_means.append(class_mean)
        
        #construct exemeplars by random selection
        if random_flag:
            self.construct_random_exemplars(images, m)
        
        #construct exemplar set by herding
        else:
            self.construct_exemplars(images, m, features, class_mean)
            
            
  

    #HERDING       
    #method for constructin exemplars with herding  
    @torch.no_grad()
    def construct_exemplars(self, images, m, features, class_mean):
        
        self.features_extractor.train(False)
        
        exemplar_set = []
        exemplar_features = []
        
        for k in range(m):
            S = np.sum(exemplar_features, axis=0)
            mu = 1.0 / (k+1)*(features+S)
            mu = mu / np.linalg.norm(mu) #l2 norm
            i = np.argmin(np.sqrt(np.sum((class_mean - mu) ** 2, axis =1)))

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
            
 

    #RANDOM EXEMPLARS
    #method to construct random exemplars
    def construct_random_exemplars(self, images, m):
        exemplar_set = []
        indexes = random.sample(range(len(images)), m)
        for i in indexes:
            exemplar_set.append(images[i])
        self.exemplar_sets.append(exemplar_set)
        
        
    
    #GET FEATURES AND MEAN OF IMAGES
    @torch.no_grad()
    def get_features_and_mean(self, images, tensor_flag=False):
        features = []
        self.features_extractor.to(DEVICE)
        self.features_extractor.train(False)
        for img in images:
            
            if not tensor_flag:
                x = Variable(transform(Image.fromarray(img))).to(DEVICE)
                
            else:
                x = img
                #feature = self.features_extractor.extract_features(x).data.cpu().numpy()
            
            feature = self.features_extractor.extract_features(x.unsqueeze(0)).data.cpu().numpy()
            feature = feature / np.linalg.norm(feature) #l2 norm
            features.append(feature[0])

        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean) #l2 norm
        
        return features, class_mean
    
    
    
    #COMPUTE MEAN OF EXEMPLARS
    @torch.no_grad()
    def compute_exemplars_mean(self):
        
        exemplar_means = []
        for exemplars in self.exemplar_sets[:self.n_known]:
            _, mean = self.get_features_and_mean(exemplars)
            exemplar_means.append(mean)
        
        self.exemplar_means = exemplar_means
        self.exemplar_means.extend(self.new_means)
        
           
        
    #CLASSIFICATION
    @torch.no_grad()
    def classify(self, x, classifier, train_dataset=None):

        #Using NME as classifier
        if self.model is None:
            
            #computing mean only if first iteration
            if self.compute_means:
                
                self.compute_exemplars_mean()  
                self.compute_means = False 

            exemplar_means = self.exemplar_means
            
            preds = []
            
            #computing features of images to be classified
            features, _ = self.get_features_and_mean(x, tensor_flag=True)
                    
            for feat in features:
                measures = []
                #feat = feat / torch.norm(feat, p=2) #l2 norm

                #print('computing distance')
                #computing l2 distance with all class means
                for mean in exemplar_means:
                    measures.append(np.sqrt(np.sum((feat - mean) ** 2)))

                #chosing closest mean label as prediction
                preds.append(np.argmin(np.array(measures)))
                
            return preds

        # Using KNN, SVC, SVC-rbf as classifier
        else:

            if self.train_model:
                X_train, y_train = [], []
                
                #computing features on exemplars to create X_train, y_train
                for i, exemplars in enumerate(self.exemplar_sets):
                    features, _ = self.get_features_and_mean(exemplars)
                    X_train.extend(features)
                    y_train.extend([i]*len(features))
                  
                #fitting the model
                self.model.fit(X_train, y_train)
                self.train_model = False #avoid refitting the model
            
            features, _ = self.get_features_and_mean(x, tensor_flag=True)
            X = features

            #getting predictions
            preds = self.model.predict(X)
            
            return preds

    
    
    #CLASSIFY ALL BATCHES OF A DATASET
    def classify_all(self, test_dataset, classifier, train_dataset=None):

        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        running_corrects = 0

        for imgs, labels, _ in  test_dataloader:
            imgs = Variable(imgs).cuda()
            preds = self.classify(imgs, classifier, train_dataset=train_dataset)
            
            #mapping back fake lable to true label
            preds = [self.map_reverse[pred] for pred in preds]
            
            #computing accuracy
            running_corrects += (preds == labels.numpy()).sum()
            
        accuracy = running_corrects / float(len(test_dataloader.dataset))
        print('Test Accuracy: {}'.format(accuracy))

        return accuracy
