import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image
from resnet import resnet32
import copy
import random


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA


####Hyper-parameters####
LR = 2
WEIGHT_DECAY = 0.00001
BATCH_SIZE = 128
STEPDOWN_EPOCHS = [49, 63]
STEPDOWN_FACTOR = 5
NUM_EPOCHS = 70
DEVICE = 'cuda'
MOMENTUM = 0.9
NUM_EPOCHS_RETRAIN = 30
BETA = 0.8
########################



#transofrmation for exemplars
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

#different combination of classification/distillation losses
bce = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()
mse = nn.MSELoss()
bce_sum = nn.BCEWithLogitsLoss(reduction='sum')
kl = nn.KLDivLoss()
ce = nn.CrossEntropyLoss()

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
    def __init__(self, n_classes, class_map, map_reverse, loss_config, lr, mix_up=False):
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
        self.map_reverse = map_reverse
        
        self.mix_up = mix_up
        self.exemplars_per_class = 0
        self.pca = None
        self.train_model = True
        self.model = None
        
        self.prev_net = None
        
    
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
            

        #adding exemplars to dataset
        self.add_exemplars(dataset, map_reverse)
        print('Datset extended to {} elements'.format(len(dataset)))
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last = True)
        
        #incrementing number of classes
        self.add_classes(n)
        
        #storing outputs of previous network
        self.features_extractor.to(DEVICE)
        f_ex = copy.deepcopy(self.features_extractor)
        f_ex.to(DEVICE)
        # save the instance of the model in attributes 
        self.prev_net = f_ex
       
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
            
            
            train_loss = 0.0
            
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

                #out = modify_output_for_loss(self.loss_config, out) # Change logits for L1, MSE, KL
   
                
                if self.mix_up and self.n_known > 0:

                    #mix up augmentation
                    exemplars = imgs[(labels < self.n_known)]
                    ex_labels = labels_hot[(labels < self.n_known)]
                    mixed_up_points = []
                    mixed_up_targets = []

                    #for i in range(128 - len(exemplars)):
                    for j in range(20):
                        i1, i2 = np.random.randint(0, len(exemplars)), np.random.randint(0, len(exemplars))

                        w=0.4
                        new_point = w*exemplars[i1]+(1-w)*exemplars[i2]
                        new_target = w*ex_labels[i1]+(1-w)*ex_labels[i2]

                        new_point = exemplars[i1]
                        new_target = ex_labels[i1]

                        mixed_up_points.append(new_point)
                        mixed_up_targets.append(new_target)
                        
                    mixed_up_points = torch.stack(mixed_up_points)
                    mixed_up_targets = torch.stack(mixed_up_targets)
                    #print('len mixed up', len(mixed_up_points))
                    
                    clf_loss = bce_sum(out[:, self.n_known:], labels_hot[:, self.n_known:])

                    
                    mixed_out = self(mixed_up_points)
                    clf_loss_mixedup = bce_sum(mixed_out[:, self.n_known:], mixed_up_targets[:, self.n_known:])
                    loss = (clf_loss+clf_loss_mixedup)/((len(out)+len(mixed_up_points))*10)
   
                
                else:
                    loss = self.clf_loss(out[:, self.n_known:], labels_hot[:, self.n_known:])
                
                
                #DISTILLATION LOSS
                if self.n_known > 0 :
                    

                    if self.mix_up:
                        
                        f_ex.to(DEVICE)
                        f_ex.train(False)
                        q_i_mixed = torch.sigmoid(f_ex(mixed_up_points))
                        q_i = q[indexes]

                        dist_loss = bce_sum(out[:, :self.n_known],q_i[:, :self.n_known])
                        dist_loss_mixed = bce_sum(mixed_out[:, :self.n_known], q_i_mixed[:, :self.n_known])

                        dist_loss =(dist_loss+dist_loss_mixed)/((len(out)+len(mixed_up_points))*self.n_known)

                        loss = (1/(iter+1))*loss + (iter/(iter+1))*dist_loss
                        
                        
                    else:
                        out = modify_output_for_loss(self.loss_config, out) # Change logits for L1, MSE, KL
                        q_i = q[indexes]
                        dist_loss = self.dist_loss(out[:, :self.n_known], q_i[:, :self.n_known])
                        
                        loss = (1/(iter+1))*loss + (iter/(iter+1))*dist_loss

                    
                    q_i = q[indexes]
                    dist_loss = self.dist_loss(out[:, :self.n_known], q_i[:, :self.n_known])
                    
                    loss = (1/(iter+1))*loss + (iter/(iter+1))*dist_loss

                   
                
                train_loss += loss.item() * imgs.size(0) 
                        
                
                #backward pass()
                loss.backward()
                optimizer.step()
            
            train_loss = train_loss / len(loader.dataset)
                
                
            if i % 10 == 0 or i == (NUM_EPOCHS-1):
                print('Epoch {} Loss:{:.4f}'.format(i, train_loss))
                for param_group in optimizer.param_groups:
                  print('Learning rate:{}'.format(param_group['lr']))
                  
                print('-'*30)
            i+=1
        return
    
    #
    
    # perform another training, only on exemplars.
    def train_on_exemplars(self, class_map, map_reverse, iter):
    
        exemplars_list = []
        labels = []

        for y, exemplars in enumerate(self.exemplar_sets):
            for ex in exemplars:
                exemplars_list.append(np.array(transform(Image.fromarray(ex))))
                labels.append(map_reverse[y])
        
        exemplars_list = torch.Tensor(exemplars_list) # transform to torch tensor
        labels = torch.Tensor(labels)
        print('Creating ex. dataloader')
        
        # Create Exemplars' Dataloader
        dataset = TensorDataset(exemplars_list,labels) # create your datset
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last = True)
        
        # Check loader length
        print(len(loader.dataset))
        
        # Save an instance of the very last model (the one trained with distillation)
        f_ex = copy.deepcopy(self.features_extractor)
        
        # Save the previous model, the one referring to the previous incremental step
        f_prev_net = self.prev_net
        f_prev_net.train(False)
        f_prev_net.to(DEVICE)
        
        self.features_extractor.train(True)
        optimizer = optim.SGD(self.features_extractor.parameters(), lr=0.2, weight_decay=0.00001, momentum=MOMENTUM)
        i = 0
        self.features_extractor.to(DEVICE)
        
        # Perform training
        for epoch in range(NUM_EPOCHS_RETRAIN):
            
            train_loss = 0.0
            
            if epoch in STEPDOWN_EPOCHS:
              for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/STEPDOWN_FACTOR
            
            self.features_extractor.train(True)
            for imgs, labels in loader:
                imgs = Variable(imgs).to(DEVICE)
                seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
                labels = Variable(seen_labels).to(DEVICE)
                
                labels_hot=torch.eye(self.n_classes)[labels] #one hot for classification loss
                labels_hot = labels_hot.to(DEVICE)
                
                optimizer.zero_grad()
                out = self(imgs)
                
                loss = nn.CrossEntropyLoss()(out, labels)
                
                # Teacher 1 : new ex. classes
                # Compute distillation loss (based on last net) : only exemplars of new classes are considered
                f_ex.to(DEVICE)
                f_ex.train(False)
                
                q_i = torch.sigmoid(f_ex.forward(imgs)) #forward pass on previous net
                
                
                
                dist_loss = bce_sum(out[:, self.n_known:], q_i[:, self.n_known:])/(len(out)*10)
                
                # Teacher 2 : old ex. classes
                # Compute second distillation loss (based on previous net) : only exemplars of old classes are considered
                f_prev_net.to(DEVICE)
                f_prev_net.train(False)
                q_i2 = torch.sigmoid(f_prev_net.forward(imgs))
                
                
                dist2_loss = bce_sum(out[:, :self.n_known], q_i2[:, :self.n_known])/(len(out)*self.n_known)
                
                #loss =  (1/(iter+1))*dist_loss + (iter/(iter+1))*dist2_loss
                
                loss =  loss + (iter/(iter+1))*dist2_loss
                
                train_loss += loss.item() * imgs.size(0) 
                        
                loss.backward()
                optimizer.step()
                
            train_loss = train_loss / len(loader.dataset)
                
            if i % 10 == 0 or i == (NUM_EPOCHS_RETRAIN-1):
                print('Epoch {} Loss:{:.4f}'.format(i, train_loss))
                for param_group in optimizer.param_groups:
                  print('Learning rate:{}'.format(param_group['lr']))
                print('-'*30)
            i+=1


    
    # Compute means of the new classes
    @torch.no_grad()
    def compute_new_means(self, images):
        
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
        
    
    def oversample_exemplars(self, m):
        
        for exemplars in self.exemplar_sets:
            
            original_lenght = len(exemplars)
            oversampled = []
            
            for _ in range(m):
                i = np.random.randint(0, original_lenght)
                #exemplars.append(exemplars[i])
                oversampled.append(exemplars[i])
            
            exemplars = np.concatenate((exemplars, np.array(oversampled)))
    
    
    #reduce exemplars lists
    @torch.no_grad()
    def reduce_exemplars_set(self, m, combine=False):        
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
    def compute_exemplars_mean(self, pca=False):
        
        if not pca:
        
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
            
        else:
            self.features_extractor.train(False)
            for exemplars in self.exemplar_sets:
                features = []
                for ex in  exemplars:
                    ex = Variable(transform(Image.fromarray(ex))).to(DEVICE)
                    feature = self.features_extractor.extract_features(ex.unsqueeze(0))
                    feature = feature.squeeze()
                    feature.data = feature.data / torch.norm(feature.data, p=2)
                    features.append(feature.cpu().numpy())
            
            
            pca = PCA(n_components=30)
            pca.fit(features)
            
            
            exemplar_means = []
            for exemplars in self.exemplar_sets:
                features = []
                for ex in  exemplars:
                    ex = Variable(transform(Image.fromarray(ex))).to(DEVICE)
                    feature = self.features_extractor.extract_features(ex.unsqueeze(0))
                    feature = feature.squeeze()
                    feature.data = feature.data / torch.norm(feature.data, p=2)
                    features.append(pca.transform(feature.unsqueeze(0).cpu().numpy()))
                    
                features = torch.from_numpy(np.array(features))
            
                #features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / torch.norm(mu_y.data, p=2) #l2 norm
                exemplar_means.append(mu_y.cpu())
            
            self.exemplar_means = exemplar_means
            
            
            self.pca = pca
            
        
    #classification method
    @torch.no_grad()
    def classify(self, x, classifier, pca=False, train_dataset=None):
        #Using NME as classifier
        if classifier == 'nme':
            
            #computing mean only if first iteration
            if self.compute_means:
                
                
                self.compute_exemplars_mean(pca=pca)
                    
            self.compute_means = False 
            exemplar_means = self.exemplar_means
            
            preds = []
            
            #computing features of images to be classified
            #print('computing pca')
            x = x.to(DEVICE)
            self.features_extractor.train(False)
            feature = self.features_extractor.extract_features(x)
            for feat in feature:
                measures = []
                feat = feat / torch.norm(feat, p=2) #l2 norm
                if pca:
                    feat = torch.from_numpy(self.pca.transform(feat.unsqueeze(0).cpu().numpy()))
               
                    
                #print('computing distance')
                #computing l2 distance with all class means
                for mean in exemplar_means:
                    measures.append((feat.cpu() - mean).pow(2).sum().squeeze().item())
                #chosing closest mean label as prediction
                preds.append(np.argmin(np.array(measures)))
                
            return preds
        # Using KNN, SVC, 3-layers MLP as classifier
        elif classifier == 'knn' or classifier == 'svc' or classifier == 'rand-forest':
            if self.train_model:
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
                elif classifier == 'svc-rbf':
                    model = SVC()
                
                elif classifier == 'rand-forest':
                    model = RandomForestClassifier()
                #fitting the model
                model.fit(X_train, y_train)
                
                self.model = model
                self.train_model = False
            #computing features of images to be classified
            x = x.to(DEVICE)
            self.features_extractor.train(False)
            feature = self.features_extractor.extract_features(x)
            X = []
            
            #l2 normalization
            for feat in feature:
                feat = feat / torch.norm(feat, p=2)
                if pca:
                    X.append(np.squeeze(self.pca.transform(feat.unsqueeze(0).cpu().numpy())))
                else:
                    X.append(feat.cpu().numpy())
            
            
            #getting predictions
            preds = self.model.predict(X)
            
            return preds
    #method to classify all batches of the test dataloader
    def classify_all(self, test_dataset, map_reverse, classifier, train_dataset=None):
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        running_corrects = 0

        for imgs, labels, _ in  test_dataloader:
            imgs = Variable(imgs).cuda()
            #classification with fully conntected layers
            if classifier == 'fc':
                 self.features_extractor.train(False)
                 self.features_extractor.to(DEVICE)
                 outputs =self.features_extractor(imgs)
                 _, preds = torch.max(outputs.data, 1)

            
            else:
                preds = self.classify(imgs, classifier, train_dataset=train_dataset)
            
            #mapping back fake lable to true label
            preds = [map_reverse[pred] for pred in preds]
            
            #computing accuracy
            running_corrects += (preds == labels.numpy()).sum()
            
        accuracy = running_corrects / float(len(test_dataloader.dataset))
        print('Test Accuracy: {}'.format(accuracy))
        return accuracy
