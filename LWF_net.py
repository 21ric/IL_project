import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from resnet import resnet32

import math
import copy

val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

####Hyper-parameters####
LR = 2
WEIGHT_DECAY = 0.00001
BATCH_SIZE = 128
NUM_EPOCHS = 30
DEVICE = 'cuda'
STEPDOWN_EPOCHS = [int(0.7 * NUM_EPOCHS), int(0.9 * NUM_EPOCHS)]
STEPDOWN_FACTOR = 5
########################

def validate(net, val_dataloader, map_reverse):
    running_corrects_val = 0
    for inputs, labels, index in val_dataloader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        net.train(False)
        # forward
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
        running_corrects_val += (preds == labels.cpu().numpy()).sum()
        #running_corrects_val += torch.sum(preds == labels.data)

    valid_acc = running_corrects_val / float(len(val_dataloader.dataset))

    net.train(True)
    return valid_acc


def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

class LwF(nn.Module):
    
    def __init__(self, n_classes, classes_map):
        super(LwF,self).__init__()        
        self.model = resnet32(num_classes=10)
        #self.model.apply(kaiming_normal_init)
        self.model.fc = nn.Linear(64, n_classes) # Modify output layers

        # Save FC layer in attributes
        self.fc = self.model.fc
        
        # Save other layers in attributes
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor = nn.DataParallel(self.feature_extractor) 
        
        
        self.class_loss = nn.BCEWithLogitsLoss() #classification loss
        self.dist_loss = nn.BCEWithLogitsLoss()    #distillation loss

        self.optimizer = optim.SGD(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        # n_classes is incremented before processing new data in an iteration
        # n_known is set to n_classes after all data for an iteration has been processed
        self.n_classes = 0
        self.n_known = 0
        self.classes_map = classes_map

        
        
    def forward(self, x):
        
        x = self.feature_extractor(x) 
        x = x.view(x.size(0), -1)
        x = self.fc(x) 
        '''
        x = self.model(x)
        '''
        return x


    def increment_classes(self, new_classes):
        """Add n classes in the final fc layer"""
        n = len(new_classes)
        print('new classes: ', n)
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        if self.n_known == 0: # First iteration
            new_out_features = n
        else: # Other iterations
            new_out_features = out_features + n
            
        print('new out features: ', new_out_features)
        
        # Update model, changing last FC layer
        self.model.fc = nn.Linear(in_features, new_out_features, bias=False)
        # Update attribute self.fc
        self.fc = self.model.fc
        # Initialize weights with kaiming normal
        #kaiming_normal_init(self.fc.weight)
        
        
        # Upload old FC weights on first "out_features" nodes
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n
    
    def classify(self, images):
        """Classify images by softmax
        Args:
            x: input image batch
        Returns:
            preds: Tensor of size (batch_size,)
        """
        _, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)
        return preds    
    
    def update(self, train_dataset, val_dataset, class_map, map_reverse):

        self.cuda()

        # Save true labels (new images)
        classes = list(set(train_dataset.dataset.targets)) #list of true labels
        print("Classes: ", classes)
        print('Known: ', self.n_known)
     
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, drop_last=False)
        
        #Store network outputs with pre-updated parameters
        if (self.n_known > 0) :
            dist_target = torch.zeros(len(train_dataset), self.n_classes).cuda()
            self.to(DEVICE)
            self.train(False)
            for images, labels, indices in train_dataloader:
                images = Variable(images).cuda()
                indexes = indices.cuda()
                g = torch.sigmoid(self.forward(images))
                #g = self.forward(images) 
                dist_target[indexes] = g.data
     
            dist_target = Variable(dist_target).cuda()
            self.train(True)

        new_classes = classes #lista (non duplicati) con targets di train. len(classes)=10
        

        if len(new_classes) > 0:
            self.increment_classes(new_classes)
            self.cuda()
               
        # Define optimizer and classification loss
        self.optimizer = optim.SGD(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer = self.optimizer
        #criterion = self.class_loss
        criterion_class = self.class_loss
        criterion_dist = self.dist_loss
        
        self.to(DEVICE)

        scores = {}
        best_acc = 0 # This is the validation accuracy for model selection
        self.train(True)
        
        for epoch in range(NUM_EPOCHS): 
             
            if epoch%5 == 0:
                print('-'*30)
                print('Epoch {}/{}'.format(epoch+1, NUM_EPOCHS))
                for param_group in optimizer.param_groups:
                    print('Learning rate:{}'.format(param_group['lr']))

            # Divide learning rate by 5 after 49 63 epochs
            if epoch in STEPDOWN_EPOCHS:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/STEPDOWN_FACTOR


            for  images, labels, indices in train_dataloader:
                images = images.to(DEVICE)
                indices = indices.to(DEVICE)
                # We need to save labels in this way because classes are randomly shuffled at the beginning
                seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
                labels = Variable(seen_labels).to(DEVICE)
                labels_hot=torch.eye(self.n_classes)[labels]
                labels_hot = labels_hot.to(DEVICE)

                # Zero-ing the gradient
                optimizer.zero_grad()

                # Compute outputs on the new model 
                logits = self(images) 

                # Compute classification loss 
                #cls_loss = criterion_class(logits[:, self.n_known:], labels_hot[:, self.n_known:])
                
                
                if self.n_known <= 0: # First iteration
                    #loss = criterion_class(logits[:, self.n_known:self.n_classes], labels_hot[:, self.n_known:self.n_classes])
                    loss = criterion_class(logits, labels_hot)
                
                elif self.n_known > 0: # If not first iteration
                    # Save outputs of the previous model on the current batch
                    dist_target_i = dist_target[indices] #BCE
                    #dist_target_batch = prev_model.forward(images)  #MCCE
                    #dist_target_raw = torch.LongTensor([label for label in dist_target]) #MCEE
                    #dist_target = Variable(dist_target_raw).cuda() #MCEE
                    #_, dist_target = torch.max(torch.softmax(dist_target_raw, dim=1), dim=1, keepdim=False)

                    # Save logits of the first "old" nodes of the network
                    # LwF doesn't use examplars, it uses the network outputs itselfs
                    #logits = torch.sigmoid(logits) #BCE
                    #logits_dist = logits[:,:self.n_known]  #MCCE

                    # Compute distillation loss
                    #target = [dist_target_i, labels_hot]
                    #dist_loss = sum(criterion_dist(logits[:, y], dist_target_i[:, y]) for y in range(self.n_known)) #BCE
                    target = torch.cat((dist_target_i[:,:self.n_known], labels_hot[:,self.n_known:self.n_classes]),dim=1)

                    loss = criterion_dist(logits, target) #richi dist_loss
                    #dist_loss = criterion_dist(logits_dist, dist_target_batch)  #MCCE

                    # Compute total loss
                    #loss = dist_loss+cls_loss
                    #print(dist_loss.item())    
                
                #else:
                #   loss = cls_loss
                    
                loss.backward()
                optimizer.step()

            # VALIDATION    
            # val_dataloader.dataset.transform = val_transform
            val_acc = validate(self, val_dataloader, map_reverse)
            '''
            running_corrects = 0.0

            for  images, labels, indices in val_dataloader: 
                images = Variable(images)
                images = images.to(DEVICE)
                indices = indices.to(DEVICE)
                labels = labels.to(DEVICE)


                # Set the network to evaluation mode
                self.train(False)

                # Forward + classify
                preds = self.classify(images)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]        
                running_corrects += (preds == labels.cpu().numpy()).sum()
                
            val_acc = running_corrects / float(len(val_dataloader.dataset))
            '''

            if (val_acc > best_acc):
                best_acc = val_acc
                best_net = copy.deepcopy(self.state_dict())

            scores[epoch+1] = val_acc 


            if epoch%5 == 0:
                print("Train Loss: {:.4f}\n".format(loss.item()))
                print('Val Acc: {:.4f}'.format(val_acc))


        #end epochs
        self.load_state_dict(best_net)  
        return [scores, self]  
        
                

