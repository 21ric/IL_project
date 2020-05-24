import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

from resnet import resnet32

import math
import copy

'''
import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import copy
'''

####Hyper-parameters####
LR = 2
WEIGHT_DECAY = 0.00001
BATCH_SIZE = 128
NUM_EPOCHS = 10
DEVICE = 'cuda'
STEPDOWN_EPOCHS = [int(0.7 * NUM_EPOCHS), int(0.9 * NUM_EPOCHS)]
STEPDOWN_FACTOR = 5
########################


def MultiClassCrossEntropy(logits, labels, T):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
    labels = torch.softmax(labels/T, dim=1)
    # print('outputs: ', outputs)
    # print('labels: ', labels.shape)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    # print('OUT: ', outputs)
    return Variable(outputs.data, requires_grad=True).cuda()

def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

class LwF(nn.Module):
	
    def __init__(self, num_classes, classes_map):
        super(LwF,self).__init__()
		
        self.model = resnet32()
        self.model.apply(kaiming_normal_init)
        self.model.fc = nn.Linear(64, num_classes) # Modify output layers

        # Save FC layer in attributes
        self.fc = self.model.fc
        # Save other layers in attributes
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor = nn.DataParallel(self.feature_extractor) 

        self.loss = nn.CrossEntropyLoss() #classification loss
        #self.dist_loss = nn.BCELoss()
        self.dist_loss = nn.CrossEntropyLoss() #distillation loss
        #self.dist_loss = nn.BCEWithLogitsLoss() #distillation loss

        self.optimizer = optim.SGD(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        # n_classes is incremented before processing new data in an iteration
        # n_known is set to n_classes after all data for an iteration has been processed
        self.n_classes = 0
        self.n_known = 0
        self.classes_map = classes_map

        #self.num_classes = num_classes
        #self.num_known = 0
		
		
		
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return(x)

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
        kaiming_normal_init(self.fc.weight)
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
	
    def update(self, dataset, class_map):
        self.cuda()

        self.compute_means = True

        # Save a copy to compute distillation outputs
        prev_model = copy.deepcopy(self)
        prev_model.to(DEVICE)

        # Save true labels (new images)
        classes = list(set(dataset.targets)) #list of true labels
        print("Classes: ", classes)
        print('Known: ', self.n_known)
	
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
	
        #Store network outputs with pre-updated parameters
        if (self.n_known > 0) :
            dist_target = torch.zeros(len(dataset), self.n_classes).cuda()
            for images, labels, indices in dataloader:
                images = Variable(images).cuda()
                indexes = indices.cuda()
                g = torch.sigmoid(self.forward(images))
                dist_target[indices] = g.data
            dist_target = Variable(dist_target).cuda()

        new_classes = classes #lista (non duplicati) con targets di train. len(classes)=10

        if len(new_classes) > 0:
            # Change last FC layer
            # adding 10 new output neurons and change self.n_classes attribute
            self.increment_classes(new_classes)  
	           
        # Define optimizer and classification loss
        self.optimizer = optim.SGD(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer = self.optimizer
        criterion = self.loss
        criterion_dist = self.dist_loss
		
        self.to(DEVICE)
        with tqdm(total=NUM_EPOCHS) as pbar:
            i = 0
            for epoch in range(NUM_EPOCHS):	
                if i%5 == 0:
                    print('-'*30)
                    print('Epoch {}/{}'.format(i+1, NUM_EPOCHS))
                    for param_group in optimizer.param_groups:
                        print('Learning rate:{}'.format(param_group['lr']))
		
                # Divide learning rate by 5 after 49 63 epochs
                if epoch in STEPDOWN_EPOCHS:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr']/STEPDOWN_FACTOR

                # train phase, the model weights are update such that it is good with the new task
                # and also with the old one
                for  images, labels, indices in dataloader:
					
                    seen_labels = []
					
                    images = images.to(DEVICE)
                    indices = indices.to(DEVICE)
                    # We need to save labels in this way because classes are randomly shuffled at the beginning
                    seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
                    labels = Variable(seen_labels).to(DEVICE)

                    # Zero-ing the gradient
                    optimizer.zero_grad()
					
                    # Compute outputs on the new model 
                    logits = self.forward(images) 
					
                    # Compute classification loss 
                    cls_loss = criterion(logits, labels)
            
					
                    # If not first iteration
                    if self.n_known > 0:
                        # Save outputs of the previous model on the current batch
                        #dist_target_i = dist_target[indices] 
                        dist_target = prev_model.forward(images)  #MCCE
			
                        # Save logits of the first "old" nodes of the network
                        # LwF doesn't use examplars, it uses the network outputs itselfs
                        #logits = torch.sigmoid(logits)
                        #logits_dist = logits[:,:-(self.n_classes-self.n_known)]  #MCCE
			
                        # Compute distillation loss
                        dist_loss = sum(criterion_dist(logits[:, y], dist_target_i[:, y]) for y in range(self.n_known))
                        #dist_loss = criterion_dist(logits_dist, dist_target_i)  #MCCE
                      
                        # Compute total loss
                        loss = dist_loss+cls_loss
                        #print(dist_loss.item())
					
                    # If first iteration
                    else:
                        loss = cls_loss

                    loss.backward()
                    optimizer.step()
				
                if i%5 == 0:
                   print("Loss: {:.4f}\n".format(loss.item()))
				
                i+=1
	
                pbar.update(1)
                

