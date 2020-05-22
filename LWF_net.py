import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from resnet import resnet32

import math

####Hyper-parameters####
LR = 2
WEIGHT_DECAY = 0.00001
BATCH_SIZE = 128
NUM_EPOCHS = 5
DEVICE = 'cuda'
########################

def kaiming_normal_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')



class LwF(nn.Module):
  	def __init__(self, num_classes):
		super(LwF,self).__init__()
		self.model = resnet32()
		self.model.apply(kaiming_normal_init)
		self.model.fc = nn.Linear(64, num_classes) # Modify output layers

		# Save FC layer in attributes
		self.fc = self.feature_extractor.fc
		# Save other layers in attributes
		self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
		self.feature_extractor = nn.DataParallel(self.feature_extractor) 


		self.loss = nn.CrossEntropyLoss()
		self.dist_loss = nn.BCEWithLogitsLoss()

		self.optimizer = optim.SGD(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

		# n_classes is incremented before processing new data in an iteration
		# n_known is set to n_classes after all data for an iteration has been processed
		self.n_classes = 0
		self.n_known = 0

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

		if self.n_known == 0:
			new_out_features = n
		else:
			new_out_features = out_features + n
		print('new out features: ', new_out_features)
		self.model.fc = nn.Linear(in_features, new_out_features, bias=False)
		self.fc = self.model.fc
		
		kaiming_normal_init(self.fc.weight)
		self.fc.weight.data[:out_features] = weight
		self.n_classes += n
    


  def update_representation(self, dataset):

    targets = list(set(dataset.targets))
    n = len(targets)
    self.to(DEVICE)
    print('{} new classes'.format(len(targets)))

    #merge new data and exemplars
    for y, exemplars in enumerate(self.exemplars):
        dataset.append(exemplars, [y]*len(exemplars))

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    #Store network outputs with pre-updated parameters
    q = torch.zeros(len(dataset), self.num_classes).to(DEVICE)
    for images, labels, indexes in dataloader:
        images = images.to(DEVICE)
        indexes = indexes.to(DEVICE)
        q[indexes] = self(images)
    q.to(DEVICE)


    #Increment classes
    in_features = self.feature_extractor.fc.in_features
    out_features = self.feature_extractor.fc.out_features
    weight = self.feature_extractor.fc.weight.data

    self.feature_extractor.fc = nn.Linear(in_features, out_features+n, bias=False)
    self.feature_extractor.fc.weight.data[:out_features] = weight
    self.num_known = self.num_classes
    self.num_classes += n

    optimizer = self.optimizer

    i = 0
    self.to(DEVICE)
    for epoch in range(NUM_EPOCHS):
        if i%5 == 0:
            print('-'*30)
            print('Epoch {}/{}'.format(i+1, NUM_EPOCHS))
        for images, labels, indexes in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            indexes = indexes.to(DEVICE)


            #zero-ing the gradients
            optimizer.zero_grad()
            out = self(images)
            #classification Loss
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

  def reduce_exemplars_set(self, m):
    for y, exemplars in enumerate(self.exemplars):
        self.exemplars[y] = exemplars[:m]


  def construct_exemplars_set(self, images, m):

    feature_extractor = self.feature_extractor.to(DEVICE)
    features = []
    for img in images:
        img = img.unsqueeze(0)
        img = img.to(DEVICE)
        feature = feature_extractor.extract_features(img).data.cpu().numpy().squeeze()
        features.append(feature)

    #print(features)

    class_mean = np.mean(np.array(features))

    print('LUNGHEZA IMMAGINI')
    print(len(images))
    print('LUNGHEZZA FEATURES')
    print(len(features))

    exemplar_set = []
    exemplar_features = []
    for k in range(m):

        S = np.sum(exemplar_features, axis=0)
        phi = features
        mu = class_mean
        mu_p = 1.0/(k+1) * (phi + S)
        i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))

        exemplar_set.append(images[i])
        exemplar_features.append(features[i])

        print('INDICE SCELTO:{}'.format(i))

        exemplar_set.append(images[i])
        exemplar_features.append(features[i])

        features = np.delete(features, i)
        images.pop(i)


    self.exemplars.append(exemplar_set)

  #da cambiare completamente
  def classify(self, x):
    #computing exemplars mean
    exemplars_mean=[]
    for exemplars in self.exemplars:
        features = []
        for ex in exemplars:
            features.append(self.feature_extractor.extract_features(ex))
        exemplars_mean.append(np.mean(features))

    feature = self.feature_extractor(x)

    distances = np.sqrt([(feature - mean)**2 for mean in exemplars_mean])
    pred = np.argmin(distances)

    return preds
