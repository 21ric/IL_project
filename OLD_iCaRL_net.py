import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

import resnet
import utils
import params

from resnet import resnet32

NUM_EPOCHS = 70


class ICarl(nn.Module):
    def __init__(self, n_classes=n_classes, k=2000):
        super().__init__()

        self.k = k
        self.n_classes = n_classes

        self.features_extractor = resnet32()

        self.examplars = {}
        self.means = None

    def forward(self, x):
        return self._features_extractor(x)


    def update(self, train_loader, test_loader, task, old_logits=None):

        optimizer = torch.optim.SGD(self.features_extractor.parameters(), lr=2., weight_decay=0.00001, momentum=0.9)
        #lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_scheduler(params.LR))


        for epoch in range(NUM_EPOCHS):
            #lr_scheduler.step()
            self.features_extractor.train(True)

            #cx = 0
            #_loss, _clf_loss, _distil_loss = 0., 0., 0.

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(params.DEVICE), targets.to(params.DEVICE)



                logits = model(inputs)
                if old_logits is None:
                    previous_logits = None
                else:
                    previous_logits = old_logits[cx:cx+inputs.shape[0]]

                clf_loss, distil_loss = compute_loss(
                    logits,
                    onehot_targets,
                    old_logits=previous_logits,
                    new_idx=task
                )
                loss = clf_loss + distil_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #stats["classification loss"] += clf_loss.item()
                #stats["distillation loss"] += distil_loss.item()

                cx += inputs.shape[0]
                _loss += loss.item()
                _clf_loss += clf_loss.item()
                _distil_loss += distil_loss.item()

            prog_bar.set_description(
                "Epoch: {}; Loss: {}; Clf loss: {}; Distill loss: {}".format(
                    epoch,
                    round(_loss / cx, 2),
                    round(_clf_loss / cx, 2),
                    round(_distil_loss / cx, 2),
                )
            )

        return stats
