import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from sync_batchnorm import SynchronizedBatchNorm2d

BatchNorm = SynchronizedBatchNorm2d

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# Number of training epochs
num_epochs = 5

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class Predictor(nn.Module):
    def __init__(self, ngpu):
        super(Predictor, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(8 * ngf, 100, 3, 1, 1),
            BatchNorm(100),
            nn.ReLU(),
            nn.Conv2d(100, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.init.constant_(m.bias.data, 0) #setting the bias to 0

predictor = Predictor(ngpu).to(device)

predictor.apply(weights_init)

#loss function
criterion = nn.BCELoss()
