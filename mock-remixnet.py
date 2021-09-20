from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

batch_size = 128
size_z = 100
ngf = 64
ndf = 64
nc = 3 # rgb = 3, graysace = 1

class View(nn.Module):
    def __init__(self, shape):
		super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

# def generator_model():
#     model = Sequential()
#     model.add(Dense(input_dim=100, output_dim=1024))
#     model.add(Activation('tanh'))
#     model.add(Dense(128*7*7))
#     model.add(BatchNormalization())
#     model.add(Activation('tanh'))
#     model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
#     model.add(UpSampling2D(size=(2, 2)))
#     model.add(Conv2D(64, (5, 5), padding='same'))
#     model.add(Activation('tanh'))
#     model.add(UpSampling2D(size=(2, 2)))
#     model.add(Conv2D(1, (5, 5), padding='same'))
#     model.add(Activation('tanh'))
#     return model


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(size_z, ngf * 16, bias = False),
            nn.Tanh(),

            nn.Linear(ngf * 16, 128 * 7 * 7, bias = False),
            nn.BatchNorm2d(128 * 7 * 7),
            nn.Tanh(),

            nn.View((7,7,128)), #home-made, there's no reshape in pytorch, and we cannot use view in nn.sequential
            nn.Upsample((2,2), mode = 'nearest'), #mode = nearest, bilinear, bicubic and trilinear
            #nn.ConvTranspose2d(ngf * 2, ngf * 16, (5,5), 2, 1, bias False),
            nn.Conv2d(ngf * 16, ngf * 2, (5,5), stride=2, padding=1, bias=False),
            nn.Tanh(),
            nn.Upsample((2,2), mode = 'nearest'), #mode = nearest, bilinear, bicubic and trilinear
            #nn.ConvTranspose2d(ngf * 2, ngf, (5,5), 2, 1, bias False),

            nn.Conv2d(ngf, 1, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

# def discriminator_model():
#     model = Sequential()
#     model.add(
#             Conv2D(64, (5, 5),
#             padding='same',
#             input_shape=(28, 28, 1))
#             )
#     model.add(Activation('tanh'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(128, (5, 5)))
#     model.add(Activation('tanh'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(1024))
#     model.add(Activation('tanh'))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#     return model

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, (5,5), 2, 1, bias=False),
            nn.Tanh(),
            nn.MaxPool2D(kernel_size=(2,2)),
            # input is (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, (5,5), 2, 1, bias=False),
            nn.Tanh(),
            nn.MaxPool2D(kernel_size=(2,2)),
            nn.Flatten(),
            # input is (ndf * 2) and output is (ndf * 8) e ndf * 8 = 1024 if ndf = 64
            nn.Linear(ndf * 2, ndf * 8, bias = False),
            nn.Tanh(),
            # input is (ndf * 8) and output is 1 where ndf * 8 = 1024 if ndf = 64
            nn.Linear(ndf * 8, 1, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
