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
import pandas as pd
import torchvision.io
from torchvision.io import read_image
import glob
import cv2
from PIL import Image
import csv
import sys
import skimage

from datacleaner import autocleaner
#datacleaning -> datacleaner, prettypandas
#pip install datacleaner

nsf = 32 #size of siamese feature map
siamese_batch = 64
batch_size = 128
size_z = 100
ngf = 64
ndf = 64
nc = 3 # rgb = 3, graysace = 1


train_data_path = '/media/imero/Elements/flarp_folding_1/kinova_color_images'
test_data_path = '/media/imero/Elements/flarp_folding_1/kinova_color_images'

#assuming the number of force data is equal to the number of existing image
#only for one image, we need another one
for img in os.listdir(train_data_path):
	img_array = cv2.imread(os.path.join(train_data_path,img))
	img_array = (img_array.flatten())
	img_array = img_array.reshape(-1,1).T
	with open('img_files.csv', 'a') as f:
		writer = csv.writer(f)
		writer.writerow('rgb image')
		writer.writerow(img_array)

#reading the force info csv
force_data = pd.read_csv("/media/imero/Elements/flarp_folding_1/csvFiles/flarp_folding_1-_my_gen3_base_feedback.csv")
#reading the image csv
img_data = pd.read_csv("img_files.csv")

#combining the data
complete_csv = img_data.merge(force_data, on = "date") # on what basis we want to merge the data?
complete_csv.to_csv("complete_files.csv", index = False)

#reading the complete data
complete_data = pd.read_csv("/media/imero/Documents/F2V-Research/Trial-Model-Alvionna/complete_files.csv")

#cleaning the data
cleaned_data = autoclean(complete_data)

# #appending the img and force info to one csv file
# for index in clean_img_data.iloc[:]: #taking all the rows
#     for header in clean_force_data.columns: #taking the force information's headers
#         complete_data[header] = clean_force_data[index]
#
# complete_data.to_csv("complete_files.csv", index = False)

#still taking 1 image
class CSVDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir, self.csv_file.iloc[idx, 0]) #extracting only one image, we need to use for loop
        image = read_image(img_name) #io.imread (skimage)
        force_header = list(self.csv_file.columns) #return a list of the header
        force_info = self.csv_file.iloc[idx, 1:] #extracting the columns
        force_info = np.array([force_info])
        sample = {'image': image, force_header[i]: force_info[i] for i in range(len(force_header))}
		#create a dictionary to easily search for the info that we want

        if self.transform:
            sample = self.transform(sample)

        return sample

#have to change it to fit 2 images
class KinovaDataset(Dataset):
    def __init__(self, img_paths, transform=None, target_transform=None):
        self.img_paths = img_paths
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_filepath = self.img_paths[idx]
        image = read_image(img_filepath)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image

        # what label do we have?

        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) #getting the location of our image datase
        # #iloc (csv file has row and columns) -> iloc[row, column] -> iloc[idx, 0] = taking values from the 1st column
        # image = read_image(img_path) #reading the image and converts it into a tensor
        # label = self.img_labels.iloc[idx, 1] #get the corresponding label from csv data
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # return image, label

# train_dataset = KinovaDataset(train_image_paths)
# validation_dataset = KinovaDataset(validation_paths)
# test_dataset = KinovaDataset(test_image_paths)
#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
# validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size, shuffle = True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

#tensor -> (C,H,W)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        height, width = image.shape[:2] #extracting height and width
        if isinstance(self.output_size, int): # if the input is an int
            if height > width:
                new_height, new_width = self.output_size * height / width, self.output_size
				#maintaining the aspect ratio
            else:
                new_height, new_width = self.output_size, self.output_size * width / height #maintaining the aspect ratio
        else:
            new_height, new_width = self.output_size

        new_height, new_width = int(new_height), int(new_width)

        img = transforms.resize(image, (new_h, new_w))

        return {'image': img} #return a dictionary


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image)}

csv_path = "/media/imero/Documents/F2V-Research/Trial-Model-Alvionna/"
csv_file = "complete_files.csv"

train_dataset = CSVDataset(csv_file=csv_file, root_dir=csv_path, transforms=transforms.Compose([Rescale((256,256)),ToTensor()]))


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

# do we want to separate the dataset to validation and test?
train_image_paths = train_image_paths[:int(0.8*len(train_image_paths))] # 80%
validation_paths = train_image_paths[int(0.8*len(train_image_paths)):] # 20%

class View(nn.Module):
    def __init__(self, shape):
		super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

# DON'T FORGET TO NORMALIZE THE IMAGE (normalize((mean,mean,mean), (sd,sd,sd))) for each channel

# the paper accepts a 28x28 pixel grayscale image & the output a vector of length 50
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(siamese_batch),
            nn.Conv2d(siamese_batch, nsf, (2,2), stride=0, padding=1, bias=False),
            nn.Tanh(),

            nn.Conv2d(nsf, nsf/2, (5,5), stride=0, padding=1, bias=False),
            nn.Linear(nsf/2, nsf * 4, bias=False),
            nn.ReLu(),

            nn.Linear(nsf * 4, 50, bias = False),
            nn.ReLU()
        )

    #using the same architecture but having 2 inputs going through the same encoder
    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.main(input1)
        # forward pass of input 2
        output2 = self.main(input2)
        return output1, output2

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
            nn.BatchNorm2d(batch_size * 7 * 7),
            nn.Tanh(),

            nn.View((7,7,batch_size)), #home-made, there's no reshape in pytorch, and we cannot use view in nn.sequential
            nn.Upsample((2,2), mode = 'nearest'), #mode = nearest, bilinear, bicubic and trilinear
            #nn.ConvTranspose2d(ngf * 2, ngf * 16, (5,5), 2, 1, bias False),
            nn.Conv2d(ngf * 16, ngf * 2, (5,5), stride=2, padding=1, bias=False),
            nn.Tanh(),
            nn.Upsample((2,2), mode = 'nearest'), #mode = nearest, bilinear, bicubic and trilinear
            #nn.ConvTranspose2d(ngf * 2, ngf, (5,5), 2, 1, bias False),

            nn.Conv2d(ngf, 1, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

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

encoder = Encoder().cuda()
generator = Generator().cuda()
discriminator = Discriminator().cuda()

# DON'T FORGET TO CONCATENATE THE INDIVUDAL OUTPUT OF THE SIAMESE ENCODER

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

learning_rate = 0.0005
beta_1 = 0.5
criterion_contrastive = ContrastiveLoss() #can use in-build loss functions, this one is for siamese network
criterion_triplet = torch.nn.TripletMarginLoss(margin = 0.1, p=2) #p = the norm degree for pairwise distance
criterion_GD = nn.BCELoss()

#weight decay = L2 penalty
optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=0.0005)
optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=0.0005)
optimizerE = optim.Adam(encoder.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=0.0005)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

num_epochs = 1000


def train():
    img_list = []
    G_losses = []
    D_losses = []
    E_losses = []
    iters = 0
    num_epochs = 1000

    for epochs in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            ###################
            # ENCODER TRAINING
            ###################
            img0, img1 , label = data
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizerE.zero_grad()
            output1,output2 = encoder(img0,img1)
            loss_encoder = criterion_contrastive(output1,output2,label)
            # loss_triplet  = criterion_triplet(output1, output2,label)
            loss_encoder.backward()
            # loss_triplet.backward()
            optimizerE.step()

            output_encoder = torch.cat((output1, output2), 1)

            #########################
            # DISCRIMINATOR TRAINING
            #########################

            #train all-fake
            optimizerD.zero_grad()
            fake_sample = generator(output_encoder)
            fake_label = torch.zeros(fake_sample.size(0), 1, device=device)
            fake_preds = discriminator(fake_sample)
            loss_discriminator_fake = criterion_GD(fake_preds, fake_label)
            loss_discriminator_fake.backward()
            D_x = fake_preds.mean().item()

            #train all-real
            real_label = torch.ones(output_encoder.size(0), 1, device=device)
            real_preds = discriminator(output_encoder) #what is the real image? we can use img0, img1
            loss_discriminator_real = criterion_GD(real_preds, real_label)
            loss_discriminator_real.backward()
            D_G_z1 = real_preds.mean().item()

            discriminator_loss = loss_discriminator_fake + loss_discriminator_fake

            optimizerD.step() #update the discriminator

            #####################
            # GENERATOR TRAINING
            #####################
            optimizerG.zero_grad()
            gen_label = torch.ones(output_encoder.size(0), 1, device=device)
            fake_image = generator(output_encoder)
            #trying to fool the discriminator
            gen_preds = discriminator(fake_image)
            loss_generator = criterion_GD(gen_preds, gen_label)

            loss_generator.backward()
            D_G_z2 = gen_preds.mean().item()

            optimizerG.step()

            #count the losses
            G_losses.append(loss_genenator.item())
            D_losses.append(discriminator_loss.item())
            E_losses.append(loss_contrastive.item())

        return G_losses, D_losses, E_losses

def main():
	return train()

if __name__ == "__main__":
    main()
