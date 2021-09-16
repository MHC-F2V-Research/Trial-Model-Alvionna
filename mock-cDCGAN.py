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
from IPython.display import HTML
from torch.autograd import Variable

# Root directory for dataset
dataroot = "data/celeba"

# Number of workers threads for loading the data with dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
# size using a transformer. (default = 64 x 64)
image_size = 64

# Number of color channels in the training images. For color images this is 3
# channels = the depth of the matrices
num_channels = 3

# Size of z latent vector (i.e. size of generator input)
size_z = 100

# the size for the label
label_dim = 10

# Size of feature maps in generator (the depths of feature maps)
# feature map -> output of convolution (the specific feature we got after apply filters to the image)
# the depth depends on how many filters you apply for a layer
g_size_feature = 64

# Size of feature maps in discriminator (the depths of feature maps)
d_size_feature = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               #normalize a tensor image with mean and standard deviation
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
            nn.ConvTranspose2d(size_z, g_size_feature * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_size_feature * 8),
            nn.LeakyReLU(0.2), #negative slope = 0.2

            nn.ConvTranspose2d(size_z, g_size_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_size_feature * 4),
            nn.LeakyReLU(0.2), #negative slope = 0.2

            nn.ConvTranspose2d(size_z, g_size_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_size_feature * 2),
            nn.LeakyReLU(0.2), #negative slope = 0.2

            nn.ConvTranspose2d(g_size_feature * 2, g_size_feature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_size_feature),
            nn.LeakyReLU(0.2), #negative slope = 0.2

            nn.Conv2d(g_size_feature, num_channels, 4, 2, 1, bias=True),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
            # input is (nc) x 64 x 64
            nn.ConvTranspose2d(size_z, g_size_feature * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(g_size_feature * 8),
            nn.LeakyReLU(0.2), #negative slope = 0.2

            nn.Conv2d(size_z, g_size_feature * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(g_size_feature * 8),
            nn.LeakyReLU(0.2), #negative slope = 0.2

            nn.Conv2d(size_z, g_size_feature * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(g_size_feature * 8),
            nn.LeakyReLU(0.2), #negative slope = 0.2

            nn.Flatten(),
            nn.Dropout(p = 0.4),
            # randomly zeroes some of the elements of the input with probability p using samples from a Bernoulli distribution

            nn.Conv2d(g_size_feature, num_channels, 3, 2, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


# initialize BCELoss function (binary_cross_entropy)
criterion = nn.BCELoss()

# initialize the custom weights
def weights_init(m):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.init.constant_(m.bias.data, 0) #setting the bias to 0

#creating the model
generator = Generator(ngpu).to(device)
discriminator = Discriminator(ngpu).to(device)

#apply the random weights
generator.apply(weights_init)
discriminator.apply(weights_init)

#loss function
criterion = nn.BCELoss()

# setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

#generate real sample
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y

# generate points in latent space as input for the generator
temp_noise = torch.randn(label_dim, G_input_dim)
fixed_noise = temp_noise
fixed_c = torch.zeros(label_dim, 1)
for i in range(9):
    fixed_noise = torch.cat([fixed_noise, temp_noise], 0)
    temp = torch.ones(label_dim, 1) + i
    fixed_c = torch.cat([fixed_c, temp], 0)

fixed_noise = fixed_noise.view(-1, size_z, 1, 1)
fixed_label = torch.zeros(size_Z, label_dim)
fixed_label.scatter_(1, fixed_c.type(torch.LongTensor), 1)
fixed_label = fixed_label.view(-1, label_dim, 1, 1)


# label preprocess
onehot = torch.zeros(label_dim, label_dim)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(label_dim, 1), 1).view(label_dim, label_dim, 1, 1)
fill = torch.zeros([label_dim, label_dim, image_size, image_size])
for i in range(label_dim):
    fill[i, i, :, :] = 1

img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (images, labels) in enumerate(dataloader):
        # image data
        mini_batch = images.size()[0]
        x_ = Variable(images.cuda())

        # labels
        y_real_ = Variable(torch.ones(mini_batch).cuda())
        y_fake_ = Variable(torch.zeros(mini_batch).cuda())
        c_fill_ = Variable(fill[labels].cuda())

        # Train discriminator with real data
        D_real_decision = discriminator(real_cpu).view(-1)
        D_real_loss = criterion(D_real_decision, y_real_)

        # Train discriminator with fake data
        z_ = torch.randn(mini_batch, size_z).view(-1, size_z, 1, 1)
        z_ = Variable(z_.cuda())

        c_ = (torch.rand(mini_batch, 1) * label_dim).type(torch.LongTensor).squeeze()
        c_onehot_ = Variable(onehot[c_].cuda())
        gen_image = generator(z_, c_onehot_)

        c_fill_ = Variable(fill[c_].cuda())
        D_fake_decision = discriminator(gen_image.detach()).view(-1)
        D_fake_loss = criterion(D_fake_decision, y_fake_)

        # Back propagation
        D_loss = D_real_loss + D_fake_loss
        discriminator.zero_grad()
        D_loss.backward()
        optimizerD.step()

        # Train generator
        z_ = torch.randn(mini_batch, size_z).view(-1, size_z, 1, 1)
        z_ = Variable(z_.cuda())

        c_ = (torch.rand(mini_batch, 1) * label_dim).type(torch.LongTensor).squeeze()
        c_onehot_ = Variable(onehot[c_].cuda())
        gen_image = G(z_, c_onehot_)

        c_fill_ = Variable(fill[c_].cuda())
        D_fake_decision = D(gen_image, c_fill_).squeeze()
        G_loss = criterion(D_fake_decision, y_real_)

        # Back propagation
        generator.zero_grad()
        G_loss.backward()
        optimizerG.step()

        # loss values
        D_losses.append(D_loss.data[0])
        G_losses.append(G_loss.data[0])

        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
              % (epoch+1, num_epochs, i+1, len(data_loader), D_loss.data[0], G_loss.data[0]))

        # ============ TensorBoard logging ============#
        D_logger.scalar_summary('losses', D_loss.data[0], step + 1)
        G_logger.scalar_summary('losses', G_loss.data[0], step + 1)
        step += 1

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    # avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)

    plot_loss(D_avg_losses, G_avg_losses, epoch, save=True)

    # Show result for fixed noise
    plot_result(G, fixed_noise, fixed_label, epoch, save=True)
