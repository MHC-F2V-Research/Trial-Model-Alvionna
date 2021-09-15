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

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

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
num_channels = 3

# Size of z latent vector (i.e. size of generator input)
size_z = 100

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

# We can use an image folder dataset the way we have it setup.
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

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) #mean = 0.0, sd = 0.02
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02) #mean = 1.0, sd = 0.02
        nn.init.constant_(m.bias.data, 0) #setting the bias to 0


# Generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( size_z, g_size_feature * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_size_feature * 8),
            nn.ReLU(True),
            # state size. (g_size_feature*8) x 4 x 4
            nn.ConvTranspose2d(g_size_feature * 8, g_size_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_size_feature * 4),
            nn.ReLU(True),
            # state size. (g_size_feature*4) x 8 x 8
            nn.ConvTranspose2d( g_size_feature * 4, g_size_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_size_feature * 2),
            nn.ReLU(True),
            # state size. (g_size_feature*2) x 16 x 16
            nn.ConvTranspose2d( g_size_feature * 2, g_size_feature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_size_feature),
            nn.ReLU(True),
            # state size. (g_size_feature) x 32 x 32
            nn.ConvTranspose2d( g_size_feature, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
netG.apply(weights_init)

# Print the model
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, d_size_feature, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (d_size_feature) x 32 x 32
            nn.Conv2d(d_size_feature, d_size_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_size_feature * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (d_size_feature*2) x 16 x 16
            nn.Conv2d(d_size_feature * 2, d_size_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_size_feature * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (d_size_feature*4) x 8 x 8
            nn.Conv2d(d_size_feature * 4, d_size_feature * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_size_feature * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (d_size_feature*8) x 4 x 4
            nn.Conv2d(d_size_feature * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
netD.apply(weights_init)

# Print the model
print(netD)

# Initialize BCELoss function (binary_cross_entropy)
loss_function = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(64, size_z, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# train the discriminator
    # we want to maximize (log(D(x)) + log (1 - D(G(z)))
    # doing it in 2 steps:
        # forward pass all real samples through D, calculate the loss log(D(x)), then backward pass
        # construct fake samples with generator, forward pass through D, calculate the loss log (1 - D(G(z))), then backward pass

# train the generator
    # we want to minimize log (1 - D(G(z)) and maximize log(D(G(z)))
    # steps:
        # classifying the Generator output from Part 1 with the Discriminator
        # computing G’s loss using real labels as ground truth
        # computing G’s gradients in a backward pass
        # updating G’s parameters with an optimizer step. It may seem counter-intuitive to use the real labels as GT labels for the loss function, but this allows us to use the log(x) part of BCELoss (rather than log(1-x))


# D(x) - the average output (across the batch) of the discriminator for the all real batch. This should start close to 1 then theoretically converge to 0.5 when G gets better.
# D(G(z)) -  average discriminator outputs for the all fake batch. These numbers should start near 0 and converge to 0.5 as G gets better.

# D_losses - discriminator loss calculated as the sum of losses for the all real and all fake batches (log(D(x)) + log (1 - D(G(z)))
# G_losses - generator loss calculated log(D(G(z)))

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errorD_real = loss_function(output, label)
        # Calculate gradients for D in backward pass
        errorD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, size_z, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errorD_fake = loss_function(output, label)
        # Calculate the gradients for this batch, summed with previous gradients
        errorD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        D_error = errorD_real + errorD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        G_error = loss_function(output, label)
        # Calculate gradients for G
        G_error.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     D_error.item(), G_error.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
