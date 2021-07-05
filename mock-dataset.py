import os
import glob
#glob patterns specify sets of filenames with wildcard characters
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
#torchvision.transforms -> image transform
#functional -> fine-grained control over the transformations (e.g in the case of segmantation tasks)
    #don’t contain a random number generator for their parameters.
#transformation accepts PIL image, tensor images, or batches of tensor images
#tensor image is a tensor with (C, H, W)
    #C -> number of channels
    #H & W -> height and width
#batch of tensor images -> tensor of shape (B, C, H, W)
    #B -> number of images in a batch
    #C -> number of channels
    #H & W -> height and width
#torchvision.transforms resource: https://pytorch.org/vision/stable/transforms.html -> about transforms
from torch.utils.data import DataLoader
from PIL import Image # Python Imaging Library ( we have to download it first )
#PIL resource: https://pillow.readthedocs.io/en/stable/reference/Image.html
from scipy.misc import imread
#scipy.misc resource: https://docs.scipy.org/doc/scipy-0.18.1/reference/misc.html
from skimage.feature import canny #the canny algorithm to extract the edge
from skimage.color import rgb2gray #compute luminance (the intensity of light per unit area) of an RGB image.
from .utils import img_kmeans
import cv2 as cv #library of programming functions aimed at real-time computer vision


class Dataset(torch.utils.data.Dataset):
    #flist is the file path to the image directory
    def __init__(self, config, flist, augment=False, training=False):
        super(Dataset, self).__init__()
        self.augment = augment #im not sure if we want to augment the data
        self.training = training
        self.data = self.load_flist(flist) # loading the image

        self.input_size = config.INPUT_SIZE # input image size for training 0 for original size
        # self.sigma = config.SIGMA #standard deviation of the Gaussian filter used in Canny (0: random, -1: no edge)
        self.km = config.KM

    def __len__(self):
        return len(self.data)

    #to get the image
    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item #item is a tensors

    #load the actual image
    def load_item(self, index):
        # returns tensors of the image, image in grayscale, the edge, and color domain

        size = self.input_size

        # load image
        img = imread(self.data[index]) #read an image from a file as an array (from scipy.misc)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # create grayscale image
        img_gray = rgb2gray(img)

        # load edge
        edge = self.load_edge(img_gray, index)

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gray = img_gray[:, ::-1, ...]
            edge = edge[:, ::-1, ...]

        # To get color domain
        # random_blur = 2 * np.random.randint(7, 18) + 1
        random_blur = 25
        img_color_domain = cv.medianBlur(img, random_blur) #the central element of the image is replaced by the median of all the pixels in the kernel area.
        K = self.km
        # K = np.random.randint(2, 6)
        img_color_domain = img_kmeans(img_color_domain, K) #kmeans
        # img_blur = cv.medianBlur(img_blur, np.random.randint(1, 4) * 2 - 1)
        img_color_domain = cv.medianBlur(img_color_domain, 3)

        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(img_color_domain)

    def load_edge(self, img, index):
        # random_sigma = np.random.randint(25, 55) / 10.
        sigma = self.sigma

        # apply the canny algorithm
        # no edge
        if sigma == -1:
            return np.zeros(img.shape).astype(np.float)

        # random sigma
        if sigma == 0:
            sigma = random.randint(1, 4)

        return canny(img, sigma=sigma, mask=None).astype(np.float)

    #converting the image to tensor
    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True, interp='bilinear'):
        #interp -> interpolation with bilinear method
        imgh, imgw = img.shape[0:2] #2 is not inclusive

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw) #searching the minimum between height and width
            j = (imgh - side) // 2 #dividing by 2 and floor division (rounding below)
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width], interp=interp)  #imresize -> resize an image

        return img
    # to load the image path
    def load_flist(self, flist):
        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, list):
            return flist

        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []
