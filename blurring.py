"""
 * Python script to demonstrate Gaussian blur.
 *
 * usage: python GaussBlur.py <filename> <sigma>
"""
from PIL import Image
import skimage
from skimage.viewer import ImageViewer
import sys
import cv2
import numpy as np
import os
import random
import itertools
from matplotlib import pyplot as plt


##############################################
##############################################
# METHODS
##############################################
##############################################

def resize(directory):
    for filename in os.listdir(directory):
        image = cv2.imread(os.path.join(directory, filename))
        dim = (480,270)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        path = "kinova_color_resized_images" #RENAME IT
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(os.path.join(path, filename), img_color)

def blurring(directory_color, directory_depth):
    n_avg1 = random.randint(0,100)
    n_avg2 = random.randint(0,100)
    n_gauss1 = random.randint(0,100)
    n_gauss2 = random.randint(0,100)
    n_med = random.randint(0,100)
    n_bil1 = random.randint(0,100)
    n_bil2 = random.randint(0,100)
    n_bil3 = random.randint(0,100)
    float = random.uniform(0.1, 30.0)

    for (file_color, file_depth) in zip (os.listdir(directory_color), os.listdir(directory_depth)):
        if file_color.endswith(".png") or file_depth.endswith(".png"):
            img_color = cv2.imread(os.path.join(directory_color, file_color))
            img_depth =  cv2.imread(os.path.join(directory_depth, file_depth))

            blur_avg_color = cv2.blur(img_color, (n_avg1, n_avg2))
            blur_avg_depth = cv2.blur(img_depth, (n_avg1, n_avg2))

            blur_gaussian_color = cv2.GaussianBlur(img_color, (n_gauss1, n_gauss2), float, float)
            blur_gaussian_depth = cv2.GaussianBlur(img_depth, (n_gauss1, n_gauss2), float, float)

            median_blur = cv2.medianBlur(img_color, n_med)
            median_blur = cv2.medianBlur(img_depth, n_med)

            bilateral_color = cv2.bilateralFilter(img_color, n_bil1, n_bil2, n_bil3)
            bilateral_depth = cv2.bilateralFilter(img_depth, n_bil1, n_bil2, n_bil3)

            path_color = "kinova_color_blur_images"
            if not os.path.exists(path_color):
                os.makedirs(path)
            cv2.imwrite(os.path.join(path_color, file_color), img_color)

            path_depth = "kinova_depth_blur_images"
            if not os.path.exists(path_depth):
                os.makedirs(path)
            cv2.imwrite(os.path.join(path_depth, file_depth), img_depth)


    # for file_color in os.listdir(directory_color):
    #     if file_color.endswith(".png"):
    #         img_color = cv2.imread(os.path.join(directory_color, file_color))
    #         blur_average = cv2.blur(img_color, (n_avg1, n_avg2))
    #         blur_gaussian = cv2.GaussianBlur(img_color, (n_gauss, n_gauss), float, float)
    #
    #         median_blur = cv2.medianBlur(img_color, n_med)
    #         bilateral = cv2.bilateralFilter(img_color, n_bil1, n_bil2, n_bil3)
    #
    #         path = "kinova_color_blur_images"
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         cv2.imwrite(os.path.join(path, file_color), img_color)
    #     else:
    #         #do smth
    #         break
    # for file_depth in os.listdir(directory_depth):
    #     if file_depth.endswith(".png"):
    #         img_depth =  cv2.imread(os.path.join(directory_depth, file_depth))
    #         blur_average = cv2.blur(img_depth, (n_avg1, n_avg2))
    #         blur_gaussian = cv2.GaussianBlur(img_depth, (n_gauss1, n_gauss2), float, float)
    #
    #         median_blur = cv2.medianBlur(img_depth, n_med)
    #         bilateral = cv2.bilateralFilter(img_depth, n_bil1, n_bil2, n_bil3)
    #
    #         path = "kinova_depth_blur_images"
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         cv2.imwrite(os.path.join(path, file_depth), img_depth)

##############################################
##############################################
# RESIZING
##############################################
##############################################

#######################
# PIL WAY
#######################

image = Image.open('kinova_color_images/myframe000320.png')
new_image = image.resize((480,270))
new_image.save('resized_myframe000320.png')

#######################
# OpenCV WAY
#######################

img = cv2.imread('kinova_color_images/myframe000320.png')
scale_percent = 220 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

##############################################
##############################################
# BLURRING
##############################################
##############################################


#######################
# SKIMAGE WAY
#######################
# get filename and kernel size from command line
# filename = sys.argv[1]
# sigma = float(sys.argv[2])

# read and display original image
image = skimage.io.imread('kinova_color_images/myframe000320.png')
viewer = ImageViewer(image)
viewer.show()

# apply Gaussian blur, creating a new image
blurred = skimage.filters.gaussian(
    image, sigma=(4, 4), truncate=3.5, multichannel=True)

blurred_show = ImageViewer(blurred)
blurred_show.show()

#######################
# OpenCV WAY
#######################

img = cv2.imread('kinova_color_images/myframe000320.png')

directory = "kinova_color_images"

for filename in os.listdir(directory):
    if filename.endswith(".png"):
        n = random.randint(0,100)
        img = cv2.imread(os.path.join(directory,filename))
        blur_average = cv2.blur(img, (n,n))

        float = random.uniform(0.1, 30.0)
        blur_gaussian = cv2.GaussianBlur(img, (n,n), float, float)

        cv2.medianBlur(img, n)
        cv2.bilateralFilter(img,n,n,n)
        continue
    else:
        #do smth
        break

### averaging method -> convolving the image with a normalized box filter

#cv2.blur(img, kernel_size, dst, anchor, borderType)
    # dst => the output image of the same size and type as src.
    # anchor => an integer representing anchor point and it’s default value point is (-1, -1) -> the anchor is at the kernel center.
blur_average = cv2.blur(img,(30,30))

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur_average),plt.title('Blurred Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

### gaussian blurring -> using gaussian kernel

#cv2.GaussianBlur(src, dst, ksize, sigmaX, sigmaY, borderType=BORDER_DEFAULT)
    # ksize => kernel size, must be a tuple (width, height), must be ODD numbers
    # sigmaX => kernel standard deviation along X-axis (a double)
    # sigmaY => kernel standard deviation along Y-axis (a double)

#we can create our own gaussian kernel by using
    #cv2.GaussianKernel(src, ksize, sigma, ktype)
        # ksize => aperture linear size, must be ODD number and greater than 0
        # sigma => gaussian standard deviation
        # ktype => type of filter coefficients, it can be CV_32F or CV_64F (default)
blur_gaussian = cv2.GaussianBlur(img,(51,51), 10.0, 10.0)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur_gaussian),plt.title('Gaussian Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


### median filtering -> computes the median of all the pixels under the kernel window and the central pixel is replaced with this median value.

#cv2.medianBlur(src, dst, ksize)
    # ksize => aperture linear size, must be ODD number and greater than 0
median = cv2.medianBlur(img,41)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median),plt.title('Median Filtering')
plt.xticks([]), plt.yticks([])
plt.show()


### bilateral filtering -> highly effective in noise removal while keeping edges sharp, but the operation is slower

#cv2.bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, borderType)
    # d => diameter of each pixel neighborhood that is used during filtering, if it's negative, it's computed from sigmaSpace
    # sigmaColor => filter sigma in the color space
        # the larger the value, the farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color.
    # sigmaSpace => filter sigma in the coordinate space.
        # the larger the value, the farther pixels will influence each other as long as their colors are close enough # when d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.

bilateral = cv2.bilateralFilter(img,50,100,100)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(bilateral),plt.title('Bilateral Filtering')
plt.xticks([]), plt.yticks([])
plt.show()
