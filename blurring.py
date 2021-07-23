"""
 * Python script to demonstrate Gaussian blur.
 *
 * usage: python GaussBlur.py <filename> <sigma>
"""
import skimage
from skimage.viewer import ImageViewer
import sys
# import cv2
import numpy as np
from matplotlib import pyplot as plt

#######################
# SKIMAGE WAY
#######################
# # get filename and kernel size from command line
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
#
# #######################
# # OpenCV WAY
# #######################
#
# img = cv2.imread('kinova_color_images/myframe000320.png')
#
# ### averaging method -> convolving the image with a normalized box filter
#
# #cv2.blur(img, kernel_size, dst, anchor, borderType)
#     # dst => the output image of the same size and type as src.
#     # anchor => an integer representing anchor point and it’s default value point is (-1, -1) -> the anchor is at the kernel center.
# blur_average = cv2.blur(img,(5,5))
#
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(blur_average),plt.title('Blurred Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()
#
# ### gaussian blurring -> using gaussian kernel
#
# #cv2.GaussianBlur(src, dst, ksize, sigmaX, sigmaY, borderType=BORDER_DEFAULT)
#     # ksize => kernel size, must be a tuple (width, height)
#     # sigmaX => kernel standard deviation along X-axis (a double)
#     # sigmaY => kernel standard deviation along Y-axis (a double)
#
# #we can create our own gaussian kernel by using
#     #cv2.GaussianKernel(ksize, sigma, ktype)
#         # ksize => aperture linear size, must be ODD number and greater than 0
#         # sigma => gaussian standard deviation
#         # ktype => type of filter coefficients, it can be CV_32F or CV_64F (default)
# blur_gaussian = cv2.GaussianBlur(img,(5,5), 0.1, 0.2)
#
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(blur_gaussian),plt.title('Gaussian Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()
#
#
# ### median filtering -> computes the median of all the pixels under the kernel window and the central pixel is replaced with this median value.
#
# #cv2.medianBlur(src, dst, ksize)
#     # ksize => aperture linear size, must be ODD number and greater than 0
# median = cv2.medianBlur(img,5) # added 50% noise to the original image
#
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(median),plt.title('Median Filtering')
# plt.xticks([]), plt.yticks([])
# plt.show()
#
#
# ### bilateral filtering -> highly effective in noise removal while keeping edges sharp, but the operation is slower
#
# #cv2.bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, borderType)
#     # d => diameter of each pixel neighborhood that is used during filtering, if it's negative, it's computed from sigmaSpace
#     # sigmaColor => filter sigma in the color space
#         # the larger the value, the farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color.
#     # sigmaSpace => filter sigma in the coordinate space.
#         # the larger the value, the farther pixels will influence each other as long as their colors are close enough # when d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
#
# bilateral = cv2.bilateralFilter(img,9,75,75)
#
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(bilateral),plt.title('Bilateral Filtering')
# plt.xticks([]), plt.yticks([])
# plt.show()
