import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import torchvision.transforms as transforms

images_gray = np.load('input/l/gray_scale.npy')
images_lab = np.load('input/ab/ab/ab1.npy')

def rgb_from_lab(gray_imgs, ab_imgs, n=10):
    # empty array for storing images
    imgs = np.zeros((n,224,224,3))

    imgs[:,:,:,0] = gray_imgs[0:n:]
    imgs[:,:,:,1:] = ab_imgs[0:n:]

    # convert images to uint8
    imgs = imgs.astype("uint8")

    imgs_ = []

    for i in range(0,n):
        imgs_.append(cv2.cvtColor(imgs[i], cv2.COLOR_LAB2RGB))

    imgs_ = np.array(imgs_)

    print(imgs_.shape)

    return imgs_

rgb_from_lab(images_gray, images_lab)