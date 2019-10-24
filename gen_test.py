import torch
import numpy as np

import matplotlib.pyplot as plt
import cv2

import dataset
from dataset import *
import network
from network import UNet

import torchvision
import torchvision.transforms as transforms

tensor_transform = transforms.ToTensor()

dataset = torchvision.datasets.CIFAR10(root='./testinput', train=False, download = True, transform=tensor_transform)


generator = UNet(True)
generator.load_state_dict(torch.load('train_generator', map_location=torch.device('cpu')))

for i,data in enumerate(lab_loader):
    lab_images = data
    l_images = lab_images[:, 0:1, :, :]
    c_images = lab_images[:, 1: , :, :]


print(l_images.shape)

#l_test_img = test_img[0:1, :, :]

#print(type(generator))
gen_imgs = generator(l_images)
gen_imgs_cat = torch.cat((l_images, c_images), 1)

gen_img = gen_imgs_cat[0]

np_gen_img = gen_img.detach().numpy()
np_gen_img *= 255
np_gen_img = np_gen_img.astype('uint8')
np_gen_img = np.transpose(np_gen_img, (1,2,0))

img = cv2.cvtColor(np_gen_img, cv2.COLOR_LAB2RGB)

plt.imshow(img)
plt.show()
