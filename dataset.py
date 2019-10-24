import numpy as np
import matplotlib.pylab as plt
import cv2

import torch
import torchvision
import torchvision.transforms as transforms

tensor_transform = transforms.ToTensor()

dataset = torchvision.datasets.CIFAR10(root='./input', train=True, download = True, transform=tensor_transform)

# transform images to LAB color space in order to separate
# images into luminosity and chromatic

rgb_images = []
np_lab_images = []

for img,label in dataset:
    rgb_images.append(img)

# convert to LAB color space
for img in rgb_images:
    np_img = np.transpose(img.numpy(), (1,2,0))
    np_lab_image = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
    np_lab_images.append(np_lab_image)

# "normalize" and convert to tensors
lab_images = []
for np_lab_image in np_lab_images:
    np_lab_image[:, :, 0] *= 255 / 100
    np_lab_image[:, :, 1] += 128
    np_lab_image[:, :, 2] += 128
    np_lab_image /= 255
    torch_lab_image = torch.from_numpy(np.transpose(np_lab_image, (2, 0, 1)))
    lab_images.append(torch_lab_image)

class LABDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(lab_images)

    def __getitem__(self,index):
        img = lab_images[index]
        return img

lab_dataset = LABDataset()
lab_loader = torch.utils.data.DataLoader(lab_dataset, batch_size=64, shuffle=True)


'''
print((np_lab_images[0]))
img = np_lab_images[0,:,:,0]
img = img.astype('uint8')
img_ = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
#print(img_)
plt.imshow(img)
plt.show()
'''
