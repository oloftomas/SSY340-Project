import torch
import numpy as np

import matplotlib.pyplot as plt
import cv2

import network
from network import UNet

import networksmall
from networksmall import UNetSmall

import torchvision
import torchvision.transforms as transforms

tensor_transform = transforms.ToTensor()

tdataset = torchvision.datasets.CIFAR10(root='./testinput', train=False, download = True, transform=tensor_transform)


generator = UNet(True)
generator.load_state_dict(torch.load('train_generator_big', map_location=torch.device('cpu')))

rgb_images = []
np_lab_images = []

for img,label in tdataset:
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

class LABTDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(lab_images)

    def __getitem__(self,index):
        img = lab_images[index]
        return img

lab_tdataset = LABTDataset()
lab_loader = torch.utils.data.DataLoader(lab_tdataset, batch_size=64, shuffle=False)

batch = next(iter(lab_loader))
print(batch.shape)

l_images = batch[:, 0:1, :, :]
c_images = batch[:, 1: , :, :]

print(l_images.shape)

#l_test_img = test_img[0:1, :, :]

#print(type(generator))
gen_imgs = generator(l_images)
gen_imgs_cat = torch.cat((l_images, gen_imgs), 1)

real_images = torch.cat((l_images, c_images), 1)

### 10 is good  
imgN = 55
gen_img = gen_imgs_cat[imgN]
real_img = real_images[imgN]

np_gen_img = gen_img.detach().numpy()
np_gen_img *= 255
np_gen_img = np_gen_img.astype('uint8')
np_gen_img = np.transpose(np_gen_img, (1,2,0))

np_real_img = real_img.detach().numpy()
np_real_img *= 255
np_real_img = np_real_img.astype('uint8')
np_real_img = np.transpose(np_real_img, (1,2,0))


img1 = cv2.cvtColor(np_gen_img, cv2.COLOR_LAB2RGB)
img2 = cv2.cvtColor(np_real_img, cv2.COLOR_LAB2RGB)

plt.imshow(img1)
plt.show()

plt.imshow(img2)
plt.show()
