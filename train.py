import torch
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable
from torch import cat

from network import UNet, DNet
import dataset
from dataset import *

# initialize generator and discriminator
gen = UNet(True)
disc = DNet()

device = torch.device("cuda" if torch.cuda.is_available() 
                             else "cpu")

gen.to(device)
disc.to(device)

# use Adam as optimizer for both nets
learning_rate = 0.001
g_optimizer = optim.Adam(gen.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(disc.parameters(), lr=learning_rate)

d_criterion = nn.BCELoss()
g_criterion1 = nn.BCELoss()
g_criterion2 = nn.L1Loss()

def train():
    # testing these values
    g_lambda = 100
    smooth = 0.1
    epochs = 10

    for epoch in range(epochs):
        d_run_loss = 0.0
        g_run_loss = 0.0

        for i, data in enumerate(lab_loader):
            print(i)
            lab_images = data
            # separete images into l and c channels
            l_images = lab_images[:, 0:1, :, :]
            c_images = lab_images[:, 1: , :, :]

            # generate fake images
            fake_images = gen(l_images)

            ### Train the discriminator ###
            batch_size = l_images.shape[0]
            d_optimizer.zero_grad()
            d_loss = 0
            logits = disc(cat([l_images, c_images], 1))
            d_real_loss = d_criterion(logits, ((1 - smooth) * torch.ones(batch_size)))

            logits = disc(cat([l_images, fake_images], 1))
            d_fake_loss = d_criterion(logits, (torch.zeros(batch_size)))

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            ### Train the generator ###
            g_optimizer.zero_grad()
            g_loss = 0
            fake_logits = disc(cat([l_images, fake_images], 1))
            g_fake_loss = g_criterion1(fake_logits, (torch.ones(batch_size)))

            g_l1_loss = g_lambda * g_criterion2(fake_images, c_images)

            g_loss = g_fake_loss + g_l1_loss
            g_loss.backward()
            g_optimizer.step()

            d_run_loss += d_loss
            g_run_loss += g_loss

            if i % 10 == 0:
                print('[%d, %5d] d_loss: %.5f g_loss: %.5f' %
                    (epoch + 1, i + 1, d_run_loss / 10, g_run_loss / 10))
                d_run_loss = 0.0
                g_run_loss = 0.0

        torch.save(gen.state_dict(), 'networkstates/cifar10_train_generator_bigx')
        torch.save(disc.state_dict(), 'networkstates/cifar10_train_discriminator_bigx')

    print("HURRA LES GO!")

train()
