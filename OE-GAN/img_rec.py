from net import *
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image

input_dim = 100
batch_size = 32
num_epoch = 100

dataset = datasets.MNIST(root="data", train=True, transform=transforms.ToTensor(), download=False)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

amplitude = nn.ParameterList(nn.Parameter(torch.ones((Ny, Nx))) for i in range(3))
phase = nn.ParameterList(nn.Parameter(torch.randn((Ny, Nx))) for i in range(3))

transform = transforms.Compose([transforms.Resize((Ny,Nx)), transforms.ToTensor()])

G = Generator(amplitude, phase).to(device)
D = Cond_Discriminator().to(device)

optim_G = torch.optim.RMSprop(G.parameters(), lr=0.001)
optim_D = torch.optim.RMSprop(D.parameters(), lr=0.0001)

critertion = nn.MSELoss(reduction='sum')


for epoch in range(num_epoch):
    total_loss_D, total_loss_G = 0, 0
    for i, (x, y) in enumerate(dataloader):
        real_x = x.to(device)
        real_x = torch.clamp(real_x, 0.1, 1)
        y = y.to(device)

        #noise = torch.ones((len(x), 1, N, N)).to(device)
        #indices = torch.randint(N, (len(x), 2)).to(device)
        #noise[torch.arange(len(x)), :, indices[:, 0], indices[:, 1]] = 0

        noise = torch.randint(2, (len(x), 1, N, N)).float().to(device)
        noise = noise.repeat_interleave(Ny//N, 2).repeat_interleave(Nx//N, 3)
        input = transforms.Resize((Ny,Nx))(real_x) * noise
        input = torch.clamp(input, 0.1, 1)

        fake_x = G(input).detach()
        loss_D = -torch.mean(D(real_x, y)) + torch.mean(D(fake_x, y))

        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        for p in D.parameters():
            p.data.clamp_(-0.01, 0.01)

        noise = torch.randint(2, (len(x), 1, N, N)).float().to(device)
        noise = noise.repeat_interleave(Ny//N, 2).repeat_interleave(Nx//N, 3)
        input = transforms.Resize((Ny,Nx))(real_x) * noise 
        input = torch.clamp(input, 0.1, 1)

        fake_x = G(input).to(device)
        loss_G = -torch.mean(D(fake_x, y)) + 0.001*critertion(fake_x, real_x)

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        total_loss_D += loss_D
        total_loss_G += loss_G

        if (i + 1) % 100 == 0 or (i + 1) == len(dataloader):
            print('Epoch {:02d} | Step {:04d} / {} | Loss_D {:.4f} | Loss_G {:.4f}'.format(epoch, i + 1, len(dataloader), total_loss_D / (i + 1), total_loss_G / (i + 1)))

    noise = torch.randint(2, (len(x), 1, N, N)).float().to(device)

    noise = noise.repeat_interleave(Ny//N, 2).repeat_interleave(Nx//N, 3)
    input = transforms.Resize((Ny,Nx))(real_x) * noise
    input = torch.clamp(input, 0.1, 1)

    img = G(input)
    save_image(torch.cat((transforms.Resize((28,28))(input), img),dim=0), './data/results2/' + '%d_epoch.png' % epoch)
