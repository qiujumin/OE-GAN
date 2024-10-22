import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

m = 1.0
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9
W = 1

wavelength = 632.8 * nm
d = 8 * um
Nx = 1080
Ny = 1080
extent_x = Nx * d
extent_y = Ny * d
intensity = 0.1 * W / (m**2)

z = 10 * cm

N = 4

class MonochromaticField:
    def __init__(self, wavelength, extent_x, extent_y, Nx, Ny, intensity=intensity):
        self.extent_x = extent_x
        self.extent_y = extent_y

        self.dx = extent_x / Nx
        self.dy = extent_y / Ny

        self.x = self.dx * (torch.arange(Nx) - Nx // 2)
        self.y = self.dy * (torch.arange(Ny) - Ny // 2)
        self.xx, self.yy = torch.meshgrid(self.x, self.y, indexing="xy")

        self.Nx = Nx
        self.Ny = Ny
        self.E = np.sqrt(intensity) * torch.ones((self.Ny, self.Nx)).to(device)
        self.λ = wavelength
        self.z = 0

    def modulate(self, amplitude, phase):
        self.E = amplitude * self.E * torch.exp(1j * phase)

    def source(self, amplitude):
        self.E = amplitude

    def propagate(self, z):
        fft_c = torch.fft.fft2(self.E)
        c = torch.fft.fftshift(fft_c)

        fx = torch.fft.fftshift(torch.fft.fftfreq(self.Nx, d=self.dx))
        fy = torch.fft.fftshift(torch.fft.fftfreq(self.Ny, d=self.dy))
        fxx, fyy = torch.meshgrid(fx, fy, indexing="xy")

        argument = (2 * torch.pi) ** 2 * ((1.0 / self.λ) ** 2 - fxx**2 - fyy**2)

        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j * tmp).to(device)

        self.z += z
        self.E = torch.fft.ifft2(torch.fft.ifftshift(c * torch.exp(1j * kz * z)))

    def get_intensity(self):
        return torch.real(self.E * torch.conj(self.E))


class Generator(nn.Module):
    def __init__(self, amplitude, phase):
        super(Generator, self).__init__()
        self.amplitude = amplitude
        self.phase = phase
        self.ln = nn.LayerNorm([Ny, Nx])

    def forward(self, x):
        F = MonochromaticField(wavelength, extent_x, extent_y, Nx, Ny)
        
        for i in range(3):
            F.source(x)
            F.propagate(z)
            F.modulate(torch.sigmoid(self.amplitude[i]), self.phase[i])
            F.propagate(z)
            x = F.get_intensity()

            x = self.ln(x)
            x = torch.clamp(x, 0, 1)

            x = nn.functional.avg_pool2d(x, 3)
            x = x.repeat_interleave(3, 2).repeat_interleave(3, 3)

        x = nn.functional.avg_pool2d(x, 38)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = x.view(x.size(0), 1, 28, 28)

        return x
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, *args):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Cond_Discriminator(Discriminator):
    def __init__(self):
        super(Cond_Discriminator, self).__init__()
        self.fc = nn.Linear(10 + 28 * 28, 28 * 28)

    def forward(self, x, y):
        y_one_hot = torch.zeros(len(y),10).to(device).scatter_(1, y.view(-1,1), 1)
        x = torch.cat((x.view(x.size(0), -1), y_one_hot), -1)
        x = self.fc(x)
        return super(Cond_Discriminator, self).forward(x, y)
