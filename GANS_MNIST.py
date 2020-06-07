# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 09:37:32 2020

@author: guru_
"""
import os
path =  'C:\\Users\\guru_\\pytorch\\ModelSaveFolder'
import torch
from torch import nn, optim
from torchvision import transforms, datasets
import logger
device = torch.device(0)
def mnistData():
    compose = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(0.5, 0.5)])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, 
                          transform=compose, download=True)

data = mnistData()
batch_size = 1000
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
num_batches = len(data_loader)
class DiscriminatorNet(nn.Module):
    def __init__(self):
        #super(DiscriminatorNet, self).__init__()
        super().__init__()
        self.hidden1 = nn.Linear(784, 1024)
        self.act1 = nn.LeakyReLU(0.3)
        self.drop1 = nn.Dropout(0.2)
        self.hidden2 = nn.Linear(1024, 512)
        self.act2 = nn.LeakyReLU(0.3)
        self.drop2 = nn.Dropout(0.2)
        self.hidden3 = nn.Linear(512, 256)
        self.act3 = nn.LeakyReLU(0.3)
        self.drop3 = nn.Dropout(0.2)
        self.hidden4 = nn.Linear(256, 1)
        self.act4 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.hidden3(x)
        x = self.act3(x)
        x = self.drop3(x)
        x = self.hidden4(x)
        x = self.act4(x)
        return x

class GeneratorNet(nn.Module):
    def __init__(self):
        #super(GeneratorNet, self).__init__()
        super().__init__()
        self.hidden1 = nn.Linear(100, 256)
        self.act1 = nn.LeakyReLU(0.3)
        self.drop1 = nn.Dropout(0.2)
        self.hidden2 = nn.Linear(256, 512)
        self.act2 = nn.LeakyReLU(0.3)
        self.drop2 = nn.Dropout(0.2)
        self.hidden3 = nn.Linear(512, 784)
        self.act3 = nn.Tanh()
    
    def forward(self, x):
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.drop1(x) 
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.hidden3(x)
        x = self.act3(x)
        return x
generator = GeneratorNet()
generator.load_state_dict(torch.load(os.path.join(path,"generator1.pth")))
generator.to(device)
discriminator = DiscriminatorNet()
discriminator.load_state_dict(torch.load(os.path.join(path,"discriminator1.pth")))
discriminator.to(device)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
loss = nn.BCELoss()
log = logger.Logger(model_name='VGAN', data_name='MNIST')
#cudnn.benchmark = True

num_epoch = 200
for epoch in range(num_epoch):
    for nbatch, (real_batch,_) in enumerate(data_loader):
        # Get Real data
        x_i = real_batch.view(batch_size, 784)
        x_i = x_i.to(device)
        # Get random sample and create fake data
        z_i =  torch.randn(batch_size, 100)
        z_i = z_i.to(device)
        gz_i = generator(z_i).detach()
        # Discriminator training with real data
        d_optimizer.zero_grad()
        d_x_i = discriminator(x_i)
        error_x_i = loss(d_x_i, torch.ones(batch_size).cuda())
        error_x_i.backward()
        # Discriminator training with fake data
        d_gz_i = discriminator(gz_i)
        error_gz_i = loss(d_gz_i, torch.zeros(batch_size).cuda())
        error_gz_i.backward()
        ## update weights
        d_optimizer.step()
        # Generator training
        g_optimizer.zero_grad()
        # Get random sample and create fake data
        z_i =  torch.randn(batch_size, 100).cuda()
        gz_i = generator(z_i)
        d_gz_i = discriminator(gz_i)
        error_d_gz_i = loss(d_gz_i, torch.ones(batch_size).cuda())
        error_d_gz_i.backward()
        g_optimizer.step()
        ##
        log.log(error_x_i+error_gz_i, error_d_gz_i, epoch, nbatch, num_batches)
        if (nbatch) % 200 == 0: 
            test_noise = torch.randn(8,100).cuda()
            test_images = generator(test_noise)
            test_images = test_images.view(test_noise.size(0),1,28,28)
            test_images = test_images.data
            log.log_images(
                test_images.cpu(), 8, 
                epoch, nbatch, num_batches
            );
            # Display status Logs
            log.display_status(
                epoch, num_epoch, nbatch, num_batches,
                error_x_i+error_gz_i, error_d_gz_i, d_x_i, d_gz_i
            )

torch.save(discriminator.state_dict(), os.path.join(path,"discriminator1.pth"))
torch.save(generator.state_dict(), os.path.join(path,"generator1.pth"))
