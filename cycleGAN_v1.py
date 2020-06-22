# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 08:39:54 2020

@author: guru_
"""
import torch
import time
import numpy as np
from torch import nn, optim
from models_cyclegan import Generator, Discriminator
import os
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torch.utils.data as data
path =  'C:\\Users\\guru_\\pytorch\\ModelSaveFolder'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## dataset
batch_size = 4
X_DATA_PATH = 'C:\\Users\\guru_\\pytorch\\horse2zebra\\trainA'
Y_DATA_PATH = 'C:\\Users\\guru_\\pytorch\\horse2zebra\\trainB'
X_transforms = transforms.Compose([
            transforms.ToTensor()])
X_data = datasets.ImageFolder(root=X_DATA_PATH, transform = X_transforms)
X_data_loader = data.DataLoader(X_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
Y_transforms = transforms.Compose([
                transforms.ToTensor()])
Y_data =datasets.ImageFolder(root=Y_DATA_PATH, transform = Y_transforms)
Y_data_loader  = data.DataLoader(Y_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0) 
to_pil = transforms.ToPILImage()

def do_sample(data_loader):
    (Sample, _) = next(iter(data_loader))
    return Sample

g = Generator()
f = Generator()
dX = Discriminator()
dY = Discriminator()
try:
    g.load_state_dict(torch.load(os.path.join(path,"g_cycleGAN_v1.pth")))
    f.load_state_dict(torch.load(os.path.join(path,"f_cycleGAN_v1.pth")))
    dX.load_state_dict(torch.load(os.path.join(path,"dX_cycleGAN_v1.pth")))
    dY.load_state_dict(torch.load(os.path.join(path,"dY_cycleGAN_v1.pth")))
    print("Loading model...")
except FileNotFoundError:
    print("Training from scratch...")
    
g.to(device)
f.to(device)
dX.to(device)
dY.to(device)
g_optimizer = optim.Adam(g.parameters(), lr=0.0002)
f_optimizer = optim.Adam(f.parameters(), lr=0.0002)
dX_optimizer = optim.Adam(dX.parameters(), lr=0.0002)
dY_optimizer = optim.Adam(dY.parameters(), lr=0.0002)

adversarial_loss = nn.BCELoss()
consistency_loss = nn.L1Loss()
loss_log = np.empty([])
t1 = time.time()
for epoch in range(100):
    loss_per_epoch = np.empty([])
    for ntrain in range(3):
        # Train dY
        x = do_sample(X_data_loader).to(device)
        yhat = g(x).detach()
        y = do_sample(Y_data_loader).to(device)
        
        dY_optimizer.zero_grad()
        y_prob = dY(y) #y_prob is the probability that the discriminator outputs 
        error_real_y = adversarial_loss(y_prob, torch.ones(batch_size, device=device))
        loss_per_epoch = np.append(loss_per_epoch, error_real_y.cpu().detach().numpy())
        error_real_y.backward()
        
        y_prob = dY(yhat)
        error_fake_y = adversarial_loss(y_prob, torch.zeros(batch_size, device=device))
        loss_per_epoch = np.append(loss_per_epoch, error_fake_y.cpu().detach().numpy())
        error_fake_y.backward(retain_graph=True)
        dY_optimizer.step()
    
    # Train G1
    x = do_sample(X_data_loader).to(device)
    g_optimizer.zero_grad()
    yhat = g(x)
    y_prob = dY(yhat)
    error_g_generator = adversarial_loss(y_prob, torch.ones(batch_size, device=device))
    loss_per_epoch = np.append(loss_per_epoch, error_g_generator.cpu().detach().numpy())
    error_g_generator.backward()
    g_optimizer.step()

    for ntrain in range(3):    
        # Train D2
        y = do_sample(Y_data_loader).to(device)
        xhat = f(y).detach()
        x = do_sample(X_data_loader).to(device)
        
        dX_optimizer.zero_grad()
        x_prob = dX(x)
        error_real_x = adversarial_loss(x_prob, torch.ones(batch_size, device=device))
        loss_per_epoch = np.append(loss_per_epoch, error_real_x.cpu().detach().numpy())
        error_real_x.backward()
        x_prob = dX(xhat)
        error_fake_x = adversarial_loss(x_prob, torch.zeros(batch_size, device=device))
        loss_per_epoch = np.append(loss_per_epoch, error_fake_x.cpu().detach().numpy())
        error_fake_x.backward()
        dX_optimizer.step()
    
    # Train G2
    y = do_sample(Y_data_loader).to(device)
    f_optimizer.zero_grad()
    xhat = f(y)
    x_prob = dX(xhat)
    error_f_generator = adversarial_loss(x_prob, torch.ones(batch_size, device=device))
    loss_per_epoch = np.append(loss_per_epoch, error_f_generator.cpu().detach().numpy())
    error_f_generator.backward()
    f_optimizer.step()
    
    #X loop
    f_optimizer.zero_grad()
    g_optimizer.zero_grad()
    yhat = g(x)
    xhat = f(yhat)
    error_fg_x = consistency_loss(xhat,x)
    loss_per_epoch = np.append(loss_per_epoch, error_fg_x.cpu().detach().numpy())
    error_fg_x.backward()
    
    #Y loop
    #f_optimizer.zero_grad()
    #g_optimizer.zero_grad()
    xhat = f(y)
    yhat = g(xhat)
    error_gf_y = consistency_loss(yhat, y)
    loss_per_epoch = np.append(loss_per_epoch, error_gf_y.cpu().detach().numpy())
    error_gf_y.backward()

    g_optimizer.step()
    f_optimizer.step()
    
    loss_log = np.append(loss_log, [loss_per_epoch])
    if epoch % 10 == 0:
        torch.save(g.state_dict(), os.path.join(path,"g_cycleGAN_v1.pth"))
        torch.save(f.state_dict(), os.path.join(path,"f_cycleGAN_v1.pth"))
        torch.save(dX.state_dict(), os.path.join(path,"dX_cycleGAN_v1.pth"))
        torch.save(dY.state_dict(), os.path.join(path,"dY_cycleGAN_v1=.pth"))
        print("Loss per epoch: {}\n".format(loss_per_epoch))
        print("Epoch {}".format(epoch))
        x_ = x[3].unsqueeze(dim=0)
        xhat_ = f(g(x_))
        y_ = y[3].unsqueeze(dim=0)
        yhat_ = g(f(x_))
        plt.figure(figsize=(10,10))
        plt.subplot(2,2,1)
        plt.imshow(to_pil(x_[0].cpu()))
        plt.subplot(2,2,2)
        plt.imshow(to_pil(xhat_[0].cpu()))
        plt.subplot(2,2,3)
        plt.imshow(to_pil(y_[0].cpu()))
        plt.subplot(2,2,4)
        plt.imshow(to_pil(yhat_[0].cpu()))
        plt.show()
        t2 = time.time()
        print("Time in s {}".format(t2-t1))
        t1 = time.time()

