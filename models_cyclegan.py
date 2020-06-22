# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:35:36 2020

@author: Anand Jebakumar
"""

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import LeakyReLU, Sigmoid
from torch.nn import InstanceNorm2d
from torch.nn import Sequential
import torch
import functools
from torch import nn
from torch.nn import ReflectionPad2d
from torch.nn import ReLU
from torch.nn import ConvTranspose2d
from torch.nn import Tanh

# discriminator based on Jason's work
# for a 70x70 input, the discriminator output should be 1; this doesn't make sense

class Discriminator(Module):
    def __init__(self):
        super().__init__()
        # initialization to be figured out
        layers = []
        # for a kernal size of 4, padding of 1 and stride of 2,
        # the size of image H decreases by a factor 2 (provided H is an even number)
        # layer 1
        conv1 = Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1) 
        act1  = LeakyReLU(0.2) #  the original paper uses in place = True
        layers.append(Sequential(conv1,act1))
        # layer 2
        conv2 = Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1) 
        norm2 = InstanceNorm2d(128) # out channels of previous layers
        act2  = LeakyReLU(0.2)
        layers.append(Sequential(conv2,norm2,act2))
        # layer 3
        conv3 = Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1) 
        norm3 = InstanceNorm2d(256) # out channels of previous layers
        act3  = LeakyReLU(0.2)
        layers.append(Sequential(conv3,norm3,act3))
        # layer 4
        conv4 = Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=1) 
        norm4 = InstanceNorm2d(512) # out channels of previous layers
        act4  = LeakyReLU(0.2)
        layers.append(Sequential(conv4,norm4,act4))
        # layer 5
        conv5 = Conv2d(in_channels=512,out_channels=512,kernel_size=4,stride=1,padding=1) 
        norm5 = InstanceNorm2d(512) # out channels of previous layers
        act5  = LeakyReLU(0.2)
        layers.append(Sequential(conv5,norm5,act5))
        # layer 6 - patch output; changed stride to 2
        conv6 = Conv2d(in_channels=512,out_channels=1,kernel_size=4,stride=2,padding=1) 
        act6 = Sigmoid()
        layers.append(Sequential(conv6, act6))        
        self.model = Sequential(* layers)
        
    def forward(self, X):
        X = self.model(X)
        # average patches
        X = X.mean(dim=(2,3))
        return X
        
d = Discriminator()
inp = torch.randn((16,3,256,256))
out = d(inp)
#print('expected discriminator output: 1,1,2,2')
#print('----discriminator-----')
#print(out.size())

# resnet block based on Alexei Efros; Jason had same padding instead of reflection2d
class ResNetBlock(Module):
    def __init__(self,num_ch):
        super().__init__()
        refl_pad1  = ReflectionPad2d(1)
        conv1      = Conv2d(num_ch,num_ch,kernel_size=3,padding=0,stride=1)
        norm1      = InstanceNorm2d(num_ch)
        act1       = ReLU()
        refl_pad2  = ReflectionPad2d(1)
        conv2      = Conv2d(num_ch,num_ch,kernel_size=3,padding=0,stride=1)
        norm2      = InstanceNorm2d(num_ch)    
        self.block = Sequential(refl_pad1,conv1,norm1,act1,refl_pad2,conv2,norm2)
    
    def forward(self,X):
        out = X+self.block(X)
        return out

r = ResNetBlock(256)
inp = torch.randn((1,256,64,64))
out = r(inp)
#print('expected resnet block output: 1,256,64,64')
#print(out.size())

# generator has a encoder-decoder architecture (symmetric)
# first conv layer increases num_channels from 3 to 64; size is constant
# conv layers 2 and 3 each doubles num_channels and halves size 
# layer 4 - there are 9 resnet blocks
# convTranspose layers 5 and 6 each halves the num_channels and double size
# conv layer 7 decreases num_channels from 64 to 3; size is constant
class Generator(Module):
    def __init__(self):
        super().__init__()
        in_ch     = [3,64,128]
        out_ch    = [64,128,256]
        resnet_ch = 256
        num_resnet_blocks = 3
        layers = []
        # layer 1
        rpad1 = ReflectionPad2d(3)
        conv1 = Conv2d(in_ch[0],out_ch[0],kernel_size=7,padding=0,stride=1)
        norm1 = InstanceNorm2d(out_ch[0])
        act1  = ReLU()
        layers.append(Sequential(rpad1,conv1,norm1,act1))
        # layer 2
        conv2 = Conv2d(in_ch[1],out_ch[1],kernel_size=3,padding=1,stride=2)
        norm2 = InstanceNorm2d(out_ch[1])
        act2  = ReLU()
        layers.append(Sequential(conv2,norm2,act2))
        # layer 3
        conv3 = Conv2d(in_ch[2],out_ch[2],kernel_size=3,padding=1,stride=2)
        norm3 = InstanceNorm2d(out_ch[2])
        act3  = ReLU()
        layers.append(Sequential(conv3,norm3,act3))
        # layer 4
        for i in range(num_resnet_blocks):
            layers.append(ResNetBlock(resnet_ch))
        # layer 5 - reversing out_ch and in_ch since we are decoding; output_padding=1 to get the right o/p size
        convTr5 = ConvTranspose2d(out_ch[2],in_ch[2],kernel_size=3,padding=1,stride=2,output_padding=1)
        norm5   = InstanceNorm2d(in_ch[2])
        act5    = ReLU()
        layers.append(Sequential(convTr5,norm5,act5))
        # layer 6
        convTr6 = ConvTranspose2d(out_ch[1],in_ch[1],kernel_size=3,padding=1,stride=2,output_padding=1)
        norm6   = InstanceNorm2d(in_ch[1])
        act6    = ReLU()
        layers.append(Sequential(convTr6,norm6,act6))
        # layer 7
        rpad7 = ReflectionPad2d(3)
        conv7 = Conv2d(out_ch[0],in_ch[0],kernel_size=7,padding=0,stride=1)
        norm7 = InstanceNorm2d(in_ch[0])
        act7  = Tanh()
        layers.append(Sequential(rpad7,conv7,norm7,act7))
        self.model = Sequential(* layers)
        
    def forward(self, X):
        return self.model(X)
        
g = Generator()
inp = torch.randn((1,3,256,256))
out = g(inp)
#print('expected generator block output: 1,3,256,256')
#print(out.size())

# default discriminator found in Alexei Efros' work (original author - Prof at UC Berkeley)
# for a 70x70 input, the discriminator output should be 1; this doesn't make sense
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        #print(kw,padw,ndf*nf_mult)
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

d = NLayerDiscriminator(3)
inp = torch.randn((1,3,70,70))
out = d(inp)
#print('expected discriminator output: 1,1,6,6')
#print(out.size())