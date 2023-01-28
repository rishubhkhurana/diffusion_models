import torch
import torch.nn as nn
from ..layers.conv_layers import *

class UpConvLayer(nn.Module): 
  def __init__(self, in_channels, out_channels, **kwargs):
    super().__init__()
    self.read(**kwargs)
    if self.conv_type == 'upsample':
      self.lyr = nn.Sequential(nn.UpsamplingNearest2d(scale_factor =2),
                               nn.Conv2d(in_channels, out_channels, kernel_size = self.k, 
                                         stride = self.stride,
                                         padding = self.padding, 
                                         dilation = self.dilation, 
                                         groups = self.groups, 
                                         bias = self.bias))
      
    elif self.conv_type == 'convtranspose':
      self.lyr = nn.ConvTranspose2d(in_channels, out_channels, 
                                    kernel_size= self.k, stride = self.stride,
                                    dilation= self.dilation, 
                                    groups = self.groups, 
                                    bias = self.bias)
    
  def read(self, **kwargs):
    self.conv_type = kwargs.get('conv_type', 'upsample')
    self.k = kwargs.get('k', 3)
    self.padding = kwargs.get('padding', 1)
    self.stride = kwargs.get('s', 1)
    self.groups = kwargs.get('groups', 1)
    self.dilation = kwargs.get('dilation', 1)
    self.bias = kwargs.get('bias', True)

  def forward(self, x):
    return self.lyr(x)