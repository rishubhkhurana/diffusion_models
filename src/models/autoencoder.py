import torch
import torch.nn as nn
from typing import List
from .layers.conv_layers import *
from ..utils import *

class AutoEncoder(nn.Module): 
    def __init__(self, in_channels: int, out_channels: 
               int, init_dims = 16, 
               mults: List[int] = [1,2,4], 
               input_pad = 2, 
               upconv_type: str = 'upsample'): 
        super().__init__()
        self.down_layers = nn.ModuleList()
        self.mid_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.padding_layer = nn.ZeroPad2d(input_pad) if exists(input_pad) else nn.Identity()
        self.init_conv = ConvLayer(in_channels, init_dims, k=3, stride =1, padding =1) # o = (i+2p-k)/s +1
        # channel
        channel_series = [init_dims, *map(lambda x: x*init_dims, mults)]
        in_out_channels = [(_in, _out) for _in, _out in zip(channel_series[:-1], channel_series[1:])]
        for _in, _out in in_out_channels: 
            self.down_layers.append(nn.ModuleList([ConvLayer(_in, _out), 
                                              nn.BatchNorm2d(_out),
                                              nn.ReLU(), 
                                              nn.MaxPool2d(2,2)]
                                              )
                                              )

        mid_dims = channel_series[-1]
        self.mid_layers = nn.ModuleList(
          [ConvLayer(mid_dims, mid_dims),
          nn.BatchNorm2d(mid_dims), 
          nn.ReLU(), 
          ConvLayer(mid_dims, mid_dims, k=1, s=1)]
          )
        for _in, _out in reversed(in_out_channels):
            self.up_layers.append(nn.ModuleList([UpConvLayer(_out, _in, conv_type = upconv_type), nn.BatchNorm2d(_in),nn.ReLU(),]))
        up_dims = init_dims
        self.to_out = ConvLayer(init_dims, out_channels, k=1, s=1)
        self.out_reverse_pad = nn.ZeroPad2d(-2) if exists(input_pad) else nn.Identity()
        self.out_act = nn.Sigmoid()
  
    def forward(self, x):
        x = self.padding_layer(x)
        x = self.init_conv(x)


        for conv, norm, act, down in self.down_layers:
            x = conv(x)
            x = norm(x)
            x = act(x)
            x = down(x)

        for lyr in self.mid_layers:
            x = lyr(x) 

        for upconv, norm, act in self.up_layers:
            x = upconv(x)
            x = norm(x)
            x = act(x)

        x = self.to_out(x)
        x = self.out_reverse_pad(x)
        x = self.out_act(x)
        return x