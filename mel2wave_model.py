import os
import random
import torch.nn as nn
import torch
import math
import config
from typing import Union,Tuple
import pytorch_lightning as pl
import copy

class ConvNorm1d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 0, 
                dilation: Union[int, Tuple[int]]= 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class ChannelAttention(nn.Module): 
    # please refer to SENet.
    def __init__(self,channels:int):
        super().__init__()
        self.convnorm = ConvNorm1d(channels,channels,1)
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(channels,channels,1)
    
    def forward(self,x:torch.Tensor):
        x_a = torch.relu(self.convnorm(x))
        x_a = self.GAP(x_a)
        x_a = self.conv(x_a)
        x_a = torch.sigmoid(x_a)
        x = x*x_a
        return x


class DilatedDepthUnit(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:int,
        num_layers:int = 4,
        dilation_rate:int =2,
        divs:int =2,
        end_activation=torch.relu,
        channel_attn:bool = True) -> None:
        super().__init__()
        assert (kernel_size%2==1)
        assert (in_channels>1)
        self.ef = end_activation
        
        dilations = [dilation_rate**i for i in range(num_layers)]
        pads = [int((kernel_size*i-i)/2) for i in dilations]
        _c0 = in_channels//divs
        
        
        self.fconv = ConvNorm1d(in_channels,_c0,1)
        self.cconv = ConvNorm1d(in_channels,out_channels,1)

        self.convs = nn.ModuleList([ConvNorm1d(_c0,_c0,kernel_size,dilation=dilations[i],padding=pads[i]) for i in range(num_layers)])

        # channels attention
        self.enabls_ch_attn = channel_attn
        if channel_attn:
            self.ch_attn = ChannelAttention(_c0)

        self.oconv = ConvNorm1d(_c0,out_channels,1)

        self.end_norm = nn.BatchNorm1d(out_channels)

    def forward(self,x):
        x_fork = self.cconv(x)
        
        x_o = self.fconv(x)
        x = torch.relu(x_o)
        for l in self.convs:
            _x = l(x)
            x = torch.relu(torch.add(_x,x_o))
            x_o = _x.clone()
        if self.enabls_ch_attn:
            x = self.ch_attn(x)
        x = self.oconv(x)
        x = self.end_norm(torch.add(x,x_fork))
        x = self.ef(x)
        return x

class DilatedWideUnit(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:int,
        num_layers:int = 4,
        dilation_rate:int =2,
        divs:int=2,
        end_activation=torch.relu) -> None:
        super().__init__()
        assert (kernel_size%2==1)
        assert (in_channels>1)
        self.ef = end_activation
        self.num_layers = num_layers
        
        _c0 = in_channels//divs
        dilations = [dilation_rate**i for i in range(num_layers)]
        pads = [int((kernel_size*i-i)/2) for i in dilations]
        self.fconv = ConvNorm1d(in_channels,_c0,1)
        self.cconv = ConvNorm1d(in_channels,out_channels,1)
        self.convs = nn.ModuleList([ConvNorm1d(_c0,_c0,kernel_size,dilation=dilations[i],padding=pads[i]) for i in range(num_layers)])
        self.oconv = ConvNorm1d(_c0,out_channels,1)

        self.end_norm = nn.BatchNorm1d(out_channels)

    def forward(self,x):
        x_fork = self.cconv(x)
        x_init = torch.relu(self.fconv(x))
        x = x_init/self.num_layers
        for l in self.convs:
            x = torch.add(l(x_init)/self.num_layers,x)
        x = torch.relu(x)
        x = self.oconv(x)
        x = torch.add(x_fork,x)
        x = self.end_norm(x)
        x = self.ef(x)
        return x

class Mel2Wave(pl.LightningModule):
    input_size:tuple = (1,config.mel_channels,config.speak_seq_len)
    output_size:tuple = (1,1,config.speak_length)
    
    def __init__(self,lr:float=0.001):
        super().__init__()
        self.reset_seed()
        self.lr = lr

        init_ch = self.input_size[1]
        self.criterion = nn.MSELoss()
        # layers 
        upper_x13 = nn.Upsample(scale_factor=13)
        upper_x5 = nn.Upsample(scale_factor=5)
        conv0 = DilatedDepthUnit(init_ch,256,7,4,divs=2,channel_attn=True)
        conv1 = DilatedDepthUnit(256,512,3,5,divs=2,channel_attn=True)
        conv2 = DilatedDepthUnit(512,128,5,5,divs=4,channel_attn=True)
        conv3 = DilatedDepthUnit(128,32,5,3,divs=4,channel_attn=True)
        out_conv = nn.Conv1d(32,1,3,padding=1)

        self.layers = nn.Sequential(
            upper_x5,
            conv0,
            conv1,
            copy.deepcopy(upper_x5),
            conv2,
            upper_x13,
            conv3,
            out_conv,
            nn.Tanh(),
        )

    def forward(self,x:torch.Tensor):
        x= self.layers(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        return optimizer

    def training_step(self,batch,idx):
        data,ans = batch
        out = self(data)
        loss = self.criterion(out,ans)
        self.log('train loss',loss)
        return loss

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    from torchsummaryX import summary
    
    model = Mel2Wave()
    dummy =torch.randn(model.input_size)
    summary(model,dummy)

    #writer = SummaryWriter()
    #writer.add_graph(model,dummy)
    #writer.close()