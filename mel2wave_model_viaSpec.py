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

class ConvNorm2d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 0, 
                dilation: Union[int, Tuple[int]]= 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class Conv2dAttn(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,
                kernel_size: Union[int,Tuple[int]],padding:Union[int, Tuple[int]] = 0):
        super().__init__()
        
        self.conv = ConvNorm2d(in_channels,out_channels,kernel_size,padding=padding)
        self.attn_conv = ConvNorm2d(out_channels,out_channels,kernel_size=1,padding=0)

    def forward(self,x:torch.Tensor):
        x = torch.relu(self.conv(x))
        attn_x = torch.sigmoid(self.attn_conv(x))

        x = x * attn_x
        return x

class Mel2Spec(nn.Module):
    input_size:tuple = (1,config.mel_channels,config.speak_seq_len)
    output_size:tuple = (1,config.speak_seq_len,config.fft_channels,2)

    __melch = config.mel_channels
    __fftch = config.fft_channels

    def __init__(self):
        super().__init__()
        #  layers
        self.channel_conv = ConvNorm1d(self.__melch,self.__fftch,3,padding=1)
        conv0 = Conv2dAttn(1,16,7,3)
        conv1 = Conv2dAttn(16,32,5,2)
        conv2 = Conv2dAttn(32,64,3,1)
        self.layers = nn.Sequential(
            conv0,
            conv1,
            conv2,
        )
        self.out_conv= nn.Conv2d(64,2,3,padding=1)
    
    def forward(self,x:torch.Tensor):
        x = torch.relu(self.channel_conv(x))
        x = x.unsqueeze(1)
        x = self.layers(x)
        x = torch.tanh(self.out_conv(x)).permute(0,3,2,1).contiguous()
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

class Mel2Wave(pl.LightningModule):
    input_size:tuple = (1,config.mel_channels,config.speak_seq_len)
    output_size:tuple = (1,1,config.speak_length)

    __over = config.overlap_length

    def __init__(self,lr:float=1e-4):
        super().__init__()
        self.reset_seed()
        self.lr = lr
        self.criterion = nn.MSELoss()
        #layers

        self.mel2spec = Mel2Spec()
        self.ch_conv= ConvNorm1d(2,8,kernel_size=3,padding=1)
        self.DDU0 = DilatedDepthUnit(8,16,7,4,divs=1)
        self.out_conv = nn.Conv1d(16,1,kernel_size=5,padding=2)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        pad = torch.zeros(x.size(0),1,self.__over,dtype=dtype,device=x.device)
        
        x = self.mel2spec(x).float()
        x = torch.view_as_complex(x)
        x = torch.fft.irfft(x).type(dtype)
        out0 = x[:,:,:self.__over].reshape(-1,1,config.speak_chunk_len)
        out1 = x[:,:,self.__over:].reshape(-1,1,config.speak_chunk_len)
        out0 = torch.cat([out0,pad],dim=-1)
        out1 = torch.cat([out1,pad],dim=-1)
        x = torch.cat([out0,out1],dim=1)
        x = torch.relu(self.ch_conv(x))
        x = self.DDU0(x)
        x = torch.tanh(self.out_conv(x))
        return x
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        return optimizer

    def training_step(self,batch,idx):
        data,ans = batch
        out = self(data)
        loss = self.criterion(out,ans)
        self.log('mel2wave train loss',loss)
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

    writer = SummaryWriter()
    writer.add_graph(model,dummy)
    writer.close()
