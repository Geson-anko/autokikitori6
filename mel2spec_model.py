import os
import random
import torch.nn as nn
import torch
import config
from typing import Union,Tuple
import pytorch_lightning as pl

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

class Mel2Spec(pl.LightningModule):
    input_size:tuple = (1,config.mel_channels,config.speak_seq_len)
    output_size:tuple = (1,config.speak_seq_len,config.fft_channels,2)

    __melch = config.mel_channels
    __fftch = config.fft_channels

    def __init__(self,lr:float=0.001):
        super().__init__()
        self.reset_seed()
        self.lr = lr
        self.criterion = nn.MSELoss()
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
        x = self.out_conv(x).permute(0,3,2,1).contiguous()
        return x
                
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        return optimizer

    def training_step(self,batch,idx):
        data,ans = batch
        out = self(data)
        loss = self.criterion(out,ans)
        self.log('mel2spec train loss',loss)
        return loss

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    from torchsummaryX import summary
    model = Mel2Spec()
    dummy = torch.randn(model.input_size)
    summary(model,dummy)