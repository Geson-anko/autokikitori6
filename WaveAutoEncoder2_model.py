import torch
import torch.nn as nn
import os
import random
import config
import pytorch_lightning as pl
from Additional_layers import ConvNorm1d,ConvTransposeNorm1d,DilatedCausalConv1d,nResBlocks1d,CausalConv1d

class GlobalEncoder(nn.Module):
    input_size:tuple = (1,config.channels,config.speak_length)
    out_channels = 16
    output_size:tuple = (1,config.speak_seq_len,out_channels,config.recognize_length)
    __winlen = config.recognize_length
    __overlap = config.overlap_length

    def __init__(self):
        super().__init__()
        
        # layers
        self.conv = ConvNorm1d(1,8,kernel_size=3,padding=1)
        self.DCC = DilatedCausalConv1d(8,self.out_channels,kernel_size=9,num_layers=3,divs=2,dropout=0)

    def forward(self,x:torch.Tensor):
        x = torch.relu(self.conv(x))
        x = self.DCC(x)
        x = x.transpose(2,1)
        x = x.unfold(1,self.__winlen,self.__overlap).contiguous()
        return x

class LocalEncoder(nn.Module):
    __outch = GlobalEncoder.out_channels
    input_size:tuple = (1,__outch,config.recognize_length)
    outch = 32
    output_size:tuple = (1,outch,1)

    def __init__(self):
        super().__init__()

        # layers
        poolconv0 = ConvNorm1d(self.__outch,64,kernel_size=43,stride=3)
        res0 = nResBlocks1d(64,64,kernel_size=3,nlayers=3)
        poolconv1 = ConvNorm1d(64,128,kernel_size=23,stride=3)
        res1 = nResBlocks1d(128,128,kernel_size=3,nlayers=3)
        poolconv2 = ConvNorm1d(128,256,kernel_size=18,stride=3)

        out_conv = nn.Conv1d(256,self.outch,15)

        self.layers = nn.Sequential(
            poolconv0,nn.ReLU(),
            res0,
            poolconv1,nn.ReLU(),
            res1,
            poolconv2,nn.ReLU(),
            out_conv,nn.Tanh(),
        )
    
    def forward(self,x:torch.Tensor):
        x = self.layers(x)
        return x

class LocalDecoder(nn.Module):
    input_size:tuple = LocalEncoder.output_size
    output_size:tuple = LocalEncoder.input_size
    __outch = output_size[1]
    __inch = input_size[1]
    def __init__(self):
        super().__init__()
        
        # layers
        upper0 = ConvTransposeNorm1d(self.__inch,256,kernel_size=15)
        res0 = nResBlocks1d(256,256,kernel_size=3,nlayers=3)
        upper1 = ConvTransposeNorm1d(256,128,kernel_size=18,stride=3)
        res1 = nResBlocks1d(128,128,kernel_size=3,nlayers=3)
        upper2 = ConvTransposeNorm1d(128,64,kernel_size=23,stride=3)
        res2 = nResBlocks1d(64,64,kernel_size=3,nlayers=3)
        upper3 = ConvTransposeNorm1d(64,32,kernel_size=43,stride=3)
        res3 = nResBlocks1d(32,self.__outch,kernel_size=3,nlayers=3)

        self.layers = nn.Sequential(
            upper0,nn.ReLU(),
            res0,
            upper1,nn.ReLU(),
            res1,
            upper2,nn.ReLU(),
            res2,
            upper3,nn.ReLU(),
            res3,
        )
    def forward(self,x:torch.Tensor):
        x = self.layers(x)
        return x

class GlobalDecoder(nn.Module):
    __inch = GlobalEncoder.out_channels*2
    input_size:tuple = GlobalEncoder.output_size
    output_size:tuple = GlobalEncoder.input_size

    __over = config.overlap_length
    __catshape0 = (-1,GlobalEncoder.out_channels,config.speak_length)

    def __init__(self):
        super().__init__()
        
        # layers
        self.DCC0 = DilatedCausalConv1d(self.__inch,32,kernel_size=7,num_layers=4,dropout=0)
        self.DCC1 = DilatedCausalConv1d(32,32,kernel_size=7,num_layers=4,dropout=0)
        self.out_conv = CausalConv1d(32,1,5)
    
    def forward(self,x:torch.Tensor):
        x = self.slide_concat(x)
        x = self.DCC0(x)
        x = self.DCC1(x)
        x = torch.tanh(self.out_conv(x))
        return x

    def slide_concat(self,x:torch.Tensor):
        """
        x: shape (B,C,L) -> (-1,64,16,640)
        return -> (-1,32,20800)
        """
        x = x.transpose(2,1)
        x0 = x[:,:,:,:self.__over]
        x1 = x[:,:,:,self.__over:]
        x0_pad = x1[:,:,-1:]
        x1_pad = x0[:,:,:1]
        x0 = torch.cat([x0,x0_pad],dim=2).view(self.__catshape0)
        x1 = torch.cat([x1_pad,x1],dim=2).view(self.__catshape0)
        x = torch.cat([x0,x1],dim=1)
        return x

class WaveEncoder(nn.Module):
    input_size:tuple = GlobalEncoder.input_size
    output_size:tuple = (1,config.speak_seq_len,*LocalEncoder.output_size[1:])

    __locin = (-1,*LocalEncoder.input_size[1:])
    __locout = (-1,*output_size[1:])
    def __init__(self):
        super().__init__()

        # layers
        self.global_encoder = GlobalEncoder()
        self.local_encoder = LocalEncoder()

    def forward(self,x:torch.Tensor):
        x = self.global_encoder(x)
        x = x.view(self.__locin)
        x = self.local_encoder(x)
        x = x.view(self.__locout)
        return x

class WaveDecoder(nn.Module):
    input_size:tuple = WaveEncoder.output_size
    output_size:tuple = WaveEncoder.input_size

    __locin:tuple = (-1,*LocalDecoder.input_size[1:])
    __locout:tuple = (-1,*GlobalDecoder.input_size[1:])

    def __init__(self):
        super().__init__()

        # layers 
        self.local_decoder = LocalDecoder()
        self.global_decoder = GlobalDecoder()

    def forward(self,x:torch.Tensor):
        x = x.view(self.__locin)
        x = self.local_decoder(x)
        x = x.view(self.__locout)
        x = self.global_decoder(x)
        return x

    def freeze(self):
        self.eval()
        for i in self.parameters():
            i.requires_grad = False

class WaveAutoEncoder(pl.LightningModule):
    input_size:tuple = WaveEncoder.input_size
    output_size:tuple = input_size

    def __init__(self,lr:float=0.0001):
        super().__init__()
        self.reset_seed()
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.encoder = WaveEncoder()
        self.decoder = WaveDecoder()
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)
    
    def training_step(self,batch,index):
        data, = batch
        out = self(data)
        loss = self.criterion(out,data)
        self.log('WaveAutoEncoder2 loss',loss)
        return loss

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        

if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    from torchsummaryX import summary
    model = WaveAutoEncoder()
    dummy = torch.randn(model.input_size)
    summary(model,dummy)
    print(model(dummy).shape)

    #writer = SummaryWriter()
    #writer.add_graph(model,dummy)
    #writer.close()