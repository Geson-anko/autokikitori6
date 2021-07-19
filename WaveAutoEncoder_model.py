import torch
import torch.nn as nn
import os
import random
import config
import pytorch_lightning as pl
from Additional_layers import ConvNorm1d,ConvTransposeNorm1d

class WaveEncoder(nn.Module):
    input_size:tuple = (1,config.channels,config.recognize_length)
    output_size:tuple = (1,32,1)

    def __init__(self):
        super().__init__()

        # layers
        pool = nn.AvgPool1d(3)
        relu = nn.ReLU()
        conv0,conv1,conv2,conv3 = (
            ConvNorm1d(1,16,41),ConvNorm1d(16,32,21),ConvNorm1d(32,64,16),nn.Conv1d(64,32,15)
        )
        self.layers = nn.Sequential(
            conv0,relu,pool,
            conv1,relu,pool,
            conv2,relu,pool,
            conv3,nn.Tanh()
        )

    def forward(self,x):
        x = self.layers(x)
        return x
    
    def freeze(self):
        self.eval()
        for i in self.parameters():
            i.requires_grad = False

class WaveDecoder(nn.Module):
    input_size:tuple = WaveEncoder.output_size
    output_size:tuple = WaveEncoder.input_size
    __inch = input_size[1]

    def __init__(self):
        super().__init__()
        
        # layers
        upper = nn.Upsample(scale_factor=3)
        relu = nn.ReLU()
        conv0,conv1,conv2,conv3 = (
            ConvTransposeNorm1d(self.__inch,32,15),
            ConvTransposeNorm1d(32,32,16),
            ConvTransposeNorm1d(32,16,21),
            ConvTransposeNorm1d(16,8,41)
        )

        self.layers = nn.Sequential(
            conv0,relu,upper,
            conv1,relu,upper,
            conv2,relu,upper,
            conv3,relu,
            nn.Conv1d(8,1,3,padding=1),nn.Tanh(),

        )
    
    def forward(self,x):
        x = self.layers(x)
        return x

    def freeze(self):
        self.eval()
        for i in self.parameters():
            i.requires_grad = False

class WaveAutoEncoder(pl.LightningModule):
    input_size:tuple = WaveEncoder.input_size
    output_size:tuple = input_size

    def __init__(self,lr:float=0.001):
        super().__init__()
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
        self.log('WaveAutoEncoder loss',loss)
        return loss

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)