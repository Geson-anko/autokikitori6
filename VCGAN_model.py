"""
VCGAN is "Voice Conversion GAN".
This GAN convert any human's voice mel spectrogram to specific human's voice spectrogram.
"""

import torch
import torch.nn as nn
import numpy as np
from Conformer import Conformer
import config
from hparams import VCGAN_default as hparams
from Additional_layers import CausalConv1d,nResBlocks1d,DilatedCausalConv1d
import pytorch_lightning as pl
import copy
from torchaudio.transforms import MelScale,Spectrogram,GriffinLim
import random
import os
from DatasetLib import UnfoldFFT
from collections import OrderedDict
from torchmetrics import Accuracy

class Generator(nn.Module):
    input_size:tuple = (1,config.mel_channels,config.speak_seq_len)
    output_size:tuple = (1,config.fft_channels,config.speak_stft_seq_len)
    fft_channels = config.fft_channels
    mel_channels = config.mel_channels
    __inseq = config.speak_seq_len
    __outseq= config.speak_stft_seq_len

    def __init__(self,hparams:hparams.generator_hparams = hparams.generator_hparams):
        super().__init__()
        self.hparams = hparams
        self.model_name = hparams.model_name
        self.griffin_lim = GriffinLim(config.recognize_length,hop_length=config.overlap_length)
        # layers
        init_k = self.__outseq-self.__inseq+1
        upper = nn.ConvTranspose1d(self.mel_channels,self.mel_channels,kernel_size=init_k)
        layer0 = DilatedCausalConv1d(self.mel_channels,hparams.ch0,hparams.kernel0,num_layers=hparams.num_layer0)
        layer1 = nResBlocks1d(hparams.ch0,hparams.ch1,hparams.kernel1,hparams.num_reslayers,hparams.divisor)
        layer2 = CausalConv1d(hparams.ch1,self.fft_channels,kernel_size=hparams.kernel2)

        self.layers = nn.Sequential(
            upper,
            layer0,
            layer1,
            layer2,nn.ReLU(),
        )

    def forward(self,x:torch.Tensor):
        x =self.layers(x)
        return x
    
    @torch.no_grad()
    def ToWave(self,x:torch.Tensor) -> torch.Tensor:
        """
        x: (-1,321,66), processed by log1p.
        """
        x = self(x).float()
        x = torch.expm1(x)
        x = self.griffin_lim(x)
        return x
        

class Discriminator(nn.Module):
    input_size:tuple = (1,config.fft_channels,config.speak_stft_seq_len)
    output_size:tuple = (1,1)

    ninp = config.fft_channels
    seq_len = config.speak_stft_seq_len

    def __init__(self,hparams:hparams.disciminator_hparams=hparams.disciminator_hparams):
        super().__init__()
        self.model_name = hparams.model_name
        self.hparams = hparams

        _dm = (self.ninp//hparams.nhead)*hparams.nhead
        self.init_dense = nn.Linear(self.ninp,_dm)
        conformer = Conformer(
            d_model= _dm,
            n_head=hparams.nhead,
            ff1_hsize=hparams.nhid,
            ff2_hsize=hparams.nhid,
            ff1_dropout=hparams.dropout,
            conv_dropout=hparams.dropout,
            ff2_dropout=hparams.dropout,
            mha_dropout=hparams.dropout,
            kernel_size=hparams.kernel_size,
        )
        self.conformers = nn.ModuleList([copy.deepcopy(conformer) for _ in range(hparams.nlayers)])
        self.fc = nn.Linear(self.seq_len*_dm,1)

    def forward(self,x:torch.Tensor):
        x = x.transpose(2,1)
        x = torch.relu(self.init_dense(x))
        for l in self.conformers:
            x = l(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

class VCGAN(pl.LightningModule):
    source_size = Generator.input_size
    target_size = Discriminator.input_size

    def __init__(self,hparams:hparams = hparams):
        super().__init__()
        self.__hparams = hparams
        self.model_name= hparams.model_name
        self.reset_seed()
        
        # define adversarial loss
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        # define L1loss
        self.l1_loss = nn.L1Loss()
        # define mel_scaler
        self.mel_scaler = MelScale(config.mel_channels,config.frame_rate,n_stft=config.fft_channels)
        # define accuracy
        self.accuracy = Accuracy()

        # layers
        self.generator = Generator(hparams.generator_hparams)
        self.discriminator = Discriminator(hparams.disciminator_hparams)

    def configure_optimizers(self):
        lr = self.__hparams.lr
        optimG = torch.optim.Adam(self.generator.parameters(),lr=lr)
        optimD = torch.optim.Adam(self.discriminator.parameters(),lr=lr)
        return [optimG,optimD],[]

    def true_loss(self,target:torch.Tensor) -> torch.Tensor:
        """
        true loss is the indicator is this GAN.
        We want to get fewer, but do not use `backward()` and `update()`.
        this loss is just an indicator.
        """
        audio = self.generator.griffin_lim(target)
        source = UnfoldFFT(audio)
        source = self.mel_scaler(source).log1p()
        out = self.generator(source.type_as(target))
        loss = self.l1_loss(out,target)
        return loss

    def training_step(self,batch,batch_idx,optimizer_idx):
        source,target = batch

        # generator updates
        if optimizer_idx == 0:
            generated = self.generator(source)
            # save generated data
            self.generated = generated.detach()
            
            valid = torch.ones(source.size(0),1)
            valid = valid.type_as(generated)
            out_dis = self.discriminator(generated)
            g_loss = self.adversarial_loss(out_dis,valid)
            true_loss = self.true_loss(target)
            g_acc = self.accuracy(out_dis>0,valid.bool())
            self.log('true_loss',true_loss)
            self.log('g_loss',g_loss)
            self.log('g_acc',g_acc)
            # logging
            tqdm_dict = {'g_loss':g_loss}
            output = OrderedDict({'loss': g_loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})
            return output

        # discriminator updates
        elif optimizer_idx ==1:
            # real
            valid = torch.ones(target.size(0),1)
            valid = valid.type_as(target)
            dis_out = self.discriminator(target)
            real_loss = self.adversarial_loss(dis_out,valid)
            real_acc = self.accuracy(dis_out>0,valid.bool())

            # fake
            fake = torch.zeros(source.size(0),1)
            fake = fake.type_as(source)
            dis_out = self.discriminator(self.generated.detach())
            fake_loss = self.adversarial_loss(dis_out,fake)
            fake_acc = self.accuracy(dis_out>0,fake.bool())
            d_loss = (real_loss+fake_loss)/2
            d_acc = (fake_acc+real_acc)/2
            self.log('d_acc',d_acc)
            self.log('d_loss',d_loss)
            tqdm_dict = {'d_loss':d_loss}
            output = OrderedDict({'loss': d_loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})
            return output
        
    # saving generated audios and spectrum images
    now_epochs=0
    saving_rate = 10
    max_view_len = 10
    def on_epoch_end(self) -> None:
        self.now_epochs+=1
        if self.now_epochs % self.saving_rate == 0:
            audio = self.generator.griffin_lim(self.generated[:self.max_view_len].detach().float().expm1()).reshape(-1)
            self.logger.experiment.add_audio('generated audio',audio,self.current_epoch,sample_rate=config.frame_rate)

            # spectrum images
            spects = self.generated[:self.max_view_len]
            s = torch.cat([i for i in spects],dim=-1)
            self.logger.experiment.add_image('generated spectrums',s,self.current_epoch,dataformats='HW')

        return 

    def forward(self,source:torch.Tensor):
        """
        This is data flow. Do not use __call__
        """
        generated = self.generator(source)
        out = self.discriminator(generated)
        return out 

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    from torchsummaryX import summary
    from torch.utils.tensorboard import SummaryWriter
    model = VCGAN()
    dummy = torch.randn(model.target_size).abs()
    print(model.true_loss(dummy))
    #summary(model,dummy)
    #print(model.model_name,model(dummy).shape)

    #writer = SummaryWriter(comment=model.model_name)
    #writer.add_graph(model,dummy)
    #writer.close()