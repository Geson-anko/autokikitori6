import torch
import torch.nn as nn
import os
import random
import config
import pytorch_lightning as pl
from Additional_layers import DilatedCausalConv1d
from torchaudio.transforms import MelScale

from WaveAutoEncoder_model import WaveDecoder

class OshaberiController(nn.Module):
    __inch = config.mel_channels
    __inlen = config.speak_seq_len
    input_size:tuple = (1,__inch,__inlen)
    output_size:tuple = (1,__inlen,*WaveDecoder.input_size[1:]) 
    
    def __init__(self):
        super().__init__()
        
        # layers
        self.DDC = DilatedCausalConv1d(self.__inch,64,3,4,divs=2)
        self.conv = nn.Conv1d(64,32,3,padding=1)

    def forward(self,x:torch.Tensor):
        x = self.DDC(x)
        x = torch.tanh(self.conv(x)).permute(0,2,1).unsqueeze(-1).contiguous()
        return x

class OshaberiAutoEncoder(pl.LightningModule):
    input_size:tuple = OshaberiController.input_size
    output_size:tuple = input_size
    __overlen = config.overlap_length

    __decoder_insize:tuple = (-1,*WaveDecoder.input_size[1:])
    __avg_trans_insize:tuple = (-1,config.speak_seq_len,config.recognize_length)

    def __init__(self,decoder_file:str,lr:float = 0.001):
        super().__init__()
        self.reset_seed()
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.mel_scaler = MelScale(config.mel_channels,config.frame_rate,n_stft=config.fft_channels)
        # layers
        self.oshaberi_controller = OshaberiController()
        self.wave_decoder = WaveDecoder()
        self.wave_decoder.load_state_dict(torch.load(decoder_file))
        self.wave_decoder.freeze()

    def forward(self,x:torch.Tensor):
        x = self.oshaberi_controller(x).view(self.__decoder_insize)
        x = self.wave_decoder(x).view(self.__avg_trans_insize)
        x = self.AvgTransform(x)
        x = self.ToMelSpectrogram(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),self.lr)
    
    def training_step(self,batch,index):
        data, = batch
        out = self(data)
        loss = self.criterion(out,data)
        self.log('OshaberiAutoEncoder MSELoss',loss)
        return loss

    @torch.no_grad()
    def toWave(self,x:torch.Tensor) -> torch.Tensor:
        x = self.oshaberi_controller(x).view(self.__decoder_insize)
        x = self.wave_decoder(x).view(self.__avg_trans_insize)
        x = self.AvgTransform(x)
        return x

    def AvgTransform(self,x:torch.Tensor) -> torch.Tensor:
        """
        x: (B, seqlen, recoglen) -> (-1,64,640)
        """
        x0 = x[:,:,:self.__overlen]
        x1 = x[:,:,self.__overlen:]
        x0_pad = x1[:,-1:]
        x1_pad = x0[:,:1]
        x0 = torch.cat([x0,x0_pad],dim=1)
        x1 = torch.cat([x1_pad,x1],dim=1)
        x = (x0+x1)/2
        x = x.view(-1,1,config.speak_length)
        return x
    
    def ToMelSpectrogram(self,x:torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, speak_length) -> (-1,1,20800)
        """
        x = x.view(-1,config.speak_length)
        dtype = x.dtype
        x = x.unfold(1,config.recognize_length,config.overlap_length).float()
        x = torch.fft.rfft(x).abs().float().permute(0,2,1)
        x = self.mel_scaler(x)#.permute(0,2,1).contiguous()
        x = torch.log1p(x).type(dtype)
        return x
        
    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    from torchsummaryX import summary
    model = OshaberiAutoEncoder('')
    dummy = torch.randn(model.input_size)
    summary(model,dummy)
    print(model(dummy).shape)



