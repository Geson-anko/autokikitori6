import torch
import torch.nn as nn
import config
import pytorch_lightning as pl
from hparams import waveautoencoder3_default as hparams
from Additional_layers import ConvNorm1d,ConvTransposeNorm1d,DilatedCausalConv1d,nResBlocks1d,CausalConv1d
from torchsummaryX import summary

class Encoder(nn.Module):
    seq_len = config.speak_stft_seq_len
    recog_len = config.recognize_length
    over_len = config.overlap_length

    def __init__(self,hparams:hparams.encoder_hparams):
        super().__init__()
        
        self.model_name = hparams.model_name
        self.input_size:tuple = (1,config.channels,config.speak_length)
        self.output_size = (1,hparams.latent_dim,self.seq_len)
        self.hparams = hparams
        self.latent_dim = hparams.latent_dim

        # <layers>
            # <global encoder>
        self.__global_outch = 1
        pad_layer = nn.ReflectionPad1d(self.over_len)
        self.global_layers= nn.Sequential(
            pad_layer,
        )

            # <local encoder>
        poolconv0 = ConvNorm1d(self.__global_outch, hparams.lch0, kernel_size=43, stride=3)
        res0 = nResBlocks1d( hparams.lch0, hparams.lch0, kernel_size=hparams.lk0,
                        nlayers=hparams.nlayer0, channel_divsor=hparams.divs0)
        poolconv1 = ConvNorm1d( hparams.lch0, hparams.lch1, kernel_size=23, stride=3)
        res1 = nResBlocks1d(hparams.lch1,hparams.lch1,hparams.lk1,hparams.nlayer1,hparams.divs1)
        poolconv2 = ConvNorm1d(hparams.lch1, hparams.lch2, kernel_size=18, stride=3)

        out_conv = nn.Conv1d(hparams.lch2,hparams.latent_dim,15)

        self.local_layers = nn.Sequential(
            poolconv0,nn.ReLU(),
            res0,
            poolconv1,nn.ReLU(),
            res1,
            poolconv2,nn.ReLU(),
            out_conv,nn.Tanh(),
        )

    def forward(self,x:torch.Tensor):
        h = self.global_flow(x)
        h = h.view(-1,self.__global_outch,self.recog_len)
        h = self.local_flow(h)
        h = h.view(-1,self.seq_len,self.latent_dim).transpose(2,1)
        return h

    def global_flow(self,x:torch.Tensor) -> torch.Tensor:
        # x: (-1, 1, 20800)
        # return -> (-1, 66, 1, 640)
        h = self.global_layers(x)
        h = h.unfold(-1,self.recog_len,self.over_len)
        h = h.transpose(2,1).contiguous()
        return h
        
    def local_flow(self,x:torch.Tensor) -> torch.Tensor:
        # x: (-1, 1, 640)
        # return -> (-1, 4, 1)
        y = self.local_layers(x)
        return y
    
    def summary(self):
        dummy = torch.randn(self.input_size)
        summary(self,dummy)

class Decoder(nn.Module):
    seq_len = config.speak_stft_seq_len
    recog_len = config.recognize_length
    over_len = config.overlap_length

    def __init__(self,hparams:hparams.decoder_hparams):
        super().__init__()

        self.model_name = hparams.model_name
        self.input_size = (1,hparams.latent_dim,self.seq_len)
        self.output_size =  (1,config.channels,config.speak_length)
        self.hparams = hparams
        self.latent_dim = hparams.latent_dim

        # <layers>
            # <global decoder 0>
        point_wise = ConvNorm1d(self.latent_dim,hparams.ch0,1)
        res = nResBlocks1d(hparams.ch0, hparams.ch0, hparams.k0, hparams.nlayer0, hparams.divs0)
        self.global_layers0 = nn.Sequential(
            point_wise,nn.ReLU(),res
        )
            # <local decoder>
        upper0 = ConvTransposeNorm1d(hparams.ch0, hparams.ch1, kernel_size=15)
        res0 = nResBlocks1d(hparams.ch1, hparams.ch1, hparams.k1, hparams.nlayer1, hparams.divs1)
        upper1 = ConvTransposeNorm1d(hparams.ch1, hparams.ch2,kernel_size=18, stride=3)
        res1 = nResBlocks1d(hparams.ch2,hparams.ch2, hparams.k2, hparams.nlayer2, hparams.divs2)
        upper2 = ConvTransposeNorm1d(hparams.ch2,hparams.ch3,23,3)
        res2 = nResBlocks1d(hparams.ch3, hparams.ch3, hparams.k3, hparams.nlayer3, hparams.divs3)
        upper3 = ConvTransposeNorm1d(hparams.ch3, hparams.ch4, kernel_size=43, stride=3)
        res3 = nResBlocks1d(hparams.ch4, hparams.ch4, hparams.k4, hparams.nlayer4, hparams. divs4)

        self.local_layers = nn.Sequential(
            upper0,nn.ReLU(),
            res0,
            upper1,nn.ReLU(),
            res1,
            upper2,nn.ReLU(),
            res2,
            upper3,nn.ReLU(),
            res3,
        )
            # <global decoder1>
        self.__catshape0 = (-1,hparams.ch4,config.speak_length+self.recog_len)
        DCC0 = DilatedCausalConv1d(hparams.ch4*2, hparams.ch5, hparams.k5, hparams.nlayer5,divs=hparams.divs5)
        DCC1 = DilatedCausalConv1d(hparams.ch5, hparams.ch5, hparams.k5, hparams.nlayer5,divs=hparams.divs5)
        out_conv = CausalConv1d(hparams.ch5,1,hparams.k5)
        self.global_layers1 = nn.Sequential(
            DCC0,DCC1,out_conv,nn.Tanh()
        )

    def forward(self,x:torch.Tensor):
        h = self.global_flow0(x)
        h = h.transpose(2,1).reshape(-1,self.hparams.ch0).unsqueeze(-1)
        h = self.local_flow1(h)
        h = h.view(-1,self.seq_len,self.hparams.ch4,self.recog_len)
        y = self.global_flow1(h)
        return y
    
    def global_flow0(self,x:torch.Tensor) -> torch.Tensor:
        # (-1, 4, 66) -> (-1, 64, 66)
        return self.global_layers0(x)

    def local_flow1(self,x:torch.Tensor) -> torch.Tensor:
        # x: (-1, 64, 1)
        # return (-1, 32, 640)
        return self.local_layers(x)
    
    def global_flow1(self,x:torch.Tensor) -> torch.Tensor:
        # x: (-1, 66, 32, 640) -> (-1, 1, 20800)
        h = self.slide_concat(x)
        h = self.global_layers1(h)
        return h
        
    def slide_concat(self,x:torch.Tensor) -> torch.Tensor:
        """
        x: shape (B,seq_len,C,L) -> (-1,66,32,640)
        return -> (-1,64,20800)
        """
        x = x.transpose(2,1)
        x0 = x[:,:,:,:self.over_len]
        x1 = x[:,:,:,self.over_len:]
        x0_pad = x1[:,:,-1:]
        x1_pad = x0[:,:,:1]
        x0 = torch.cat([x0,x0_pad],dim=2)
        x1 = torch.cat([x1_pad,x1],dim=2)
        x0 = x0.view(self.__catshape0)
        x1 = x1.view(self.__catshape0)

        x = torch.cat([x0,x1],dim=1)
        x = x[:,:,self.over_len:-self.over_len]
        return x

    def summary(self):
        dummy = torch.randn(self.input_size)
        summary(self,dummy)

class AutoEncoder(pl.LightningModule):

    def __init__(self,hparams:hparams):
        super().__init__()

        self.model_name = hparams.model_name
        self.lr = hparams.lr 
        self.latent_dim = hparams.latent_dim
        self.my_hparams = hparams

        # set criterion
        self.criterion = nn.MSELoss()

        # <layers>
        self.encoder = Encoder(hparams.encoder_hparams)
        self.decoder = Decoder(hparams.decoder_hparams)

        self.input_size = self.encoder.input_size
        self.output_size = self.input_size

    def forward(self,x:torch.Tensor):
        h = self.encoder(x)
        h = self.decoder(h)
        return h

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)

    def training_step(self,batch,index):
        data, = batch
        self.data = data
        out = self(data)
        loss = self.criterion(out,data)
        self.log('loss',loss)
        return loss

    @torch.no_grad()
    def on_epoch_end(self) -> None:
        if (self.current_epoch+1) % self.my_hparams.view_interval == 0:
            data = self.data[:self.my_hparams.max_view_num].float()
            out = self(data)
            audio = torch.cat([data,out],dim=1).view(-1)
            self.logger.experiment.add_audio('WaveAutoEncoder regenerated',audio,self.current_epoch,sample_rate=config.frame_rate)


    def summary(self,tensorboard):
        dummy = torch.randn(self.input_size)
        summary(self,dummy)
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            writer.add_graph(self,dummy)
            writer.close()
if __name__ == '__main__':
    model = AutoEncoder(hparams)
    model.summary(True)
