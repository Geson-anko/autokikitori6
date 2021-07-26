import torch
import torch.nn as nn
from torchaudio.transforms import GriffinLim
import config
import pytorch_lightning as pl

from hparams import Boin_default as hparams
from torchsummaryX import summary


class Encoder(nn.Module):
    __inch = config.fft_channels
    input_size = (1,__inch)
    output_size = (1,1)
    
    def __init__(self,hparams:hparams.encoder_hparams):
        super().__init__()
        self.model_name = hparams.model_name
        self.hparams = hparams
        
        self.input_fc = nn.Linear(self.__inch,hparams.nhid)
        
        layers = [nn.Linear(hparams.nhid,hparams.nhid) for _ in range(hparams.nlayers)]
        self.layers = nn.ModuleList(layers)

        self.output_fc = nn.Linear(hparams.nhid,1)

    def forward(self,x:torch.Tensor):
        h = torch.relu(self.input_fc(x))
        for l in self.layers:
            h = torch.relu(l(h))
        y = torch.tanh(self.output_fc(h))
        return y

    def summary(self):
        dummy = torch.randn(self.input_size)
        summary(self,dummy)

class Decoder(nn.Module):
    __outch = config.fft_channels
    input_size = Encoder.output_size
    output_size = Encoder.input_size

    def __init__(self,hparams:hparams.decoder_hparams):
        super().__init__()
        self.model_name = hparams.model_name
        self.hparams= hparams
        
        self.input_fc = nn.Linear(1,hparams.nhid)

        layers = [nn.Linear(hparams.nhid,hparams.nhid) for _ in range(hparams.nlayers)]
        self.layers = nn.ModuleList(layers)

        self.output_fc = nn.Linear(hparams.nhid,self.__outch)

    def forward(self,x:torch.Tensor):
        h = torch.relu(self.input_fc(x))
        for l in self.layers:
            h = torch.relu(l(h))
        y = torch.relu(self.output_fc(h))
        return y

    def summary(self):
        dummy = torch.randn(self.input_size)
        summary(self,dummy)

class AutoEncoder(pl.LightningModule):
    input_size = Encoder.input_size
    output_size = input_size

    def __init__(self,hparams:hparams):
        super().__init__()
        self.model_name = hparams.model_name
        self.my_hparams = hparams
        self.lr = hparams.lr
        # set criterion
        self.criterion = nn.MSELoss()
        # set griffin lim
        self.griffin_lim = GriffinLim(config.recognize_length,hop_length=config.overlap_length)

        # layers
        self.encoder = Encoder(hparams.encoder_hparams)
        self.decoder = Decoder(hparams.decoder_hparams)
        
    def forward(self,x:torch.Tensor):
        e = self.encoder(x)
        d = self.decoder(e)
        return d

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(),lr=self.lr)
        return optim

    batch_size = None
    def training_step(self,batch,idx):
        data, = batch
        out = self(data)
        loss = self.criterion(out,data)
        self.log('loss',loss)

        # for audio logging.
        if self.batch_size is None:
            self.batch_size = data.size(0)
        return loss

    # audio log
    @torch.no_grad()
    def on_epoch_end(self) -> None:
        if (1+self.current_epoch)%self.my_hparams.view_interval == 0:

            encoded_data = torch.linspace(-1,1,self.my_hparams.max_view_num,device=self.device,dtype=self.dtype).unsqueeze(1)
            decoded = self.decoder(encoded_data)
            audio = self.data_to_audio(decoded).cpu()
            self.logger.experiment.add_audio('generated boin',audio,self.current_epoch,sample_rate=config.frame_rate)


    def data_to_audio(self,data:torch.Tensor) -> torch.Tensor:
        """
        data: shape is (-1, fft_channels)
        """
        data = data.T.float().expm1()
        audio = self.griffin_lim(data).view(-1)
        return audio

    def summary(self):
        dummy = torch.randn(self.input_size)
        summary(self,dummy)

if __name__ == '__main__':
    model =AutoEncoder(hparams)
    model.summary()
