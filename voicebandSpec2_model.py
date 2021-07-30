import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram,GriffinLim
import config
import pytorch_lightning as pl
from Additional_layers import CausalConv1d,DilatedCausalConv1d,nResBlocks1d
from torchsummaryX import summary

from hparams import voicebandSpec2_default as hparams

from Boin2_model import Decoder

class VoiceBand(pl.LightningModule):
    __inch = config.fft_channels
    input_size:tuple = (1,config.fft_channels,config.speak_stft_seq_len)
    output_size:tuple = input_size

    def __init__(self,hparams:hparams):
        super().__init__()
        # set hparams
        self.my_hparams = hparams
        self.model_name = hparams.model_name
        self.lr = hparams.lr
        
        # set criterion
        self.criterion = nn.MSELoss()

        # set griffin lim and spectrogram
        self.griffin_lim = GriffinLim(config.recognize_length,hop_length=config.overlap_length)
        self.spectrogram = Spectrogram(config.recognize_length,hop_length=config.overlap_length)

        # set decoder
        self.decoder = Decoder(hparams.decoder_hparams)
        if hparams.decoder_parameter_file is not None:
            self.decoder.load_state_dict(torch.load(hparams.decoder_parameter_file))
        self.freeze(self.decoder)

        # layers
        init_conv = CausalConv1d(self.__inch,hparams.ch0,hparams.k0)
        DCC = DilatedCausalConv1d(hparams.ch0,hparams.ch1,hparams.k1,num_layers=hparams.layers1,divs=hparams.divs1)
        resblock = nResBlocks1d(hparams.ch1,hparams.ch2,hparams.k2,hparams.layers2,hparams.divs2)
        self.layers = nn.Sequential(
            init_conv,nn.ReLU(),
            DCC,resblock,
        )
        self.to_gain = nn.Conv1d(hparams.ch2,1,1)
        self.to_decoder = nn.Conv1d(hparams.ch2,hparams.latent_dim,1)


    def forward(self,target:torch.Tensor):
        # target : (-1, 321,66)
        output = self.layers(target) # (-1, 512, 66)
        encoded = self.to_decoder(output).tanh().transpose(2,1) # (-1, 66, 1)
        voice = self.decoder(encoded).transpose(2,1) # (-1, 321, 66)

        gain = self.to_gain(output).sigmoid() # (-1, 1, 66)
        spect = (voice.expm1() * gain).log1p()
        return spect

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(),lr=self.lr)
        return optim

    def training_step(self,batch,idx):
        target, = batch
        self.target = target
        out = self(target)
        loss = self.criterion(out,target)
        self.log('loss',loss)
        return loss

    # audio log
    @torch.no_grad()
    def on_epoch_end(self) -> None:
        if (1+self.current_epoch) % self.my_hparams.view_interval==0:
            target = self.target[:self.my_hparams.max_view_len].to(self.device).to(self.dtype)
            generated_audio = self.ToWave(target).unsqueeze(1)
            #generated_audio = self.spectrogram_to_audio(self._boin_only_flow(target)).unsqueeze(1)
            target_audio = self.spectrogram_to_audio(target).unsqueeze(1)
            audio = torch.cat([target_audio,generated_audio],dim=1).view(-1)
            self.logger.experiment.add_audio('generated boin',audio,self.current_epoch,sample_rate=config.frame_rate)
            

    def ToWave(self,target:torch.Tensor) -> torch.Tensor:
        """
        target (-1, 321, 66) -> audio (-1, 20800)
        """
        spect = self(target)
        audio = self.spectrogram_to_audio(spect)
        return audio


    def spectrogram_to_audio(self,spect:torch.Tensor) -> torch.Tensor:
        """
        spect: (-1, 321, 66)
        return -> (-1, 20800)
        """
        spect = torch.clamp(spect,max=8.0)
        audio = self.griffin_lim(spect.float().expm1()).type_as(spect)
        return audio
    def audio_to_spectrogram(self,audio:torch.Tensor) -> torch.Tensor:
        """
        audio (-1, 20800)
        return -> (-1, 321, 66)
        """
        spect = self.spectrogram(audio).log1p()
        return spect

    def summary(self,tensorboard=False):
        dummy = torch.randn(self.input_size)
        summary(self,dummy)
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            writer.add_graph(self,dummy)
            writer.close()
    
    @staticmethod
    def freeze(model:nn.Module) -> nn.Module:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

if __name__ == '__main__':
    model = VoiceBand(hparams)
    model.summary(False)
    



    