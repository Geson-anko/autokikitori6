import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram,GriffinLim
import config
import pytorch_lightning as pl
from Additional_layers import CausalConv1d,DilatedCausalConv1d,nResBlocks1d
from torchsummaryX import summary

from hparams import dict_to_attr,json_to_dict
default_hparams_file = 'hparams/voiceband_default.json'

from Boin_model import Decoder

class VoiceBand(pl.LightningModule):
    __inch = config.fft_channels
    input_size:tuple = (1,config.fft_channels,config.speak_stft_seq_len)
    output_size:tuple = input_size

    def __init__(self,hparams:dict = json_to_dict(default_hparams_file)):
        super().__init__()
        # set hparams
        hp_cls = dict_to_attr(hparams)
        self.my_hparams = hp_cls
        self.my_hparams_dict = hparams
        self.model_name  =hp_cls.model_name

        # set criterion
        self.criterion = nn.MSELoss()

        # set griffin lim and spectrogram
        self.griffin_lim = GriffinLim(config.recognize_length,hop_length=config.overlap_length)
        self.spectrogram = Spectrogram(config.recognize_length,hop_length=config.overlap_length)

        # set decoders
        self.boin_decoder = Decoder(hp_cls.BoinDecoder_hparams)
        self.boin_decoder.load_state_dict(torch.load(hp_cls.BoinDecoder_parameter_file))
        self.freeze(self.boin_decoder)

        self.siin_decoder = Decoder(hp_cls.SiinDecoder_hparams)
        self.siin_decoder.load_state_dict(torch.load(hp_cls.SiinDecoder_parameter_file))
        self.freeze(self.siin_decoder)

        # layers
        init_conv = CausalConv1d(self.__inch,hp_cls.ch0,hp_cls.k0)
        DCC = DilatedCausalConv1d(hp_cls.ch0,hp_cls.ch1,hp_cls.k1,num_layers=hp_cls.layers1,divs=hp_cls.divs1)
        resblock = nResBlocks1d(hp_cls.ch1,hp_cls.ch2,hp_cls.k2,hp_cls.layers2,hp_cls.divs2)
        self.layers = nn.Sequential(
            init_conv,nn.ReLU(),
            DCC,resblock,
        )
            # to gain, boin, siin, brend
        self.to_gain = nn.Conv1d(hp_cls.ch2,1,1)
        self.to_boin = nn.Conv1d(hp_cls.ch2,1,1)
        self.to_siin = nn.Conv1d(hp_cls.ch2,1,1)
        self.to_blend = nn.Conv1d(hp_cls.ch2,2,1)

    def forward(self,target:torch.Tensor):
        # target : (-1, 321,66)
        #audio = self.target_to_audio(target)
        #spect = self.audio_to_spectrogram(audio)
        spect = self._boin_only_flow(target)
        return spect

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(),lr=self.my_hparams.lr)
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
            #generated_audio = self.target_to_audio(target).unsqueeze(1)
            generated_audio = self.spectrogram_to_audio(self._boin_only_flow(target)).unsqueeze(1)
            target_audio = self.spectrogram_to_audio(target).unsqueeze(1)
            audio = torch.cat([target_audio,generated_audio],dim=1).view(-1)
            self.logger.experiment.add_audio('generated boin',audio,self.current_epoch,sample_rate=config.frame_rate)
            


    #def on_fit_start(self) -> None:
    #    return self.logger.experiment.add_hparams(self.my_hparams_dict,dict())

    def target_to_audio(self,target:torch.Tensor) -> torch.Tensor:
        """
        target (-1, 321, 66) -> audio (-1, 20800)
        """
        h = self.layers(target) # (-1, 512, 66)
        blend = self.to_blend(h).transpose(2,1).softmax(dim=-1) # (-1, 66, 2)
        gain = self.to_gain(h).sigmoid().transpose(2,1).mean(dim=1) # (-1, 1)
        
        boin = self.to_boin(h).tanh().transpose(2,1) # (-1, 66, 1)
        boin = self.boin_decoder(boin) # (-1, 66, 321)
        siin = self.to_siin(h).tanh().transpose(2,1) # (-1, 66, 1)
        siin = self.siin_decoder(siin)

        voice_spect = boin * blend[:,:,0].unsqueeze(-1) + siin * blend[:,:,1].unsqueeze(-1) # (-1, 66, 321)
        audio = self.spectrogram_to_audio(voice_spect.transpose(2,1))
        audio = audio * gain
        return audio

    def _boin_only_flow(self,target:torch.Tensor) -> torch.Tensor:
        h = self.layers(target) # (-1, 512, 66)
        boin = self.to_boin(h).tanh().transpose(2,1) # (-1, 66, 1)
        boin = self.boin_decoder(boin).transpose(2,1) # (-1, 321, 66)
        return boin

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
    model = VoiceBand()
    model.summary(False)
    



    