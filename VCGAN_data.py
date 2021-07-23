from DatasetLib import CreateDataset,UnfoldFFT,sound_load,DataHolder
import config
from torchaudio.transforms import MelScale,Spectrogram
import torch
import numpy as np
import glob

filepath = 'data/VCGAN_data.h5'
source_key = 'source'
target_key = 'target'

class MakeSource(CreateDataset):
    filepath = filepath

    def __init__(self) ->None:
        self.mel_scaler = MelScale(config.mel_channels,config.frame_rate,n_stft=config.fft_channels)

    def load(self):
        files = glob.glob('data/checked/*.wav')
        return files

    def process(self,input_data):
        sound = sound_load(input_data).astype('float32')
        sound = self.mod_pad(sound,config.speak_length)
        sound = sound.reshape(-1,config.speak_length)
        sound = torch.from_numpy(sound)
        uf = UnfoldFFT(sound)
        mels = self.mel_scaler(uf).log1p().half().numpy()
        return DataHolder(source_key,mels)
    
class MakeTarget(CreateDataset):
    filepath = MakeSource.filepath

    def __init__(self) -> None:
        self.spectrogramer = Spectrogram(config.recognize_length,hop_length=config.overlap_length)
    
    def load(self):
        files = glob.glob('data/checked_kiritan/*.wav')
        return files
    
    def process(self,input_data):
        sound = sound_load(input_data).astype('float32')
        sound = self.mod_pad(sound,config.speak_length)
        sound = sound.reshape(-1,config.speak_length)
        sound = torch.from_numpy(sound)
        spect = self.spectrogramer(sound).log1p().half().numpy()
        return DataHolder(target_key,spect)
if __name__ == '__main__':
    to_data = MakeSource()
    to_data.run()



        