from DatasetLib import CreateDataset,DataHolder
import torch
import numpy as np
from DatasetLib import CreateDataset,DataHolder,sound_load
from torchaudio.transforms import Spectrogram
import config
import glob

class ToData(CreateDataset):
    filepath = 'data/VoiceBand_data.h5'
    target_key = 'target'

    def __init__(self):
        self.specter = Spectrogram(config.recognize_length,hop_length=config.overlap_length)
    
    def load(self):
        files = glob.glob('data/checked/*.wav')
        return files

    def process(self,input_Data):
        sound = sound_load(input_Data)
        sound = torch.from_numpy(sound).view(-1)
        data = self.mod_pad(sound,config.speak_length)
        data = data.reshape(-1,config.speak_length)
        data = self.specter(data)
        data = data.log1p()
        data = data.half().numpy()
        data = DataHolder(self.target_key,data)
        return data

if __name__ == '__main__':
    to_data = ToData()
    to_data.run()

