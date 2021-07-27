import torch
import numpy as np
from DatasetLib import CreateDataset,DataHolder,sound_load
from torchaudio.transforms import Spectrogram
import config
import glob

class ToData(CreateDataset):
    filepath = 'data/BoinAutoEncoder_siinp.h5'
    data_key = 'data'

    def __init__(self):
        self.specter = Spectrogram(config.recognize_length,hop_length=config.overlap_length)

    def load(self):
        files = glob.glob('data/siin_only/*.wav')
        return files

    def process(self,input_data):
        data = sound_load(input_data)
        data = torch.from_numpy(data).view(-1)
        data = self.specter(data)
        data = data.T.log1p()
        data = data.half().numpy()
        data = DataHolder(self.data_key,data)
        return data

if __name__ =='__main__':
    to_data = ToData()
    to_data.run()        
