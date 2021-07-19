import numpy as np
import glob
from DatasetLib import CreateDataset,DataHolder
from pydub import AudioSegment
import config

class ToData(CreateDataset):
    filepath = 'data/WaveAutoEncoder2_data.h5'
    data_name:str='data'
    def __init__(self) -> None:
        super().__init__()

    def load(self):
        files = glob.glob('data/checked_kiritan/*.wav')
        return files

    def process(self,input_data):
        sound = self.sound_load(input_data).astype('float16')
        sound = self.mod_pad(sound,config.speak_length,dim=0).reshape(-1,1,config.speak_length)
        return DataHolder(self.data_name,sound)

    def sound_load(self,file_name):
        sound = AudioSegment.from_file(file_name)
        if sound.frame_rate != config.frame_rate:
            sound = sound.set_frame_rate(config.frame_rate)
        if sound.channels != config.channels:
            sound = sound.set_channels(config.channels)
        if sound.sample_width != config.sample_width:
            sound = sound.set_sample_width(config.sample_width)

        sound = np.array(sound.get_array_of_samples()).reshape(-1)
        sound = sound/config.sample_range
        return sound

if __name__ == '__main__':
    to_data = ToData()
    to_data.run()