import numpy as np
from pydub import AudioSegment
import torch
import config
from glob import glob
class MakeTrue:
    
    save_file:str = 'data/HumanChecker.h5'
    key_name:str = 'voice'
    def __init__(self) -> None:
        
        pass

    def load(self) -> list:
        files = glob("data/voice_only/**/*.wav")
        files += glob("data/voice_only/*.wav")
        

        return list(set(files))

    def preprocess(self,indata) -> np.ndarray:
        """
        Your data preprocess.
        return is numpy array
        """
        sound = AudioSegment.from_file(indata)
        if sound.frame_rate != config.frame_rate:
            sound = sound.set_frame_rate(config.frame_rate)
        if sound.channels !=config.channels:
            sound = sound.set_channels(config.channels)
        if sound.sample_width != config.sample_width:
            sound = sound.set_sample_width(config.sample_width)

        soundarray = np.array(sound.get_array_of_samples())/config.sample_range

        soundtensor = []
        for idx in range(0,soundarray.shape[0],config.CHUNK):
            _st = soundarray[idx:idx+config.sample_length]
            padlen = config.sample_length - _st.shape[0]
            pad = np.zeros(padlen,dtype=_st.dtype)
            _st = np.concatenate([_st,pad])
            soundtensor.append(_st)
        soundtensor = np.stack(soundtensor)
        soundtensor = torch.from_numpy(soundtensor).unfold(1,config.recognize_length,config.overlap_length)
        soundtensor = torch.fft.rfft(soundtensor,dim=-1).abs() # (-1,seq_len,channels)
        soundtensor = torch.log1p(soundtensor).permute(0,2,1)
        return soundtensor.detach().numpy().astype('float16')

    def save(self,data:np.ndarray) -> None:
        with h5py.File(self.save_file,'a') as f:
            f.create_dataset(name=self.key_name,data=data)
        print('saved')

class MakeFalse(MakeTrue):

    key_name:str = 'noize'
    def load(self) -> list:
        files = glob("data/others/*.m4a")
        
        return files


if __name__ == '__main__':
    import h5py
    from concurrent.futures import ProcessPoolExecutor

    todata = MakeTrue()
    func = todata.preprocess
    workers = 8
    database = todata.load()

    print('processing')
    with ProcessPoolExecutor(workers) as p:
        result = p.map(func,database[:workers])
    data = np.concatenate(list(result))
    print('data size is',data.shape)
    todata.save(data)


    #test
    #d = func(database[0])
    #print(d.shape)
    #print(d.dtype)

