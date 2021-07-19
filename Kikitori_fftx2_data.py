import torch
from torchaudio.transforms import MelScale
import numpy as np
import h5py
from pydub import AudioSegment
import config
import glob

class ToData:
    
    file_name:str = 'data/encoded_fftx2.h5'
    key_name:str = 'data'
    device = 'cpu'

    def __init__(self) -> None:
        self.mel_scaler = MelScale(
            config.mel_channels,
            config.frame_rate,
            n_stft=config.fft_channels
        )

    def load(self) -> list:
        files = glob.glob('data/voice_only/*.wav')

        return  list(set(files))

    def save(self,data:np.ndarray,overwrite:bool=True) -> None:

        print(data.shape)
        idxes = np.random.permutation(len(data))
        data = data[idxes]
        with h5py.File(self.file_name,'a') as f:
            if self.key_name in f and overwrite:
                del f[self.key_name]
                f.create_dataset(name=self.key_name,data=data)
            else:
                f.create_dataset(name=self.key_name,data=data)


    def run(self,indata:str) -> np.ndarray:
        
        soundarray = self.load_sound(indata)
        sound = self.preprocess(soundarray)
        sound = sound.detach().cpu().half().numpy()
        return sound

    def load_sound(self,sound_file:str) -> np.ndarray:
        sound = AudioSegment.from_file(sound_file)
        if sound.frame_rate != config.frame_rate:
            sound = sound.set_frame_rate(config.frame_rate)
        if sound.channels !=config.channels:
            sound = sound.set_channels(config.channels)
        if sound.sample_width != config.sample_width:
            sound = sound.set_sample_width(config.sample_width)

        soundarray = np.array(sound.get_array_of_samples())/config.sample_range

        return soundarray

    def preprocess(self,sound:np.ndarray) -> torch.Tensor:
        mel_scaler = self.mel_scaler.to(self.device)
        sound = torch.from_numpy(sound).to(self.device)
        padlen = (config.overlap_length - (len(sound)%config.overlap_length))
        pad = torch.zeros(padlen,dtype=sound.dtype)
        sound = torch.cat([sound,pad]).unfold(0,config.recognize_length,config.overlap_length)
        sound = torch.fft.rfft(sound,dim=-1).abs().T.float()
        sound = mel_scaler(sound).T
        sound = torch.log1p(sound)
        sound = torch.fft.rfft(sound,dim=-1).abs().float()
        sound = torch.log1p(sound)
        return sound

if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    todata = ToData()
    func = todata.run
    database = todata.load()
    print('process start!')
    with ProcessPoolExecutor(8) as p:
        result = p.map(func,database)
    result = np.concatenate(list(result))
    todata.save(result)
    # test
    #out = todata.run(database[0])
    #print(out.shape)