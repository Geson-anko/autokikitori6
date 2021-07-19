from torchaudio.transforms import MelScale
import h5py
import glob
from multiprocessing import Value
import config
from pydub import AudioSegment
import torch
import numpy as np
from typing import Tuple
class ToData:
    file_name = 'data/mel2wave_data.h5'
    data_key:str = 'data'
    answer_key:str = 'ans'
    def __init__(self):
        self.mel_scaler = MelScale(
            config.mel_channels,
            config.frame_rate,
            n_stft=config.fft_channels
        )
        self._load()
        self.file_num = len(self.files)
        self.prog = Value('d',0.0)

    def run(self,indata) -> Tuple[np.ndarray]:
        data = self.sound_load(indata)
        data = self.preprocess(data)
        self.progress()
        return data

    def preprocess(self,origin_sound:np.ndarray) -> np.ndarray:
        # padding original sound
        origin_sound = torch.from_numpy(origin_sound)
        sound = self.ToMelSpectrogram(origin_sound)
        dataset = self.ToDataset(sound,origin_sound)
        return dataset

    def ToDataset(self,melsound:torch.Tensor,orig_sound:torch.Tensor) -> Tuple[np.ndarray]:
        mels,sounds = [],[]
        melsound = self.mod_pad(melsound,config.speak_seq_len,dim=0)
        pad = torch.zeros(config.speak_length,dtype=orig_sound.dtype)
        orig_sound = torch.cat([orig_sound,pad])

        for e,i in enumerate(range(0,melsound.size(0),config.speak_seq_len)):
            _idx = e*config.speak_chunk_len
            _m = melsound[i:i+config.speak_seq_len]
            _s = orig_sound[_idx:_idx+config.speak_length]
            mels.append(_m)
            sounds.append(_s)

        mels = torch.stack(mels).half().numpy()
        sounds = torch.stack(sounds).half().unsqueeze(1).numpy()
        return mels,sounds
        

    def ToMelSpectrogram(self,sound:torch.Tensor) -> torch.Tensor:
        assert sound.dim() == 1
        sound = self.mod_pad(sound,config.recognize_length,dim=0)
        sound = sound.unfold(0,config.recognize_length,config.overlap_length)
        sound = torch.fft.rfft(sound,dim=-1).T.abs().float()
        sound = self.mel_scaler(sound).T
        sound= torch.log1p(sound)
        return sound

    def mod_pad(self,data:torch.Tensor,divisor:int or float, dim:int = 0) -> torch.Tensor:
        padlen = divisor - (data.size(dim) % divisor)
        padshapes = [*data.shape]
        padshapes[dim] = padlen
        pad = torch.zeros(*padshapes,dtype=data.dtype,device=data.device)
        data = torch.cat([data,pad],dim=dim)
        return data

    def sound_load(self,file_name:str) -> np.ndarray:
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

    def progress(self):
        self.prog.value += 1
        prog = self.prog.value / self.file_num * 100
        print(f'\rprogress\t{prog :4.1f}%',end='')
    
    def _load(self) -> list:
        files = glob.glob('data/checked/*.wav')
        self.files = list(set(files))

    def load(self) -> list:
        return self.files

    def save(self,data,ans) -> None:
        with h5py.File(self.file_name,'a') as f:
            if self.data_key in f:
                del f[self.data_key]
            if self.answer_key in f:
                del f[self.answer_key]

            f.create_dataset(self.data_key,data=data)
            f.create_dataset(self.answer_key,data=ans)

if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor

    todata = ToData()
    func = todata.run
    database = todata.load()

    with ThreadPoolExecutor() as p:
        result = p.map(func,database)
    
    data,ans = [],[]
    for (d,a) in result:
        data.append(d)
        ans.append(a)
    print('')
    data,ans = np.concatenate(data).transpose(0,2,1),np.concatenate(ans)
    print(data.shape,ans.shape)
    todata.save(data,ans)

    # test
    #out = func(database[0])
    #for i in out:
    #    print(i.shape)