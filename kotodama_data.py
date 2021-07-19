import torch
import numpy as np
from pydub import AudioSegment
import config
from torchaudio.transforms import MelScale
import h5py
import glob
from multiprocessing import Value
""" description
This code can preprocess a file per `run()`.
If you want to preprocess with HumanChecker, please inherit ToData and override `run()`
"""

class ToData:
    
    file_name = 'data/kotodamaBOS_data.h5'
    current_key:str = 'current'
    memory_key:str = 'memory'

    def __init__(self):
        self.mel_scaler = MelScale(
            config.mel_channels,
            config.frame_rate,
            n_stft=config.fft_channels
        )
        self.__load()
        self.file_num = len(self.files)
        self.prog = Value('d',0.0)

    def run(self,indata) -> np.ndarray:
        data = self.sound_load(indata)
        data = self.preprocess(data)
        self.progress()
        return data

    def progress(self):
        self.prog.value += 1
        prog = self.prog.value / self.file_num * 100
        print(f'\rprogress\t{prog :4.1f}%',end='')

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
    
    def preprocess(self,sound_:np.ndarray) -> np.ndarray:
        # padding original sound
        sound = torch.from_numpy(sound_)
        padlen = config.recognize_length - (len(sound) % config.recognize_length)
        pad = torch.zeros(padlen,dtype=sound.dtype)
        sound = torch.cat([sound,pad])
        # preprocess ----
        sound = sound.unfold(0,config.recognize_length,config.overlap_length)
        cur,mem = [],[]
        if sound.size(0) == 0:
            return cur,mem
        sound = torch.fft.rfft(sound,dim=-1).T.abs().float()
        sound = self.mel_scaler(sound).T
        sound = torch.log1p(sound)

        for idx in range(0,len(sound),config.generate_max_phonemes):
            # make current
            s = sound[idx:idx+config.generate_max_phonemes]
            padlen = config.generate_max_phonemes - len(s)
            pad = torch.zeros(padlen,config.mel_channels,dtype=s.dtype)
            cs = torch.cat([s,pad])
            cur.append(cs)

            # make memory
            ms = []
            for i in range(0,len(s),config.encoder_seq_len):
                _ms = s[i:i+config.encoder_seq_len]
                if _ms.size(0) != config.encoder_seq_len:
                    padlen = config.encoder_seq_len - _ms.size(0)
                    pad = torch.zeros(padlen,config.mel_channels,dtype=_ms.dtype)
                    _ms = torch.cat([_ms,pad])
                ms.append(_ms)

            if len(ms) < config.use_mem_len:
                mri = torch.randperm(len(s))[:8-len(ms)]
                for i in mri:
                    _ms = s[i:i+config.encoder_seq_len]
                    if _ms.size(0) != config.encoder_seq_len:
                        padlen = config.encoder_seq_len - _ms.size(0)
                        pad = torch.zeros(padlen,config.mel_channels,dtype=_ms.dtype)
                        _ms = torch.cat([_ms,pad])
                    ms.append(_ms)    


            ms = torch.stack(ms)
            if len(ms) < config.use_mem_len:
                padlen = config.use_mem_len - len(ms)
                pad = torch.zeros(padlen,config.encoder_seq_len,config.mel_channels)
                ms = torch.cat([ms,pad])
            idx = torch.randperm(config.use_mem_len)
            ms = ms[idx]

            mem.append(ms)

        # stacking and adding answer tensor
        cur = torch.stack(cur).half()
        mem = torch.stack(mem).half()

        ans_vector = torch.zeros((cur.size(0),1,config.mel_channels),dtype=cur.dtype) + config.bos_value
        #ans_vector = torch.zeros((cur.size(0),1,config.mel_channels),dtype=cur.dtype)
        cur = torch.cat([ans_vector,cur],dim=1)

        return cur.numpy(),mem.numpy()

    def __load(self) -> None:
        files = glob.glob('data/checked/*.wav')
        self.files = list(set(files))

    def load(self) -> list:
        return self.files

    def save(self,ans_current,memory) -> None:
        with h5py.File(self.file_name,'a') as f:
            if self.current_key in f:
                del f[self.current_key]
            if self.memory_key in f:
                del f[self.memory_key]
            f.create_dataset(self.current_key,data=ans_current)
            f.create_dataset(self.memory_key,data=memory)
if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor

    todata = ToData()
    func = todata.run
    database = todata.load()

    with ThreadPoolExecutor(8) as p:
        result = p.map(func,database)
    print('')

    cur,mem = [],[]
    for (c,m) in result:
        cur.append(c)
        mem.append(m)
    cur,mem = np.concatenate(cur),np.concatenate(mem)
    print(cur.shape,mem.shape)
    todata.save(cur,mem)
    
    # test
    #out = func(database[0])
    #for i in out:
    #    print(i)
