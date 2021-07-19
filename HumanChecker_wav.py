from pydub import AudioSegment
from scipy.io import wavfile
import numpy as np
import torch
import config
from HumanChecker_model import HumanChecker
from datetime import datetime
import glob
import os


class Checking:

    param_path:str = 'params/humanchecker_nhid_1024_2021-07-03_17-06-56-692953.params'
    out_folder:str = 'data/checked'
    device='cuda'
    batch_size:int = 256

    def __init__(self):
        self.humanchecker = HumanChecker(nhid=1024)#.to(self.device)
        self.humanchecker.load_state_dict(torch.load(self.param_path,map_location='cpu'))
        self.humanchecker.half()
        self.humanchecker.eval()

    def load(self) -> list:
        files = glob.glob('data/human/**/*.wav')
        dirs = [
            'data/tatoeba_original/audio',
            'data/ja',
            'data/kiritan'
        ]
        files += [os.path.join(i,q) for i in dirs for q in os.listdir(i)]
        return list(set(files))

    def load_sound(self,indata:str) -> np.ndarray:
        sound = AudioSegment.from_file(indata)
        if sound.frame_rate != config.frame_rate:
            sound = sound.set_frame_rate(config.frame_rate)
        if sound.channels != config.channels:
            sound = sound.set_channels(config.channels)
        if sound.sample_width != config.sample_width:
            sound = sound.set_sample_width(config.sample_width)
        
        usesound = np.array(sound.get_array_of_samples()).reshape(-1)
        soundarray = usesound/config.sample_range
        pad = np.zeros(config.sample_length - (len(soundarray) % config.sample_length),dtype=soundarray.dtype)
        soundarray = np.concatenate([soundarray,pad])
        soundarray = torch.from_numpy(soundarray).view(-1,config.sample_length).float()
        soundarray = soundarray.unfold(1,config.recognize_length,config.overlap_length)
        soundarray = torch.fft.rfft(soundarray,dim=-1).abs()
        soundarray = torch.log1p(soundarray).permute(0,2,1)
        return soundarray,usesound

    @torch.no_grad()
    def run(self,indata:str) -> None:

        soundarray,usesound = self.load_sound(indata)
        # send cuda and check
        humanchecker = self.humanchecker.to(self.device)
        checked = []
        for idx in range(0,len(soundarray),self.batch_size):
            _d = soundarray[idx:idx+self.batch_size].to(self.device)
            out = humanchecker(_d.half()).cpu().view(-1)
            checked.append(out)
        checked = torch.cat(checked).numpy()
        
        # cutting
        bools = checked > config.humancheck_threshold
        old = False
        startpoint = 0
        for t,b in enumerate(bools):
            if not old and b:
                startpoint = t
                old=True
            elif old and not b:
                old = False
                sou = usesound[startpoint*config.sample_length:t*config.sample_length]
                self.save(sou)

        if old:
            sou = usesound[startpoint*config.sample_length:]
            self.save(sou)

    def save(self,sound:np.ndarray) -> None:
        """
        sound : np.ndarray, int16, 1D,
        """
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f.wav')
        path = os.path.join(self.out_folder,now)
        wavfile.write(path,config.frame_rate,sound)

if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    checker = Checking()
    files = checker.load()

    with ProcessPoolExecutor(8) as p:
        p.map(checker.run,files)

    #checker.run('temp/common_voice_en_1001.mp3')