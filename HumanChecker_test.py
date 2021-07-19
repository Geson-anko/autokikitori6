import pyaudio
import numpy as np
import config
import torch
from HumanChecker_model import HumanChecker

class Test:
    device='cuda'
    param_path:str = 'params/humanchecker_nhid_1024_2021-07-03_17-06-56-692953.params'

    def __init__(self):
        
        # set audio streamer 
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            rate=config.frame_rate,
            channels=config.channels,
            format=config.pyaudio_format,
            frames_per_buffer=config.CHUNK,
            input=True
        )
        # set base tensor
        self.spectrogram = torch.zeros(
            (config.sample_seq_len,config.fft_channels),
            dtype=torch.float32,
            device=self.device
        )

        # set humanchecker
        self.humanchecker = HumanChecker(nhid=1024).to(self.device)
        self.humanchecker.load_state_dict(torch.load(self.param_path,map_location=self.device))
        self.humanchecker.half()
        self.humanchecker.eval()

    @torch.no_grad()
    def check(self):
        data = self.stream.read(config.CHUNK)
        data = np.frombuffer(data,config.audio_dtype).reshape(-1)/config.sample_range
        data = torch.from_numpy(data)
        data= data.to(self.device).unfold(0,config.recognize_length,config.overlap_length)
        data = torch.fft.rfft(data,dim=-1).abs()
        data = torch.log1p(data)
        self.spectrogram[:-config.CHUNK_seq_len] = self.spectrogram[config.CHUNK_seq_len:].clone()
        self.spectrogram[-config.CHUNK_seq_len:] = data
        checked = self.humanchecker(self.spectrogram.T.unsqueeze(0).half())
        print(f'\rhuman:{checked.view(-1).item():2.3f}',(checked > 0).item(),end='')  

    def run(self):
        try:
            while True:
                self.check()
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    Tester = Test()
    Tester.run()
