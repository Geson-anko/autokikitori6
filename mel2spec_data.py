from DatasetLib import CreateDataset,DataHolder
import glob
import config
from pydub import AudioSegment
import numpy as np
import torch
from torchaudio.transforms import MelScale
import multiprocessing as mp

class ToData(CreateDataset):
    cpu_workers=16
    filepath = 'data/mel2spec_data.h5'
    data_key:str = 'data'
    answer_key:str = 'ans'

    def __init__(self) -> None:
        super().__init__()
        self.mel_scaler = MelScale(
            config.mel_channels,
            config.frame_rate,
            n_stft=config.fft_channels
        )
    def process(self, input_data) -> DataHolder:
        sound = self.sound_load(input_data)
        mel,fft = self.ToMelAndFFT(sound)
        mel,fft = self.ToDataset(mel,fft)
        mel = DataHolder(self.data_key,mel)
        fft= DataHolder(self.answer_key,fft)
        return  mel,fft

    def ToDataset(self,melsound,fftsound) -> DataHolder:
        mels,specs = [],[]
        
        melsound = self.mod_pad(melsound,config.speak_seq_len,dim=0).reshape(-1,config.speak_seq_len,config.mel_channels)
        fftsound = self.mod_pad(fftsound,config.speak_seq_len,dim=0).reshape(-1,config.speak_seq_len,config.fft_channels,2)

        # melsound: (B,L,C), fftsound: (B,L,C,2)
        melsound = melsound.permute(0,2,1).half().numpy()
        fftsound = fftsound.half().numpy()
        
        return melsound,fftsound

    def ToMelAndFFT(self,sound:torch.Tensor) -> torch.Tensor:
        sound = torch.from_numpy(sound)
        assert sound.dim() == 1
        sound = self.mod_pad(sound,config.recognize_length,dim=0)
        sound = sound.unfold(0,config.recognize_length,config.overlap_length)
        soundfft = torch.fft.rfft(sound,dim=-1)

        sound = soundfft.T.abs().float()
        sound = self.mel_scaler(sound).T
        sound= torch.log1p(sound)
        soundfft = torch.view_as_real(soundfft)
        soundfft = self.loglog(soundfft)
        return sound ,soundfft

    @staticmethod
    def loglog(x:torch.Tensor) -> torch.Tensor:
        minus = x < 0
        plus= x> 0
        x = x.clone()
        x[minus] = -torch.log1p(-x[minus])
        x[plus] = torch.log1p(x[plus])
        return x
    
    @staticmethod
    def iloglog(x:torch.Tensor) -> torch.Tensor:
        minus = x < 0
        plus= x> 0
        x = x.clone()
        x[minus] = -torch.exp(x[minus]) + 1
        x[plus] = torch.exp(x[plus]) -1
        return x
    
    def load(self):
        files = glob.glob('data/checked/*.wav')
        return files

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

if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import matplotlib.pyplot as plt
    to_data = ToData()
    to_data.run()
    #x = to_data.test()
    #d = x[1].data
    #plt.imshow(d[0,:,:,0].astype('float'))
    #plt.show()
    #database = to_data.load()
    #with ProcessPoolExecutor(to_data.cpu_workers) as p:
    #    result = list(p.map(to_data._process,database))
    #result = to_data.Execute(database)
    #datasets= to_data.classify(result)
    #to_data.save(to_data.filepath,*datasets,overwrite=to_data.overwrite)