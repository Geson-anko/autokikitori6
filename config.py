"""
The Kotodama AutoEncoder (Kikitori Oshaberi pure) data file and descriptions.
今回の変換経路は、
Wave -> Spectrogram   | ここでHumanCheck


"""
import torch
import pyaudio

frame_rate:int = 16000
channels:int = 1
audio_dtype:str = 'int16'
pyaudio_format:int = pyaudio.paInt16 

frame_bit:int = 16
sample_width:int = int(frame_bit/8)
sample_range:int = 2**(frame_bit-1)

# Kikitori settings ---------
    # wave 2 phoneme settings --------
recognize_second:float = 0.04 # The time per a phoneme
recognize_length:int = int(frame_rate*recognize_second)
fft_channels:int = int(recognize_length/2) + 1
mel_channels:int = 128
fftx2_channels:int = int(mel_channels/2) + 1
n_cluster:int = 64
overlap:int = 2
overlap_length:int = int(recognize_length/overlap)
    # end of wave 2 phoneme settings

    # Human check settings ------
humancheck_threshold:float = 0.0
sample_seq_len:int = 32
sample_second:float = (recognize_second * sample_seq_len +recognize_second) / overlap
sample_length:int = int(sample_second * frame_rate)
CHUNK_seq_len:int = 8 # < sample_seq_len
CHUNK_second:float = (recognize_second * CHUNK_seq_len + recognize_second) / overlap
CHUNK:int = int(frame_rate * CHUNK_second)
    # ------ end of Human check settings.

    # Kotodama AutoEncoder settings ------
use_mem_len:int = 8
encoder_seq_len:int = 16
reduce_scaler:int = 16
generate_max_phonemes:int = use_mem_len*encoder_seq_len
origine_mem_size:int = encoder_seq_len*mel_channels
memory_size:int = int(origine_mem_size/reduce_scaler)
bos_value:float = -1.0
    # ------- end of Kotodama AutoEncoder Settings

# ------- End of Kikitori settings.

# Oshaberi settings --------
speak_seq_len:int = 64
speak_chunk_len:int = overlap_length*speak_seq_len
speak_length:int = speak_chunk_len + overlap_length
speak_second:float = speak_length/frame_rate


# ----------  End of Oshaberi settings.
