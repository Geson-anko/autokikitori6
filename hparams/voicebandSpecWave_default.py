from .waveautoencoder3_default import decoder_hparams

model_name = 'VoiceBandSpecWave'
lr = 0.001
view_interval = 1
max_view_len =10
latent_dim = decoder_hparams.latent_dim
decoder_parameter_file= 'params/WaveAutoEncoder3_decoder_2021-08-02_08-27-15.pth'
ch0,k0 = 512,3
ch1,k1,layers1,divs1 = 512,3,4,4
ch2,k2,layers2,divs2 = 512,3,3,4