from .Boin2_default import decoder_hparams

model_name = 'VoiceBandSpec2'
lr = 0.001
view_interval = 100
max_view_len = 10
latent_dim = decoder_hparams.latent_dim
decoder_parameter_file =  'params/BoinDecoder2_2021-07-30_12-15-32.pth'
ch0,k0 = 512,3
ch1,k1,layers1,divs1 = 512,3,4,4
ch2,k2,layers2,divs2 = 512,3,3,4
