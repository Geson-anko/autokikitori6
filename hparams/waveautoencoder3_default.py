model_name = 'WaveAutoEncoder3'
lr = 0.0001
latent_dim=4
max_view_num = 4
view_interval = 100

class encoder_hparams:
    model_name = model_name + '_encoder'
    latent_dim = latent_dim
    lch0,lk0,nlayer0,divs0 = 64,3,3,4
    lch1,lk1,nlayer1,divs1 = 128,3,3,4
    lch2 = 256

class decoder_hparams:
    model_name = model_name + '_decoder'
    latent_dim = latent_dim
    # global 0
    ch0,k0,nlayer0,divs0 = 64,3,3,4
    # local 1
    ch1,k1,nlayer1,divs1 = 256,3,3,4
    ch2,k2,nlayer2,divs2 = 128,3,3,4
    ch3,k3,nlayer3,divs3 = 64,3,3,4
    ch4,k4,nlayer4,divs4 = 32,3,3,4
    local_channels = (ch1,ch2,ch3,ch4)
    local_kernels = (k1,k2,k3,k4)
    local_nlayers = (nlayer1,nlayer2,nlayer3,nlayer4,)
    local_divs = (divs1,divs2,divs3,divs4)
    # global 2
    ch5,k5,nlayer5,divs5 = 32,7,4,4