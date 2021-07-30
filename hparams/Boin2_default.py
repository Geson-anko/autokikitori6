"""
This is boin default hyper parameters.
"""
model_name = 'BoinAutoEncoder2'
lr = 0.001
max_view_num = 1000
view_interval = 100
latent_dim = 4

class encoder_hparams:
    model_name = 'BoinEncoder2'
    nhid = 512
    nlayers = 3
    latent_dim = latent_dim

class decoder_hparams:
    model_name = 'BoinDecoder2'
    nhid = 512
    nlayers = 3
    latent_dim = latent_dim