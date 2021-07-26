"""
This is boin default hyper parameters.
"""
model_name = 'BoinAutoEncoder'
lr = 0.001
max_view_num = 500
view_interval = 100

class encoder_hparams:
    model_name = 'BoinEncoder'
    nhid = 512
    nlayers = 2

class decoder_hparams:
    model_name = 'BoinDecoder'
    nhid = 512
    nlayers = 2
