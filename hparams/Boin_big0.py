"""
This is boin default hyper parameters.
"""
model_name = 'BoinAutoEncoder_big0'
lr = 0.001
max_view_num = 1000
view_interval = 500

class encoder_hparams:
    model_name = 'BoinEncoder_big0'
    nhid = 512
    nlayers = 3

class decoder_hparams:
    model_name = 'BoinDecoder_big0'
    nhid = 512
    nlayers = 3
