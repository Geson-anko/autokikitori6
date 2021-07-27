"""
This is boin default hyper parameters.
"""
model_name = 'BoinAutoEncoder_siin0'
lr = 0.001
max_view_num = 1000
view_interval = 500

class encoder_hparams:
    model_name = 'BoinEncoder_siin0'
    nhid = 512
    nlayers = 2

class decoder_hparams:
    model_name = 'BoinDecoder_siin0'
    nhid = 512
    nlayers = 2
