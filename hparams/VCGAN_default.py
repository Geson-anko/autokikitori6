"""
This is VCGAN hyper parameter settings.
"""
model_name = 'VCGAN'
lr = 0.0001
true_loss_scaler = 100.0


class generator_hparams:
    model_name = 'VCGAN_generator'
    ch0,kernel0,num_layer0 = 256,5,3
    ch1, kernel1, num_reslayers, divisor = 512,3,3,4
    kernel2 = 3

class disciminator_hparams:
    model_name = 'VCGAN_discriminator'
    nlayers:int = 4
    nhid:int = 1024
    dropout:float = 0.1
    nhead:int = 4
    kernel_size:int = 5

