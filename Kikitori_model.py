import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import os
import random
import config

class Encoder(nn.Module):
    input_size:tuple = (1,config.mel_channels,config)