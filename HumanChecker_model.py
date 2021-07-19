import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import random
import copy
import config
from Conformer import Conformer
import torchaudio

import pytorch_lightning as pl
class HumanChecker(pl.LightningModule):
    ninp:int = config.fft_channels
    seq_len:int = config.sample_seq_len
    input_size:tuple = (1,ninp,seq_len)
    output_size:tuple = (1,1)
    __insize:tuple = (-1,*input_size[1:])

    # training settings
    threshold=0
    lr = 0.001

    def __init__(
        self,
        nlayers:int = 4,
        nhid:int = 1024,
        dropout:float = 0.1,
        nhead:int = 4,
        kernel_size:int = 5,
    ):
        super().__init__()
        self.reset_seed()
        self.criterion = nn.BCEWithLogitsLoss()

        # variables
        self.conf_layers=nlayers
        self.confhid=nhid
        self.conf_dropout=dropout
        self.nhead=nhead
        self.kernel_size=kernel_size
        
        self.log("layers,hid,drop,head,kernel",[nlayers,nhid,dropout,nhead,kernel_size])
        
        # Model layers
        _dm = (self.ninp // self.nhead) * self.nhead
        self.init_dense = nn.Linear(self.ninp,_dm)
        conformer = Conformer(
            d_model=_dm,
            n_head=self.nhead,
            ff1_hsize=self.confhid,
            ff2_hsize=self.confhid,
            ff1_dropout=self.conf_dropout,
            conv_dropout=self.conf_dropout,
            ff2_dropout=self.conf_dropout,
            mha_dropout=self.conf_dropout,
            kernel_size=self.kernel_size,
        )
        self.conformers = nn.ModuleList([copy.deepcopy(conformer) for _ in range(self.conf_layers)])
        self.dense = nn.Linear(self.seq_len*_dm,1)


    def forward(self,x:torch.Tensor):

        x = x.view(self.__insize).permute(0,2,1)
        x = self.init_dense(x)
        for l in self.conformers:
            x = l(x)
        x = x.view(x.size(0),-1)
        x = self.dense(x)
        return x

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(),lr=self.lr)
        return optim

    def training_step(self,batch,idx):
        data,ans = batch
        
        out = self(data)
        loss = self.criterion(out,ans)
        acc = self.Accuracy(out,ans)
        self.log('training_loss',loss)
        self.log('training_accuracy',acc)
        return loss

    def validation_step(self,batch,idx):
        data,ans = batch
        
        out = self(data)
        loss = self.criterion(out,ans)
        acc = self.Accuracy(out,ans)
        self.log('validation_loss',loss)
        self.log('validation_accuracy',acc)
        return loss

    @classmethod
    def Accuracy(cls,output:torch.Tensor,answer:torch.Tensor) -> torch.Tensor:
        """
        output: (batch,*) output range is 0~1
        answer: (batch,*)
        return -> (1,)
        """
        assert output.shape == answer.shape

        output[output >= cls.threshold] = 1
        output[output < cls.threshold] = 0

        output = output.type(torch.bool)
        answer = answer.type(torch.bool)

        OT,OF = output==True, output==False
        AT,AF = answer==True, answer==False
        TP = torch.sum(OT==AT)
        FP = torch.sum(OT==AF)
        TN = torch.sum(OF==AF)
        FN = torch.sum(OF==AT)
        error = (TP+TN) / (TP+FP+FN+TN)
        return error

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    from torchsummaryX import summary
    model = HumanChecker()#.cuda()
    dummy = torch.randn(model.input_size)
    #dmsh = (1024,*model.input_size[1:])
    #dummy = torch.randn(dmsh,device='cuda')
    summary(model,dummy)