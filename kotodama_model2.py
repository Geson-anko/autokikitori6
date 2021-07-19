import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import os
import random
import config


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Encoder(nn.Module):
    ninp:int=config.mel_channels
    seq_len:int = config.encoder_seq_len
    input_size:tuple = (1,seq_len,ninp)
    output_size:tuple = (1,config.memory_size)
    __encoded_size:tuple =(-1,config.origine_mem_size)
    def __init__(
        self,nlayers:int = 4, nhid:int = 1024,nhead:int = 8, dropout:float=0.1
        ):
        super().__init__()
        self.reset_seed()

        # layers
        self.pos_encoder =PositionalEncoding(self.ninp,dropout,self.seq_len)
        encoder_layer = nn.TransformerDecoderLayer(self.ninp,nhead,nhid,dropout)
        self.encoder_layer = nn.TransformerDecoder(encoder_layer,nlayers)
        self.encoding = nn.Linear(config.origine_mem_size,config.memory_size)

    def forward(self,x:torch.Tensor):
        x = x.permute(1,0,2).contiguous()
        x = self.pos_encoder(x)
        x = self.encoder_layer(x,x).permute(1,0,2).reshape(self.__encoded_size)
        x = torch.tanh(self.encoding(x))
        return x

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

class Decoder(nn.Module):
    ninp:int = config.mel_channels
    seq_len:int = config.generate_max_phonemes
    enc_seq_len:int = Encoder.seq_len*config.use_mem_len
    cat_seq_len:int = enc_seq_len + seq_len
    input_current:tuple = (1,seq_len,ninp)
    input_memory:tuple = (1,config.use_mem_len,*Encoder.output_size[1:])
    output_size:tuple = input_current
    __inmemory:tuple = (-1,config.memory_size)
    __decoded_inmemory:tuple = (-1,enc_seq_len,ninp)

    def __init__(
        self,nlayers:int = 6, nhid:int = 1024, nhead:int = 8, dropout:float = 0.1,
        ):
        super().__init__()
        self.reset_seed()
        
        self.square_mask = self.generate_square_subsequent_mask(self.cat_seq_len)

        # layers
        self.decoding = nn.Linear(config.memory_size,config.origine_mem_size)
        self.pos_encoder= PositionalEncoding(self.ninp,dropout,self.seq_len)
        decoder_layer = nn.TransformerDecoderLayer(self.ninp,nhead,nhid,dropout)
        self.decoder_layer = nn.TransformerDecoder(decoder_layer,nlayers)
        self.fc = nn.Linear(self.ninp,self.ninp)

    def forward(self,current:torch.Tensor,memory:torch.Tensor):
        """
        current: (N,L,E) -> (-1,128,128)
        memory : (N,memlen,memsize) -> (-1,16,128)
        """
        mask = self.square_mask.to(current.device)

        # memory
        memory = memory.view(self.__inmemory)
        memory = self.decoding(memory).view(self.__decoded_inmemory).permute(1,0,2)

        # current
        current = current.permute(1,0,2)
        current = self.pos_encoder(current)
        current_memory = torch.cat([memory,current],dim=0)
        current = self.decoder_layer(current_memory,current_memory,mask,mask)[self.enc_seq_len:]
        
        current = torch.relu(self.fc(current)).permute(1,0,2).contiguous()
        return current

    @torch.no_grad()
    def predict(self,memory:torch.Tensor) -> torch.Tensor:
        """
        memory  : (N,memlen,memory_size) -> (-1,16,128)
        return  : (N,seq_len,ninp) -> (-1,128,128)
        """
        current = torch.zeros(
            memory.size(0),self.seq_len,self.ninp,
            device=memory.device,dtype=memory.dtype
        )
        current[:,:1] = config.bos_value
        for idx in range(self.seq_len -1):
            current[:,idx+1] = self(current,memory)[:,idx]
        out = current[:,1:]
        current = self(current,memory)
        out = torch.cat([out,current[:,-1:]],dim=1)
        return out


    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

class AutoEncoder(pl.LightningModule):
    input_current:tuple = Decoder.input_current
    input_memory:tuple = (1,config.use_mem_len,Encoder.seq_len,Encoder.ninp)
    __inmemory:tuple= (-1,Encoder.seq_len,Encoder.ninp)
    __encoded_inmemory:tuple = (-1,config.use_mem_len,config.memory_size)
    # learning settings
    lr:float = 0.001

    def __init__(self):
        super().__init__()
        self.reset_seed()
        self.criterion = nn.MSELoss()
        
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self,current:torch.Tensor,memory:torch.Tensor):
        memory = self.encoder(memory.view(self.__inmemory)).view(self.__encoded_inmemory)
        current = self.decoder(current,memory)
        return current

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        return optimizer
    
    def training_step(self,train_batch,batch_idx):
        _c,memory = train_batch
        memidx = torch.randperm(memory.size(1))
        memory = memory[:,memidx]
        ans,current = _c[:,1:],_c[:,:-1]
        out = self(current,memory)
        loss = self.criterion(out,ans)
        self.log('training loss',loss)
        return loss

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    from torchsummaryX import summary
    model = Decoder()
    current = torch.randn(model.input_current)
    memory = torch.randn(model.input_memory)
    summary(model,current,memory)
    #out = model.predict(memory)
    #print(out.shape)
    #from torch.utils.tensorboard import SummaryWriter
    #writer = SummaryWriter()
    #writer.add_graph(model,[current,memory])
    #writer.close()