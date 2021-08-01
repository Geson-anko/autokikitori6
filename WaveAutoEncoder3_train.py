#%%
import torch
import pytorch_lightning as pl
from DatasetLib import Dataset_onMemory,get_now
from torch.utils import data as DataUtil
from WaveAutoEncoder2_data import ToData
from WaveAutoEncoder3_model import AutoEncoder
from hparams import waveautoencoder3_default as hparams

#%%
data_set = Dataset_onMemory(ToData.filepath,ToData.data_name,using_length=1024,log=True)
batch_size = 16
EPOCHS = 100
data_loader = DataUtil.DataLoader(data_set,batch_size,shuffle=True,num_workers=0,pin_memory=True)
model = AutoEncoder(hparams)

# %% training
if __name__ == '__main__':
    trainer = pl.Trainer(gpus=1,num_nodes=1,precision=16,max_epochs=EPOCHS)#,gradient_clip_val=0.5)
    trainer.fit(model,data_loader)
    now = get_now()
    name = f'params/{model.model_name}_{now}.pth'
    nameD = f'params/{model.decoder.model_name}_{now}.pth'
    torch.save(model.state_dict(),name)
    torch.save(model.decoder.state_dict(),nameD)
    print('saved')