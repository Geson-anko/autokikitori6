#%% importing libs
import torch
import pytorch_lightning as pl
from DatasetLib import Dataset_onMemory,get_now
from Boin_data import ToData
import config
from hparams import Boin2_default as hparam
from torch.utils import data as Datautil
from Boin2_model import AutoEncoder
#%% loading dataset and defining some settings
data_set = Dataset_onMemory(ToData.filepath,ToData.data_key,using_length=-1,log=False)
batch_size = 8192*4
EPOCHS = 5000
data_loader = Datautil.DataLoader(data_set,batch_size,shuffle=False,num_workers=0,pin_memory=True)

model = AutoEncoder(hparam)

#%% training
if __name__ == '__main__':
    trainer = pl.Trainer(gpus=1,num_nodes=1,precision=16,max_epochs=EPOCHS)#,gradient_clip_val=0.5)
    trainer.fit(model,data_loader)
    now = get_now()
    name = f'params/{model.model_name}_{now}.pth'
    nameD =f'params/{model.decoder.model_name}_{now}.pth'
    torch.save(model.state_dict(),name)
    torch.save(model.decoder.state_dict(),nameD)
    print('saved')
