#%%
import torch
import pytorch_lightning as pl
from DatasetLib import Dataset_onMemory,get_now
from voicebandSpec2_model import VoiceBand
from voicebandSpec_data import ToData
from torch.utils import data as DataUtil
from hparams import json_to_dict,voicebandSpec2_default as hparams
hparam_dict = json_to_dict('hparams/voiceband_default.json')

#%% loading dataset and defining some settings
data_set = Dataset_onMemory(ToData.filepath,ToData.target_key,using_length=1024,log=True)
batch_size = 64
EPOCHS = 100
data_loader = DataUtil.DataLoader(data_set,batch_size,shuffle=True,num_workers=0,pin_memory=False)
model = VoiceBand(hparams)

# %% training
if __name__ == '__main__':
    trainer = pl.Trainer(gpus=1,num_nodes=1,precision=16,max_epochs=EPOCHS)#,gradient_clip_val=0.5)
    trainer.fit(model,data_loader)
    now = get_now()
    name = f'params/{model.model_name}_{now}.pth'
    torch.save(model.state_dict(),name)
    print('saved')
