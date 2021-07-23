from VCGAN_model import VCGAN
import VCGAN_data as ToData
import config
import torch
import pytorch_lightning as pl
from torch.utils import data as DataUtil
from DatasetLib import Dataset_VCGAN, get_now
from hparams import VCGAN_default as hparam

data_set = Dataset_VCGAN(ToData.filepath,ToData.source_key,ToData.target_key,using_length=2048,log=False)

model = VCGAN(hparams=hparam)
batch_size = 512
EPOCHS =10000
data_loader = DataUtil.DataLoader(data_set,batch_size,shuffle=True,num_workers=4,pin_memory=True)

if __name__ == '__main__':
    trainer = pl.Trainer(gpus=1,num_nodes=1,precision=16,max_epochs=EPOCHS,gradient_clip_val=0.5)
    trainer.fit(model,data_loader)
    now = get_now()
    nameG = f'params/{model.generator.model_name}_{now}.pth'
    name = f'params/{model.model_name}_{now}.pth'
    torch.save(model.generator.state_dict(),nameG)
    torch.save(model.state_dict(),name)
    print('saved')