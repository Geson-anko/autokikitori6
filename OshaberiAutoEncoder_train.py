from oshaberi_model import OshaberiAutoEncoder
from mel2wave_data import ToData
import config
import torch
import pytorch_lightning as pl
from DatasetLib import Dataset_onMemory
import matplotlib.pyplot as plt
from torch.utils import data as DataUtil

data_set = Dataset_onMemory(ToData.file_name,ToData.data_key,using_length=-1,log=False)

model = OshaberiAutoEncoder('params/WaveDecoder_2021-07-16_08-48-54.params')
batch_size = 16
EPOCHS = 1000
data_loader = DataUtil.DataLoader(data_set,batch_size,shuffle=True,num_workers=4,pin_memory=True)

if __name__ == '__main__':
    trainer = pl.Trainer(gpus=1,num_nodes=1,precision=16,max_epochs=EPOCHS)
    trainer.fit(model,data_loader)

    from datetime import datetime
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    name = f'params/OshaberiAutoEncoder_{now}.params'
    torch.save(model.state_dict(),name)
    print('saved')