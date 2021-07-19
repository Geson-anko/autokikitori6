from WaveAutoEncoder_model import WaveAutoEncoder
from WaveAutoEncoder_data import ToData
import config
import torch
import pytorch_lightning as pl
from DatasetLib import Dataset_onMemory
import matplotlib.pyplot as plt
from torch.utils import data as DataUtil

data_set = Dataset_onMemory(ToData.filepath,ToData.data_name,using_length=-1,log=False)

model = WaveAutoEncoder()
batch_size = 1024
EPOCHS = 1000
data_loader = DataUtil.DataLoader(data_set,batch_size,shuffle=True,num_workers=4,pin_memory=True)

if __name__ == '__main__':
    trainer = pl.Trainer(gpus=1,num_nodes=1,precision=16,max_epochs=EPOCHS)
    trainer.fit(model,data_loader)

    from datetime import datetime
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    nameE = f'params/WaveEncoder_{now}.params'
    nameD = f'params/WaveDecoder_{now}.params'
    name = f'params/WaveAutoEncoder_{now}.params'
    torch.save(model.encoder.state_dict(),nameE)
    torch.save(model.decoder.state_dict(),nameD)
    torch.save(model.state_dict(),name)
    print('saved')