from mel2spec_model import Mel2Spec
from mel2spec_data import ToData
import config
import torch
from torch.utils import data as DataUtil
import h5py
import pytorch_lightning as pl

uselen = -1
class OriginalData(DataUtil.Dataset):
    def __init__(self) -> None:
        super().__init__()

        with h5py.File(ToData.filepath,'r',swmr=True) as f:
            self.data = torch.from_numpy(f[ToData.data_key][:uselen])
            self.ans = torch.from_numpy(f[ToData.answer_key][:uselen])
        #print(self.data.shape,self.ans.shape)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        return self.data[index],self.ans[index]

data_set = OriginalData()

model = Mel2Spec()
batch_size = 16
EPOCHS = 1000
data_loader = DataUtil.DataLoader(data_set,batch_size,shuffle=True,num_workers=4,pin_memory=True)

if __name__ == '__main__':
    trainer = pl.Trainer(gpus=1,num_nodes=1,precision=16,max_epochs=EPOCHS)
    trainer.fit(model,data_loader)

    from datetime import datetime
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  
    name = f'params/Mel2Spec_{now}.params'
    torch.save(model.state_dict(),name)
    print('saved')