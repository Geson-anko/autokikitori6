from mel2wave_model import Mel2Wave
from mel2wave_data import ToData
import config
import torch
from torch.utils import data as DataUtil
import h5py
import pytorch_lightning as pl

uselen = -1
class OriginalData(DataUtil.Dataset):
    def __init__(self) -> None:
        super().__init__()

        #with h5py.File(ToData.file_name,'r',swmr=True) as f:
        #    self.data = torch.from_numpy(f[ToData.data_key][:uselen])
        #    self.ans = torch.from_numpy(f[ToData.answer_key][:uselen])


        data_file = h5py.File(ToData.file_name,'r',swmr=True)
        data = data_file[ToData.data_key]
        ans = data_file[ToData.answer_key]
        self.__len = data.shape[0]
        print(data.shape,ans.shape)
        data_file.close()

    
    def __len__(self):
        return self.__len

    def __getitem__(self,index):
        data_file = h5py.File(ToData.file_name,'r',swmr=True)
        data = data_file[ToData.data_key]
        ans = data_file[ToData.answer_key]
        out = torch.from_numpy(data[index]),torch.from_numpy(ans[index])
        data_file.close()
        return out

#    def __del__(self):
#        self.data_file.close()

data_set = OriginalData()

model = Mel2Wave()
batch_size = 2
EPOCHS = 10
data_loader = DataUtil.DataLoader(data_set,batch_size,shuffle=True,num_workers=4,pin_memory=True)

if __name__ == '__main__':
    trainer = pl.Trainer(gpus=1,num_nodes=1,precision=16,max_epochs=EPOCHS)
    trainer.fit(model,data_loader)
    
    from datetime import datetime
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    name = f'params/Mel2Wave_{now}.params'
    torch.save(model.state_dict(),name)
    print('saved')