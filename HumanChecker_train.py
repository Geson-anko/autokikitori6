
from HumanChecker_model import HumanChecker
import torch
from torch.utils import data as DataUtil
import config
import h5py
import pytorch_lightning as pl


tr_p,val_p = 0.9,0.1
class original_data(DataUtil.Dataset):
    def __init__(self) -> None:
        super().__init__()

        with h5py.File('data/HumanChecker.h5','r') as f:
            voices = torch.from_numpy(f['voice'][...])
            noizes = torch.from_numpy(f['noize'][...])
        self.data = torch.cat([voices,noizes])
        self.ans = torch.cat([torch.ones(voices.size(0)),torch.zeros(noizes.size(0))]).unsqueeze(-1).half()
        print(self.data.shape,self.ans.shape,self.data.dtype,self.ans.dtype)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index],self.ans[index]

orig = original_data()
trlen = int(len(orig)*tr_p)
vlen = len(orig) - trlen
trainset,valset = DataUtil.random_split(orig,[trlen,vlen])

model = HumanChecker()
batch_size = 64
EPOCHS = 50
train_loader = DataUtil.DataLoader(trainset,batch_size,shuffle=True,num_workers=4)
val_loader = DataUtil.DataLoader(valset,batch_size,shuffle=False,num_workers=4)


if __name__ == '__main__':
    trainer = pl.Trainer(gpus=1,num_nodes=1,precision=16,max_epochs=EPOCHS)
    trainer.fit(model,train_loader,val_loader)
