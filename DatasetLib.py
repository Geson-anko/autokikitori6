from dataclasses import dataclass
import h5py 
import numpy as np
import torch
import os
import multiprocessing as mp
from typing import Tuple,Any,List,Union
from concurrent.futures import ProcessPoolExecutor
from torch.utils import data as DataUtil

@dataclass
class DataHolder:
    name:str
    data:np.ndarray = None

    def __str__(self):
        ds = 'None'
        dt = 'None'
        if self.data is not None:
            ds = self.data.shape
            dt = self.data.dtype
        mes = f'name: {self.name}, data shape: {ds}, dtype: {dt}'
        return mes

    def __repr__(self):
        return self.__str__()

class CreateDataset:
    cpu_workers:int = os.cpu_count()//2
    filepath:str
    overwrite:bool = True

    """ descriptions

        you need to define these methods and constants
        <constants>
            `cpu_workers:int`, <- max cpu workers
            `filepath:str`, <- saving file path (h5 file).
        <methods>
            `def load(self) -> list:' <- loading input_data list.
            `def process(self,input_data) -> DataHolder:` <- your processing.
            
    """
    def __init__(self) -> None:
        #self.__prog_value = mp.Value('i',0)
        pass
    
    def test(self) -> DataHolder:
        database = self.load()
        out = self._process(database[0])
        for i in out:
            print(i)
        return out
        
    def run(self) -> None:
        database = self.load()
        result = self.Execute(database)
        datasets= self.classify(result)
        self.save(self.filepath,*datasets,overwrite=self.overwrite)

    def Execute(self,database:list) -> List[DataHolder]:
        self.__len = len(database)
        print('processing size',self.__len,end='\n')
        if self.cpu_workers <= 1:
            result = [self._process(i) for i in database]
        else:
            #with mp.Pool(self.cpu_workers) as p:
            with ProcessPoolExecutor(self.cpu_workers) as p:
                result = list(p.map(self._process,database))
        return result

    def __len__(self):
        return self.__len

    def process(self,input_data:Any) -> DataHolder:pass
    def _process(self,input_data:Any) -> DataHolder:
        out = self.process(input_data)
        #self.__progress()
        if type(out) is DataHolder:
            out = (out,)

        return out
    
    def __progress(self):
        self.__prog_value.value += 1
        per = self.__prog_value.value/self.__len * 100
        endl = ''
        if int(per) == 100:endl = '\n'
        print(f'\rprogress\t{per:4.1f}%',end=endl)


    def classify(self,result:List[DataHolder]) -> Tuple[DataHolder]:

        length  = len(result[0])
        names = [i.name for i in result[0]]
        
        results = [[] for _ in range(length)]
        for r in result:
            for idx,d in enumerate(r):
                if d.data is not None:
                    results[idx].append(d.data)
        datasets = []
        for idx in range(length):
            t = results[idx][-1].dtype
            d = np.concatenate(results[idx]).astype(t)
            dh = DataHolder(name=names[idx],data=d)
            datasets.append(dh)
        return tuple(datasets)

    def __classify_old(self,result:List[DataHolder]) -> Tuple[DataHolder]:
        if type(result[0]) is tuple:
            one = False
            length  = len(result[0])
            names = [i.name for i in result[0]]

        else:
            one = True
            length = 1
            names = [result[0].name]
        
        results = [[] for _ in range(length)]
        for r in result:
            if one:
                if r.data is not None:
                    results[0].append(r.data)
            else:
                for idx,d in enumerate(r):
                    if d.data is not None:
                        results[idx].append(d.data)
        datasets = []
        for idx in range(length):
            t = results[idx][-1].dtype
            d = np.concatenate(results[idx]).astype(t)
            dh = DataHolder(name=names[idx],data=d)
            datasets.append(dh)
        return tuple(datasets)
    def load(self) -> list:pass
    def save(self,filepath:str,*datasets:DataHolder,overwrite:bool = True) -> None:
        with h5py.File(filepath,'a') as f:
            for d in datasets:
                if d.name in f and overwrite:
                    del f[d.name]
                f.create_dataset(d.name,data=d.data)
                print('saved array:',d)

    def mod_pad(self,data:Any,divisor:int or float, dim:int = 0,padding_value:Any = 0) -> Any:
        t = type(data)
        if t is torch.Tensor:
            out = self.mod_pad_torch(data,divisor,dim,padding_value)
        elif t is np.ndarray:
            out = self.mod_pad_numpy(data,divisor,dim,padding_value)
        else:
            raise NotImplementedError(f'mod pad is not implemented {t}')
        return out

    def mod_pad_torch(self,data:torch.Tensor,divisor:int or float, dim:int = 0,padnum:Any=0) -> torch.Tensor:
        padlen = divisor - (data.size(dim) % divisor)
        padshapes = [*data.shape]
        padshapes[dim] = padlen
        pad = torch.full(tuple(padshapes),fill_value=padnum,dtype=data.dtype,device=data.device)
        #pad = torch.zeros(*padshapes,dtype=data.dtype,device=data.device)
        data = torch.cat([data,pad],dim=dim)
        return data
    
    def mod_pad_numpy(self,data:np.ndarray,divisor:int or float, dim:int = 0,padnum:Any=0) -> np.ndarray:
        padlen = divisor - (data.shape[dim] % divisor)
        padshapes = [*data.shape]
        padshapes[dim] = padlen
        pad = np.full(tuple(padshapes),fill_value=padnum,dtype=data.dtype)
        #pad = np.zeros(tuple(padshapes),dtype=data.dtype)
        data = np.concatenate([data,pad],axis=dim)
        return data

class Dataset_onMemory(DataUtil.Dataset):
    """
    this class loads datasets from h5 file.
    This class sends datasets along order of "key_names" argument.
    """

    def __init__(self,file_name:str, *key_names:str, using_length:int or slice, log:bool = False):
        super().__init__()
        if type(using_length) is int:
            using_length = slice(using_length)

        with h5py.File(file_name,'r',swmr=True) as f:
            self.data = []
            for k in key_names:
                d = f[k][using_length]
                self.data.append(torch.from_numpy(d))
                if log:
                    print(f'loaded: {k}, shape: {d.shape}')
        self.__len = len(self.data[0])

    def __len__(self):
        return self.__len

    def __getitem__(self,index):
        data = [d[index] for d in self.data]
        return tuple(data)

class Dataset_fromDrive(DataUtil.Dataset):
    """
    this class loads datasets from h5 file.
    This class sends datasets along order of "key_names" argument.
    """

    def __init__(self,file_name:str, *key_names:str, using_length:Union[int,slice], log:bool = False):
        super().__init__()
        if type(using_length) is int:
            using_length = slice(using_length)

        self.__file_name = file_name
        self.__key_names = key_names
        
        with h5py.File(file_name,'r',swmr=True) as f:
            l = f[key_names[0]].shape[0]
            if l < using_length.stop:
                pass
