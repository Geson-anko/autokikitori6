import config
import torch
import numpy as np
from torch_KMeans import torch_KMeans
from pydub import AudioSegment
import torch.nn as nn
from Kikitori_fftx2_data import ToData

class to_corpus:
    def __init__(
        self,
        devcie='cuda',
        centroids_path:str='params/centroids_fftx2_euclid_32_datasize65_2021-07-04 20_28_30_k_means++.tensor',
        chars_path:str='params/chars.txt',
        batch_size:int=512,
        ):
        self.device = devcie
        self.batch_size = batch_size
        self.dtype= torch.float32

        self.centroids = torch.load(centroids_path,map_location='cpu').type(self.dtype)
        self.kmeans = torch_KMeans(0)

        self.todata = ToData()

        with open(chars_path,'r',encoding='utf-8') as f:
            self.char = f.read()

    def run(self,soundpath:str) -> str:
        centroids = self.centroids.to(self.device)
        encoded = self.todata.run(soundpath)
        encoded = torch.from_numpy(encoded).to(self.device)
        classes = self.kmeans.clustering(centroids,encoded).type(torch.long).to('cpu').detach().numpy()
        print(classes)
        sentence = ''.join([self.char[i] for i in classes])
        return f'{sentence}\n'



if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    import os
    import random
    tc = to_corpus('cuda')
    func = tc.run
    #sound = 'data/tatoeba_original/audio/4704.mp3'
    #print(func(sound))
    dirs =[
        'data/onnaotoko/'
    ]
    files = [i+q for i in dirs for q in os.listdir(i)]
    print(len(files))
    random.shuffle(files)
    print(files[:20])
    #with ProcessPoolExecutor(16) as p:
    #    result = p.map(func,files)
    result = [func(i) for i in files]
    sentence = list(result)
    with open(f'data/ajarvis_fftx2_euclid{tc.centroids.size(0)}.txt','w',encoding='utf-8') as f:
        f.writelines(sentence)