import pandas as pd
import torch
import torchvision
import numpy as np
from tqdm import tqdm


class Dataset_Biovid_image_binary_class(torch.utils.data.Dataset):
    def __init__(self,PATH_IMG,PATH_ANOT,transform = lambda x: x,IDs = None,set_type = None,nb_image = None, preload=False,loaded_resolution=224):
        self.PATH_IMG = PATH_IMG
        self.PATH_ANOT = PATH_ANOT
        self.transform = transform
        self.nb_image = nb_image
        self.set_type = set_type
        self.IDs = IDs
        self.preload = preload
        self.reset()

        dataframe = pd.read_csv(self.PATH_ANOT)
        N = len(dataframe)
        self.resize  = torchvision.transforms.Resize((loaded_resolution,loaded_resolution),antialias=True)
        self.img_loaded = torch.zeros((N,) + (3,loaded_resolution,loaded_resolution),dtype=torch.uint8)
        self.dic_image = {}
        if preload:
            loader = tqdm(dataframe.index)
            for i,p in enumerate(loader):
                img = torchvision.io.read_image(self.dataframe.loc[p,'path'])
                self.dic_image[p] = i
                img = self.resize(img)
                self.img_loaded[i]=img
        
    def __len__(self):
        return len(self.index_link)
    
    def reset(self):
        dataframe = pd.read_csv(self.PATH_ANOT)
        self.dataframe = dataframe
        
        self.dic_ID = {d : i for i,d in enumerate(set(self.dataframe['ID']))}
        if not self.nb_image is None:
            self.index_link = list(self.dataframe.sample(self.nb_image).index)
        else:
            self.index_link = list(self.dataframe.index)


    def __getitem__(self,idx):
        index = self.index_link[idx]
        try :
            i = self.dic_image[index]
        except:
            i = len(self.dic_image)
            self.dic_image[index] = i
            self.img_loaded[i] = self.resize(torchvision.io.read_image(self.dataframe.loc[index,'path']))
        
        img_tensor = self.img_loaded[i]
        img_tensor = self.transform(img_tensor)

        pain_tensor = self.dataframe.loc[index,'pain']
        video_tensor = self.dataframe.loc[index,'id_video']
        ID_tensor = self.dic_ID[self.dataframe.loc[index,'ID']]
        
        return img_tensor, pain_tensor,ID_tensor, video_tensor
