import numpy as np
from torch.autograd import Variable
import torch
import torchvision
import pandas as pd



def del_extra_keys(model_par_dir):
    # the pretrained model is trained on old version pytorch, some extra keys should be deleted before loading
    model_par_dict = torch.load(model_par_dir)
    model_par_dict_clone = model_par_dict.copy()
    # delete keys
    for key, value in model_par_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_par_dict[key]
    
    return model_par_dict


class data_augm:
    def __init__(self,resolution):
        self.H_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.Jitter = torchvision.transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.5,1.5),saturation=(0.5,1.5),hue=(-0.1,0.1))
        self.resize = torchvision.transforms.Resize((resolution,resolution),antialias=True)

    def transform(self,x):
        x = self.resize(x)
        x = self.H_flip(x)
        x = self.Jitter(x)
        x = x/255
        return x

class data_adapt:
    def __init__(self,resolution):
        self.resize = torchvision.transforms.Resize((resolution,resolution),antialias=True)

    def transform(self,x):
        x = self.resize(x)
        x = x/255
        return x
    

def gen_dataframe(text_file,save_file_train, save_file_test, save_file_valid, select_name_valid=None, select_name_test=None):
    dataframe = pd.read_csv(text_file, sep=" ", header=None)
    dataframe = dataframe.rename({0:'path',1:'pain'},axis='columns')
    dataframe.loc[:,['ID','id_video']] = -1
    if not select_name_valid is None and not select_name_test is None:
        id_name_file = dataframe['path'].str.split('/',expand=True)[1]
        set_name = set(id_name_file)
        for i,name in enumerate(set_name):
            dataframe.loc[id_name_file[id_name_file == name].index,['ID']] = i

            name_video = dataframe[id_name_file == name]['path'].str.split('/',expand=True)[2]
            set_video = set(name_video)
            for j,video in enumerate(set_video):
                dataframe.loc[name_video[name_video == video].index,['id_video']] = j
        valid_lines = id_name_file.isin(select_name_valid)
        test_lines = id_name_file.isin(select_name_test)
        dataframe_valid  = dataframe[valid_lines]
        dataframe_test  = dataframe[test_lines]
        dataframe_train  = dataframe[(valid_lines == False) & (test_lines == False) ]
        print(len(dataframe_train))
        dataframe_train.to_csv(save_file_train,index=False)
        dataframe_test.to_csv(save_file_test,index=False)
        dataframe_valid.to_csv(save_file_valid,index=False)
    else:
        dataframe.to_csv(save_file_train)



def combinefig_dualcon(real, fake, fake_neutral, save_num=3):
 
    save_num = min(real.shape[0], save_num)
    imgsize = np.shape(real)[-1]
    img = np.zeros([imgsize * save_num, imgsize * 5, 3])
    for i in range(0, save_num):
        img[i * imgsize: (i + 1) * imgsize, 0 * imgsize: 1 * imgsize, :] = real[i, :, :, :].transpose([1, 2, 0])
        img[i * imgsize: (i + 1) * imgsize, 1 * imgsize: 2 * imgsize, :] = fake[i, :, :, :].transpose([1, 2, 0])
        img[i * imgsize: (i + 1) * imgsize, 2 * imgsize: 3 * imgsize, :] = fake_neutral[i, :, :, :].transpose([1, 2, 0])


    return img
