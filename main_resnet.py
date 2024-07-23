import os
import torch
import pandas as pd
import torchvision
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from utils import data_augm,data_adapt
from dataloader import Dataset_Biovid_image_binary_class
import torch
import random
import train_resnet

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

device = "cuda"

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_deterministic(seed=0):
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)



BATCH_SIZE = 200
RESOLUTION = 128
nb_ID = 49

LEARNING_RATE = 0.01
LEARNING_RATE_FINETUNE = 0.000005
EPOCH_PRETRAIN = 10
EPOCH_FINETUNE = 20
FOLD=5

seed=42

make_deterministic(seed)

g = torch.Generator()
g.manual_seed(seed)

tr = data_augm(RESOLUTION)
tr_test = data_adapt(RESOLUTION)
tr_size = torchvision.transforms.Resize((RESOLUTION,RESOLUTION),antialias=True)


for fold in range(1, FOLD+1):

    print(f"Fold {fold}")
    print("-------")

    save_path_pretrained="/home/ens/eollivier/ipd_disentanglement/model_resnet/"+str(fold)+"/"
    save_path="/home/ens/eollivier/ipd_disentanglement/model_resnet_test/"+str(fold)+"/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_path="/home/ens/eollivier/ipd_disentanglement/model_resnet_test/"+str(fold)+"/img/"

    if not os.path.exists(img_path):
        os.makedirs(img_path)


    save_log_name='log_lr'+str(LEARNING_RATE)+'_pretrained.csv'
    biovid_annot_train="/home/ens/eollivier/Biovid_corrected/train"+str(fold)+"_order.csv"
    biovid_annot_val = "/home/ens/eollivier/Biovid_corrected/valid"+str(fold)+"_order.csv"

    Biovid_img_all =  "/state/share1/datasets/Biovid/sub_red_classes_img/"
    dataset_train = Dataset_Biovid_image_binary_class(Biovid_img_all,biovid_annot_train,transform = tr.transform,IDs = None,nb_image = None, preload=False)
    loader_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=BATCH_SIZE, shuffle=True,
                                                num_workers=10,drop_last = True, worker_init_fn=seed_worker, generator=g) 

    dataset_test = Dataset_Biovid_image_binary_class(Biovid_img_all,biovid_annot_val,transform = tr_test.transform,IDs = None,nb_image = None, preload=False)
    loader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=1,
                                                num_workers=10, worker_init_fn=seed_worker, generator=g)


    train_resnet.train_id(save_path, fold, loader_train, loader_test, device, EPOCH=3)

    encoder_id=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    encoder_id.load_state_dict(torch.load(save_path+'encoder_id_pretrained.pt'))

    train_resnet.train(save_path, img_path, encoder_id, loader_train, loader_test,device, batch_size=BATCH_SIZE)


