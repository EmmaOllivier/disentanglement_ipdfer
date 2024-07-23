
import os
import torch
import pandas as pd
import numpy as np
import torchvision
from torchvision.models import resnet50, ResNet50_Weights,resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from tqdm import tqdm
import model_resnet
import utils
import PIL.Image as Image
import random
import dataloader


os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

device="cuda"

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


def test():
    seed=0

    make_deterministic(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    save_path="../model/"
    Biovid_img_all = '../Biovid/sub_red_classes_img/'
    RESOLUTION = 128
    tr_test = utils.data_adapt(RESOLUTION)

    biovid_annot_test = '../test_set.csv'

    dataset_test = dataloader.Dataset_Biovid_image_binary_class(Biovid_img_all,biovid_annot_test,transform = tr_test.transform,IDs = None,nb_image = None,preload=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=20, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    loss_affect = torch.nn.CrossEntropyLoss(reduction='sum')

    encoder_expression=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    classifier_expression = model_resnet.Classif(1,False).to(device)
    
    encoder_expression.load_state_dict(torch.load(save_path+'encoder.pkl'))
    classifier_expression.load_state_dict(torch.load(save_path+'classifier.pkl'))

    encoder_expression.eval()
    classifier_expression.eval()
    loss_task_tot_val = 0
    elem_sum_val = 0
    true_response_affect_val  =0
    loop_test = tqdm(test_loader,colour='GREEN')
    for pack in loop_test:
        img_tensor = pack[0].to(device)
        pain_tensor = pack[1].float().to(device)
        ID_tensor = pack[2].to(device)

        elem_sum_val += img_tensor.shape[0]

        with torch.no_grad():

            encoded_img  = encoder_expression(img_tensor)
            output = classifier_expression(encoded_img)
            loss_task_affect_val = loss_affect(output,pain_tensor)
            loss_task_tot_val += float(loss_task_affect_val) 
            true_response_affect_val +=  float(torch.sum(output.round() == pain_tensor))


        loop_test.set_postfix(loss_task = loss_task_tot_val/elem_sum_val,accuracy_pain=true_response_affect_val/elem_sum_val*100)
        acc = true_response_affect_val/elem_sum_val*100

        
    print('\n')
    print('accuracy is : %f' % (acc))
        

if __name__ == '__main__':
    test()
