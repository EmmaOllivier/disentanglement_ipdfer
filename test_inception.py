import os
import torch
import pandas as pd
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights,resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import model_inception
import utils
import random
import dataloader
from torch.autograd import Variable

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


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def test_per_video_softmax_output():

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

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    dic_log = {'accuracy':[],'threshold':[]}

    loss_affect = torch.nn.CrossEntropyLoss(reduction='sum')
   
    Gen = model_inception.Gen(clsn_ER=2, Nz=256, GRAY=False, Nb=6).to(device)
    Gen.load_state_dict(torch.load(save_path+'generator.pkl'))


    threshold = [round(x * 0.05, 2) for x in range(0, 22)]
    #threshold = [0.52]


    for t in threshold:

        print(t)

        Gen.eval()
        pre_list = []
        GT_list = []

        val_ce = 0
        video_results=[]
        video_output=[]
        current_video=-1
        loop_test = tqdm(test_loader ,colour='GREEN')
        for i, (batch_val_x, batch_val_y, batch_val_id, batch_val_video) in enumerate(loop_test):

            if(batch_val_video.data[0] != current_video and current_video!=-1):
                video_mean=sum(video_output)/len(video_output)

                if(video_mean>=t):
                    reg_video=[1]*len(video_results)
                else:
                    reg_video=[0]*len(video_results)

                pre_list = np.hstack((pre_list, reg_video))
                val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
                loop_test.set_postfix(accuracy_pain=val_acc*100)
                current_video=batch_val_video.data[0]
                video_results=[]
                video_output=[]
 
            elif(batch_val_video.data[0] != current_video):
                current_video=batch_val_video.data[0]


            GT_list = np.hstack((GT_list, batch_val_y.numpy()))
            batch_val_x = Variable(batch_val_x).to(device)
            batch_val_y = Variable(batch_val_y).to(device)
            batch_val_y_np = batch_val_y.data.cpu().numpy()

            batch_fea, _= Gen.enc_ER(batch_val_x)
            batch_p = Gen.fc_ER(batch_fea)
            batch_fea_np = batch_p.data.cpu().numpy()

            batch_results = batch_p.cpu().data.numpy().argmax(axis=1)
            video_results=np.hstack((video_results, batch_results))
            batch_output = batch_p.cpu().data.numpy()
            batch_output=softmax(batch_output.flatten())

            video_output = np.hstack((video_output, batch_output[1]))
            val_ce += loss_affect(batch_p, batch_val_y).cpu().data.numpy()

            if(i==len(test_loader)-1):
                print("last")
                video_mean=sum(video_results)/len(video_results)
                print(video_mean)

                if(video_mean>t):
                    reg_video=[1]*len(video_results)
                else:
                    reg_video=[0]*len(video_results)

                pre_list = np.hstack((pre_list, reg_video))
                val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
                loop_test.set_postfix(accuracy_pain=val_acc*100)
            

        val_acc_pain = (np.sum(((GT_list != pre_list) & (GT_list == 1 )).astype(float)) / (np.sum((GT_list == 1).astype(float))))
        print(val_acc_pain)

        val_acc_no_pain = (np.sum(((GT_list != pre_list) & (GT_list == 0 )).astype(float)) / (np.sum((GT_list == 0).astype(float))))
        print(val_acc_no_pain)

        val_acc = (np.sum((GT_list == pre_list).astype(float))) / len(GT_list)
        val_ce = val_ce / i
        print(val_acc)


        dic_log['threshold'].append(t)
        dic_log['accuracy'].append(val_acc)


        dataframe = pd.DataFrame(dic_log)
        dataframe.to_csv(save_path+"result_per_video_threshold.csv")



def test_per_subject():
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

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    loss_affect = torch.nn.CrossEntropyLoss(reduction='sum')

    Gen = model_inception.Gen(clsn_ER=2, Nz=256, GRAY=False, Nb=6).to(device)
    Dis_ER = model_inception.Dis(GRAY=False, cls_num=2).to(device)
    
    Gen.load_state_dict(torch.load(save_path+'generator.pkl'))
    Dis_ER.load_state_dict(torch.load(save_path+'discriminator.pkl'))

    Gen.eval()
    Dis_ER.train()
    loss_task_tot_val = 0
    elem_sum_val = 0
    true_response_affect_val  =0

    accuracy_list=[]
    current_subject=-1
    loop_test = tqdm(test_loader,colour='GREEN')
    for pack in loop_test:
        
        img_tensor = pack[0].to(device)
        pain_tensor = pack[1].to(device)
        ID_tensor = pack[2].to(device)
        if (current_subject==ID_tensor or current_subject==-1):
            current_subject=ID_tensor
            elem_sum_val += img_tensor.shape[0]

            with torch.no_grad():
                #Encoding
                encoded_img_exp, encoded_vec_exp  = Gen.enc_ER(img_tensor)
                # TASK Affect
                output = Gen.fc_ER(encoded_img_exp)
                loss_task_affect_val = loss_affect(output,pain_tensor)
                loss_task_tot_val += float(loss_task_affect_val) 
                true_response_affect_val +=  float(torch.sum(output.max(dim=-1)[1] == pain_tensor))

            acc = true_response_affect_val/elem_sum_val*100
            loop_test.set_postfix(accuracy_pain=acc)
              
        elif(current_subject!=ID_tensor):
            print("new")
            current_subject=ID_tensor
            accuracy_list.append(acc)

            loss_task_tot_val = 0
            elem_sum_val = 0
            true_response_affect_val=0

            elem_sum_val += img_tensor.shape[0]

            with torch.no_grad():
                encoded_img_exp, encoded_vec_exp  = Gen.enc_ER(img_tensor)
                output = Gen.fc_ER(encoded_img_exp)
                loss_task_affect_val = loss_affect(output,pain_tensor)
                loss_task_tot_val += float(loss_task_affect_val) 
                true_response_affect_val +=  float(torch.sum(output.max(dim=-1)[1] == pain_tensor))

            acc = true_response_affect_val/elem_sum_val*100
            loop_test.set_postfix(accuracy_pain=acc)
    
    return accuracy_list
            



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

    dataset_test = dataloader.Dataset_Biovid_image_binary_class(Biovid_img_all,biovid_annot_valid,transform = tr_test.transform,IDs = None,nb_image = None,preload=False)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=100, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)


    loss_affect = torch.nn.CrossEntropyLoss(reduction='sum')

    Gen = model_inception.Gen(clsn_ER=2, Nz=256, GRAY=False, Nb=6).to(device)
    Dis_ER = model_inception.Dis(GRAY=False, cls_num=2).to(device)
    
    Gen.load_state_dict(torch.load(save_path+'generator.pkl'))
    Dis_ER.load_state_dict(torch.load(save_path+'discriminator.pkl'))

    Gen.eval()
    Dis_ER.train()
    loss_task_tot_val = 0
    elem_sum_val = 0
    true_response_affect_val  =0

    loop_test = tqdm(test_loader,colour='GREEN')
    for pack in loop_test:
        img_tensor = pack[0].to(device)
        pain_tensor = pack[1].to(device)
        ID_tensor = pack[2].to(device)

        elem_sum_val += img_tensor.shape[0]

        with torch.no_grad():

            encoded_img_exp, encoded_vec_exp  = Gen.enc_ER(img_tensor)

            output = Gen.fc_ER(encoded_img_exp)
            loss_task_affect_val = loss_affect(output,pain_tensor)
            loss_task_tot_val += float(loss_task_affect_val) 
            true_response_affect_val +=  float(torch.sum(output.max(dim=-1)[1] == pain_tensor))


        loop_test.set_postfix(loss_task = loss_task_tot_val/elem_sum_val,accuracy_pain=true_response_affect_val/elem_sum_val*100)
        acc = true_response_affect_val/elem_sum_val*100

        
    print('\n')
    print('accuracy is : %f' % (acc))


if __name__ == '__main__':
    test()


