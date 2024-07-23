import os
import torch
import pandas as pd
import numpy as np
import torchvision
from torchvision.models import resnet50, ResNet50_Weights,resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import model_resnet
import utils
import PIL.Image as Image


def train_id(save_path, fold, loader_train, loader_test, device, EPOCH=20):

    encoder_id=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    classifier_id = model_resnet.Classif(49).to(device)

    save_model_encoder = 'encoder_id_pretrained.pt'
    loss_ID = torch.nn.CrossEntropyLoss(reduction='sum')
    
    param_classifier_id = [{'params': encoder_id.parameters(), 'lr': 0.0001},
                        {'params': classifier_id.parameters(), 'lr': 0.000005},
                        ]
    optimizer = torch.optim.Adam(param_classifier_id )

    for epoch in range(EPOCH):
        
        loader_train.dataset.reset()
        loss_task_tot = 0
        elem_sum = 0
        true_response_affect = 0
        true_response_ID = 0

        encoder_id.train()
        classifier_id.train()

        loop_train = tqdm(loader_train,colour='BLUE')
        for i,pack in enumerate(loop_train):

            img_tensor = pack[0].to(device)
            pain_tensor = pack[1].float().to(device)
            ID_tensor = pack[2].to(device)
            elem_sum += img_tensor.shape[0]

            encoded_img  = encoder_id(img_tensor)

            output = classifier_id(encoded_img)
            loss_task_ID = loss_ID(output,ID_tensor) 
            loss_task_tot += float(loss_task_ID) 
            true_response_ID += float(torch.sum(output.max(dim=-1)[1] == ID_tensor))

            loss =  loss_task_ID

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        

            loop_train.set_description(f"Epoch [{epoch}/{EPOCH}] training")
            loop_train.set_postfix(loss_task = loss_task_tot/elem_sum,accuracy=true_response_ID/elem_sum*100)
        
        
            
        torch.save(encoder_id.state_dict(),save_path+save_model_encoder)



def train(save_path, img_path, encoder_id, loader_train, loader_test,device,lr=0.0001,EPOCH=20, batch_size=20):

    save_log_name='ipd.csv'


    encoder_expression=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    feature_extractor_exp = torch.nn.Sequential(*list(encoder_expression.children())[:-1])
    feature_extractor_id = torch.nn.Sequential(*list(encoder_id.children())[:-1])
    classifier_expression = model_resnet.Classif(1,False).to(device)

    encoder_discriminator=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    classifier_discriminator = model_resnet.Classif(1,False).to(device)

    decoder=model_resnet.decoder().to(device)

    loss_affect = torch.nn.BCELoss(reduction='sum')
    loss_cosine = torch.nn.CosineEmbeddingLoss(reduction='sum')
    L1_loss = torch.nn.L1Loss()

    optimizer_classifier = torch.optim.Adam(list(encoder_expression.parameters())+list(classifier_expression.parameters()),lr=0.000001)
    optimizer_discriminator = torch.optim.Adam(list(encoder_discriminator.parameters())+list(classifier_discriminator.parameters()),lr=0.000001)
    
    param_generator = [{'params': encoder_expression.parameters(), 'lr': 0.000001},
                        {'params': decoder.parameters(), 'lr': 0.0001},
                        ]
    optimizer_generator = torch.optim.Adam(param_generator)

    dic_log = {'acc_val':[], 'true_positive':[], 'true_negative':[], 'F1':[], 'acc_val_dis':[], 'acc_train':[],'acc_train_dis':[]}
    acc_max = 0 
    acc_max_b = 0 
    tt_acc_mat=[]
    acc_max_dis = 0 
    tt_acc_dis_mat=[]
    
    
    for epoch in range(EPOCH):
        

        loader_train.dataset.reset()
        loss_task_tot = 0
        loss_task_tot_d = 0
        loss_tot_id = 0
        loss_tot_recon = 0
        loss_tot_task_dis = 0
        loss_cos_tot=0
        loss_task_neutral_tot_g = 0
        loss_task_affect_tot_g = 0
        elem_sum = 0
        true_response_affect = 0
        true_response_affect_d = 0
        true_response_affect_g = 0

        encoder_expression.train()
        classifier_expression.train()
        encoder_discriminator.train()
        classifier_discriminator.train()
        decoder.train()


        loop_train = tqdm(loader_train,colour='BLUE')
        for i,pack in enumerate(loop_train):

            img_tensor = pack[0].to(device)
            pain_tensor = pack[1].float().to(device)
            ID_tensor = pack[2].to(device)
            elem_sum += img_tensor.shape[0]

            encoded_img_exp  = encoder_expression(img_tensor)
            with torch.no_grad():
                encoded_img_id = encoder_id(img_tensor)
            
            output = classifier_expression(encoded_img_exp)
            loss_task_affect = loss_affect(output,pain_tensor)
            loss_task_tot += float(loss_task_affect) 
            true_response_affect += float(torch.sum(output.round() == pain_tensor))

            y = torch.full((batch_size,), -1).to(device)

            loss_cos = loss_cosine(encoded_img_exp, encoded_img_id, y)

            loss_c =  loss_task_affect + loss_cos
            loss_cos_tot += float(loss_cos) 
            optimizer_classifier.zero_grad()
            loss_c.backward()
            optimizer_classifier.step()

            encoded_img_exp_d  = encoder_discriminator(img_tensor)
            output_d = classifier_discriminator(encoded_img_exp_d)
            loss_task_affect_d = loss_affect(output_d,pain_tensor)
            loss_task_tot_d += float(loss_task_affect_d) 
            true_response_affect_d += float(torch.sum(output_d.round() == pain_tensor))
            
            loss_d =  loss_task_affect_d

            optimizer_discriminator.zero_grad()
            loss_d.backward()
            optimizer_discriminator.step()


            img_g = decoder(feature_extractor_exp(img_tensor), feature_extractor_id(img_tensor), device=device)
            img_id_g = decoder(feature_extractor_id(img_tensor), feature_extractor_id(img_tensor), device=device)

            encoded_g = encoder_discriminator(img_g)
            output_g = classifier_discriminator(encoded_g)

            encoded_id_g = encoder_discriminator(img_id_g)
            output_id_g = classifier_discriminator(encoded_id_g)

            loss_task_affect_g = loss_affect(output_g,pain_tensor)
            loss_task_affect_tot_g += float(loss_task_affect_g)
            true_response_affect_g += float(torch.sum(output_g.round() == pain_tensor))

            neutral_label = torch.zeros(batch_size).to(device)

            loss_task_neutral_g = loss_affect(output_id_g,neutral_label)

            loss_recon_affect = L1_loss(img_g, img_tensor)


            loss_recon_neutral=0
            for j in range(batch_size):
                if(pain_tensor[j]==torch.zeros(1).to(device)):
                    loss_recon_neutral += L1_loss(img_id_g[j], img_tensor[j])

            loss_recon = loss_recon_affect + loss_recon_neutral

            id_fea_r = encoder_id(img_tensor)
            id_fea_f = encoder_id(img_g)
            id_fea_neutral_f = encoder_id(img_id_g)

            loss_id_1 = L1_loss(id_fea_r, id_fea_f)
            loss_id_2 = L1_loss(id_fea_r, id_fea_neutral_f)
            loss_id = loss_id_1 + loss_id_2
            loss_tot_id += float(loss_id)
            loss_tot_recon += float(loss_recon)
            loss_tot_task_dis += float(loss_task_affect_g)

            loss_g = 25 * loss_recon + 0.8 * loss_task_affect_g + 0.2 * loss_task_neutral_g +loss_id
            optimizer_generator.zero_grad()
            loss_g.backward()
            optimizer_generator.step()

            if i % 100 == 0:

                comb_img = utils.combinefig_dualcon(img_tensor.cpu().data.numpy(),
                                                img_g.cpu().data.numpy(), img_id_g.cpu().data.numpy())
                comb_img = Image.fromarray((comb_img * 255).astype(np.uint8))
                comb_img.save(os.path.join(img_path, str(epoch) + '_' + str(i) + '.jpg'))

            loop_train.set_description(f"Epoch [{epoch}/{EPOCH}] training")
            loop_train.set_postfix(loss_task = loss_task_tot/elem_sum,accuracy_pain=true_response_affect/elem_sum*100, recon=loss_tot_recon/elem_sum, id=loss_tot_id/elem_sum, task_affect_g=loss_task_affect_tot_g/elem_sum, task_neutral_g=loss_task_neutral_tot_g/elem_sum,accuracy_dis=true_response_affect_d/elem_sum*100, cos=loss_cos_tot/elem_sum)
        

        encoder_expression.eval()
        classifier_expression.eval()
        encoder_discriminator.eval()
        classifier_discriminator.eval()

        loss_task_tot_val = 0
        loss_task_tot_val_dis = 0
        elem_sum_val = 0
        true_response_affect_val  =0
        true_response_affect_val_pain  =0
        true_response_affect_val_neutral  =0
        true_response_affect_val_dis = 0
        predicted_positive = 0
        elem_pain=0
        elem_no_pain=0
        loop_test = tqdm(loader_test,colour='GREEN')
        for pack in loop_test:
            img_tensor = pack[0].to(device)
            pain_tensor = pack[1].float().to(device)
            ID_tensor = pack[2].to(device)

            elem_sum_val += img_tensor.shape[0]

            with torch.no_grad():

                encoded_img_exp  = encoder_expression(img_tensor)
                output = classifier_expression(encoded_img_exp)
                loss_task_affect_val = loss_affect(output,pain_tensor)
                loss_task_tot_val += float(loss_task_affect_val) 
                true_response_affect_val +=  float(torch.sum(output.round() == pain_tensor))
                true_response_affect_val_pain +=  float(torch.sum(output.round() == pain_tensor) & (pain_tensor==torch.ones(img_tensor.shape[0]).to(device)))
                true_response_affect_val_neutral +=  float(torch.sum(output.round() == pain_tensor) & (pain_tensor==torch.zeros(img_tensor.shape[0]).to(device)))
                predicted_positive += float(torch.sum(output.round() == torch.ones(img_tensor.shape[0]).to(device)))

                encoded_img_exp_d = encoder_discriminator(img_tensor)
                output_d = classifier_discriminator(encoded_img_exp_d)
                loss_task_affect_val_dis = loss_affect(output_d,pain_tensor)
                loss_task_tot_val_dis += float(loss_task_affect_val_dis) 
                true_response_affect_val_dis +=  float(torch.sum(output_d.round() == pain_tensor))

                elem_pain += float(torch.sum(pain_tensor==torch.ones(img_tensor.shape[0]).to(device)))
                elem_no_pain += float(torch.sum(pain_tensor==torch.zeros(img_tensor.shape[0]).to(device)))

            loop_test.set_postfix(loss_task = loss_task_tot_val/elem_sum_val,accuracy_pain=true_response_affect_val/elem_sum_val*100, precision=true_response_affect_val_pain/elem_sum_val*100, recall=true_response_affect_val_neutral/elem_sum_val*100, accuracy_dis=true_response_affect_val_dis/elem_sum_val*100)
            acc = true_response_affect_val/elem_sum_val*100
            acc_dis = true_response_affect_val_dis/elem_sum_val*100

        precision = true_response_affect_val_pain/predicted_positive
        recall = true_response_affect_val_pain/elem_pain
        true_positive= true_response_affect_val_pain/elem_pain*100
        true_negative= true_response_affect_val_neutral/elem_no_pain*100
        f1= 2 * (precision*recall)/(precision+recall)
        tt_acc_mat.append(acc)
        tt_acc_dis_mat.append(acc_dis)

        if acc > acc_max_b and true_negative>52 and true_positive>52:
            acc_max_b = acc
            torch.save(encoder_expression.state_dict(), os.path.join(save_path, 'encoder_balanced.pkl'))
            torch.save(classifier_expression.state_dict(), os.path.join(save_path, 'classifier_b.pkl'))

        if acc > acc_max :
            acc_max = acc
            torch.save(encoder_expression.state_dict(), os.path.join(save_path, 'encoder.pkl'))
            torch.save(classifier_expression.state_dict(), os.path.join(save_path, 'classifier.pkl'))


        if acc_dis > acc_max_dis:
            acc_max_dis = acc_dis
            torch.save(encoder_discriminator.state_dict(), os.path.join(save_path, 'encoder_dis.pkl'))
            torch.save(classifier_discriminator.state_dict(), os.path.join(save_path, 'classifier_dis.pkl'))
            
            
        print('\n')
        print('the %d-th epoch' % (epoch))
        print('accuracy is : %f' % (acc))
        print('now the best accuracy is %f\n' % (np.max(tt_acc_mat)))


        print('\n')
        print('the %d-th epoch' % (epoch))
        print('accuracy is : %f' % (acc_dis))
        print('now the best accuracy is %f\n' % (np.max(tt_acc_dis_mat)))
         

        dic_log['acc_val'].append(acc)
        dic_log['true_positive'].append(true_positive)
        dic_log['true_negative'].append(true_negative)
        dic_log['F1'].append(f1)
        dic_log['acc_val_dis'].append(acc_dis)
        dic_log['acc_train'].append(true_response_affect/elem_sum*100)
        dic_log['acc_train_dis'].append(true_response_affect_d/elem_sum*100)
        if not save_path is None:
            dataframe = pd.DataFrame(dic_log)
            dataframe.to_csv(save_path+save_log_name)
        




