import torch
from torch import nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torchvision.models import resnet50, ResNet50_Weights,resnet18, ResNet18_Weights


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Classif(torch.nn.Module):
    def __init__(self,nb_class,softmax=True):
        super(Classif,self).__init__()

        self.fc1 = torch.nn.Linear(1000,100)
        self.fc2 = torch.nn.Linear(100,nb_class)
        self.softmax = softmax


    def forward(self,x):
        x = self.fc1(x).relu()
        if self.softmax:
            x = self.fc2(x).softmax(dim=-1)
        else:
            x = self.fc2(x).sigmoid().reshape(-1)
        return x




class resblock(nn.Module):
    '''
    residual block
    '''
    def __init__(self, n_chan):
        super(resblock, self).__init__()
        self.infer = nn.Sequential(*[
            nn.Conv2d(n_chan, n_chan, 3, 1, 1),
            nn.ReLU()
        ])

    def forward(self, x_in):
        self.res_out = x_in + self.infer(x_in)
        return self.res_out



class decoder(nn.Module):
    def __init__(self, Nz=100, Nb=3, Nc=128, GRAY=False):

        super(decoder, self).__init__()

        self.Nz = Nz

        # embedding layer
        self.emb1 = nn.Sequential(*[
            nn.Conv2d(512*2 + Nz, Nc, 3, 1, 1),
            nn.ReLU(),
        ])
        self.emb2 = self._make_layer(resblock, Nb, Nc)

        # decoding layers
        self.us1 = nn.Sequential(*[
            nn.ConvTranspose2d(Nc, 512, 10, 2, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),
        ])
        self.us2 = nn.Sequential(*[
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        ])
        self.us3 = nn.Sequential(*[
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
        ])
        self.us4 = nn.Sequential(*[
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        ])
        self.us5 = nn.Sequential(*[
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
        ])
        if GRAY:
            self.us6 = nn.Sequential(*[
                nn.ConvTranspose2d(32, 1, 3, 1, 1, bias=False),
                nn.Sigmoid()
            ])
        else:
            self.us6 = nn.Sequential(*[
                nn.ConvTranspose2d(32, 3, 3, 1, 1, bias=False),
                nn.Sigmoid()
            ])

    def _make_layer(self, block, num_blocks, n_chan):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(n_chan))
        return nn.Sequential(*layers)

    def forward(self, enc_FR, enc_ER, noise=None, device=None):
        # features of the branch
        fea_ER = enc_ER
        fea_FR = enc_FR

        # concatenate the inputs with noises
        if noise is not None:
            noise = noise
        else:
            noise = Variable(torch.rand(fea_ER.shape[0], self.Nz, 1, 1))

        if device is not None:
            noise = noise.to(device)

        if self.Nz == 0:
            emb_in = torch.cat((fea_ER, fea_FR), dim=1)
        else:
            emb_in = torch.cat((fea_ER, fea_FR, noise), dim=1)
        # embedding: bsx(256+Nz)x8x8 -> bsxNcx8x8
        self.emb1_out = self.emb1(emb_in)
        # bsxNcx8x8 -> bsxNcx8x8
        self.emb2_out = self.emb2(self.emb1_out)

        # decoding:
        # bsxNcx8x8 -> bsx512x16x16
        self.us1_out = self.us1(self.emb2_out)
        # bsx512x16x16 -> bsx256x32x32
        self.us2_out = self.us2(self.us1_out)
        # bsx256x32x32 -> bsx128x64x64
        self.us3_out = self.us3(self.us2_out)
        # bsx128x64x64 -> bsx64x128x128
        self.us4_out = self.us4(self.us3_out)
        # bsx128x64x64 -> bsx64x128x128
        self.us5_out = self.us5(self.us4_out)
        # bsx64x128x128 -> bsxout_chanx128x128
        self.img = self.us6(self.us5_out)

        return self.img
