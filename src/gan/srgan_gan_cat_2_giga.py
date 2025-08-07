import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

class Discriminator(nn.Module):
    def __init__(self, in_shape, in_channels, layer_num):
        super(Discriminator, self).__init__()

        self.in_shape = in_shape
        self.layer_num = layer_num
        self.in_channels = in_channels

        in_height, in_width = self.in_shape
        patch_h, patch_w = int(in_height / 2 ** 1), int(in_width / 2 ** 1)
        self.output_shape_1 = (1, patch_h, patch_w)
        patch_h, patch_w = int(in_height / 2 ** 2), int(in_width / 2 ** 2)
        self.output_shape_2 = (1, patch_h, patch_w)
        patch_h, patch_w = int(in_height / 2 ** 3), int(in_width / 2 ** 3)
        self.output_shape_3 = (1, patch_h, patch_w)

        filters = [in_channels, 32, 64, 128]
        self.con_layer11 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.con_layer1_attn = Self_Attn(filters[1], 'relu')
        self.con_layer21 =  nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.con_layer2_attn = Self_Attn(filters[2], 'relu')
        self.con_layer31 = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters[3], filters[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.out_conv_layer = []
        self.out_conv_layer11 = nn.Conv2d(filters[1], filters[1]//2, kernel_size=3, stride=1, padding=1)
        self.out_conv_layer12 = nn.Conv2d(filters[1]//2, 1, kernel_size=3, stride=1, padding=1)
        self.out_conv_layer21 = nn.Conv2d(filters[2], filters[2]//2, kernel_size=3, stride=1, padding=1)
        self.out_conv_layer22 = nn.Conv2d(filters[2]//2, 1, kernel_size=3, stride=1, padding=1)
        self.out_conv_layer31 = nn.Conv2d(filters[3], filters[3]//2, kernel_size=3, stride=1, padding=1)
        self.out_conv_layer32 = nn.Conv2d(filters[3]//2, 1, kernel_size=3, stride=1, padding=1)
        
        self.out_cat_layer = []
        self.out_cat_layer1 = nn.Sequential(
            nn.Conv2d(filters[1]+filters[0], filters[1], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.out_cat_layer2 = nn.Sequential(
            nn.Conv2d(filters[0]+filters[2], filters[2], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.conv_combine = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x = []
        for i in range(len(x1)):
            xi = self.conv_combine(torch.cat((x1[i],x2[i]),dim=1))    
            x.append(xi)   
        
        out1 = []
        out2 = []
        out3 = []

        con_1 = self.con_layer11(x[0])
        sa_1 = self.con_layer1_attn(con_1)
        out_1 = self.out_conv_layer12(self.out_conv_layer11(sa_1))
        out1.append(out_1)

        con_2 = self.con_layer21(self.out_cat_layer1(torch.cat((sa_1,x[1]),dim=1)))
        sa_2 = self.con_layer2_attn(con_2)
        out_2 = self.out_conv_layer22(self.out_conv_layer21(sa_2))
        out1.append(out_2)
        out2.append(out_2)

        con_3 = self.con_layer31(self.out_cat_layer2(torch.cat((sa_2,x[2]),dim=1)))
        out_3 = self.out_conv_layer32(self.out_conv_layer31(con_3))
        out1.append(out_3)
        out2.append(out_3)
        out3.append(out_3)
        # print("ok")
        # print(out1[0].shape)
        # print(out1[1].shape)
        # print(out1[2].shape)
        # print(out2[0].shape)
        # print(out2[1].shape)
        # print(out3[0].shape)
        # print("ok")
        # ok
        # torch.Size([1, 1, 128, 128])
        # torch.Size([1, 1, 64, 64])
        # torch.Size([1, 1, 32, 32])
        # torch.Size([1, 1, 64, 64])
        # torch.Size([1, 1, 32, 32])
        # torch.Size([1, 1, 32, 32])
        # ok
        return out1, out2, out3