import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/home/linyunong/project/style_transfer/main/utils/model/backbone")
# print(sys.path)
from fusion_module import SelfAttention, CrossAttention, ChannelSpatialAttention, ChannelSpatialAttention_v2, CrossAttention_v2
import matplotlib.pyplot as plt
# from .fusion_module import SelfAttention, CrossAttention, ChannelSpatialAttention


import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torchvision.models as models

class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out


# light version
class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []

        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


# UNetPP network - light, no desnse
class UNetPP(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True, fusion_type=0, scale_head=4, fusion_op=True, classification=False):
        super(UNetPP, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1
        
        self.fusion_type=fusion_type
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()
        # =========================================================
        # encoder_A
        self.conv0_A = ConvLayer(input_nc, output_filter, 1, stride)
        self.EB1_0_A = block(output_filter, nb_filter[0], kernel_size, 1)
        self.EB2_0_A = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.EB3_0_A = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.EB4_0_A = block(nb_filter[2], nb_filter[3], kernel_size, 1)
        
        self.upsample_conv10_A = nn.ConvTranspose2d(nb_filter[1], nb_filter[1], kernel_size=4, stride=2, padding=1)
        self.upsample_conv20_A = nn.ConvTranspose2d(nb_filter[2], nb_filter[2], kernel_size=4, stride=2, padding=1)
        self.upsample_conv30_A = nn.ConvTranspose2d(nb_filter[3], nb_filter[3], kernel_size=4, stride=2, padding=1)
        self.upsample_conv11_A = nn.ConvTranspose2d(nb_filter[1], nb_filter[1], kernel_size=4, stride=2, padding=1)
        self.upsample_conv21_A = nn.ConvTranspose2d(nb_filter[2], nb_filter[2], kernel_size=4, stride=2, padding=1)
        self.upsample_conv12_A = nn.ConvTranspose2d(nb_filter[1], nb_filter[1], kernel_size=4, stride=2, padding=1)
        
        # encoder_B
        self.conv0_B = ConvLayer(input_nc, output_filter, 1, stride)
        self.EB1_0_B = block(output_filter, nb_filter[0], kernel_size, 1)
        self.EB2_0_B = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.EB3_0_B = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.EB4_0_B = block(nb_filter[2], nb_filter[3], kernel_size, 1)

        self.upsample_conv10_B = nn.ConvTranspose2d(nb_filter[1], nb_filter[1], kernel_size=4, stride=2, padding=1)
        self.upsample_conv20_B = nn.ConvTranspose2d(nb_filter[2], nb_filter[2], kernel_size=4, stride=2, padding=1)
        self.upsample_conv30_B = nn.ConvTranspose2d(nb_filter[3], nb_filter[3], kernel_size=4, stride=2, padding=1)
        self.upsample_conv11_B = nn.ConvTranspose2d(nb_filter[1], nb_filter[1], kernel_size=4, stride=2, padding=1)
        self.upsample_conv21_B = nn.ConvTranspose2d(nb_filter[2], nb_filter[2], kernel_size=4, stride=2, padding=1)
        self.upsample_conv12_B = nn.ConvTranspose2d(nb_filter[1], nb_filter[1], kernel_size=4, stride=2, padding=1)
        
        
        self.fusion_op = fusion_op
        self.fusion_type = fusion_type
        self.classification = classification
        # =========================================================
        
        #scale_kv=16
        scale_kv=8
        if fusion_type==0:
            self.fusion0=nn.Conv2d(in_channels=nb_filter[0]*2,out_channels=nb_filter[0],kernel_size=1,bias=False)
            self.fusion1=nn.Conv2d(in_channels=nb_filter[1]*2,out_channels=nb_filter[1],kernel_size=1,bias=False)
            self.fusion2=nn.Conv2d(in_channels=nb_filter[2]*2,out_channels=nb_filter[2],kernel_size=1,bias=False)
            self.fusion3=nn.Conv2d(in_channels=nb_filter[3]*2,out_channels=nb_filter[3],kernel_size=1,bias=False)
        #spatial-channel
        if fusion_type==1:
            self.fusion0=ChannelSpatialAttention()
            self.fusion1=ChannelSpatialAttention()
            self.fusion2=ChannelSpatialAttention()
            self.fusion3=ChannelSpatialAttention()
        #self attention
        if fusion_type==2:
            self.fusion0=SelfAttention(nb_filter[0], nb_filter[0]//scale_kv, nb_filter[0]//(scale_kv*scale_head), nb_filter[0]//scale_kv)
            self.fusion1=SelfAttention(nb_filter[1], nb_filter[1]//scale_kv, nb_filter[1]//(scale_kv*scale_head), nb_filter[1]//scale_kv)
            self.fusion2=SelfAttention(nb_filter[2], nb_filter[2]//scale_kv, nb_filter[2]//(scale_kv*scale_head), nb_filter[2]//scale_kv)
            self.fusion3=SelfAttention(nb_filter[3], nb_filter[3]//scale_kv, nb_filter[3]//(scale_kv*scale_head), nb_filter[3]//scale_kv)
        #cross attention
        elif fusion_type==3:
            self.fusion0=CrossAttention(nb_filter[0], nb_filter[0]//scale_kv, nb_filter[0]//(scale_kv*scale_head), nb_filter[0]//scale_kv)
            self.fusion1=CrossAttention(nb_filter[1], nb_filter[1]//scale_kv, nb_filter[1]//(scale_kv*scale_head), nb_filter[1]//scale_kv)
            self.fusion2=CrossAttention(nb_filter[2], nb_filter[2]//scale_kv, nb_filter[2]//(scale_kv*scale_head), nb_filter[2]//scale_kv)
            self.fusion3=CrossAttention(nb_filter[3], nb_filter[3]//scale_kv, nb_filter[3]//(scale_kv*scale_head), nb_filter[3]//scale_kv)
        #spatial-channel with dimension reduction feature fusion
        elif fusion_type==4:
            self.fusion0=ChannelSpatialAttention_v2(nb_filter[0]*2)
            self.fusion1=ChannelSpatialAttention_v2(nb_filter[1]*2)
            self.fusion2=ChannelSpatialAttention_v2(nb_filter[2]*2)
            self.fusion3=ChannelSpatialAttention_v2(nb_filter[3]*2)
        elif fusion_type==5:
            self.fusion0=CrossAttention_v2(nb_filter[0], nb_filter[0]//scale_kv, nb_filter[0]//(scale_kv*scale_head), nb_filter[0]//scale_kv)
            self.fusion1=CrossAttention_v2(nb_filter[1], nb_filter[1]//scale_kv, nb_filter[1]//(scale_kv*scale_head), nb_filter[1]//scale_kv)
            self.fusion2=CrossAttention_v2(nb_filter[2], nb_filter[2]//scale_kv, nb_filter[2]//(scale_kv*scale_head), nb_filter[2]//scale_kv)
            self.fusion3=CrossAttention_v2(nb_filter[3], nb_filter[3]//scale_kv, nb_filter[3]//(scale_kv*scale_head), nb_filter[3]//scale_kv)
        #=======================================================
        # decoder_A
        self.DB0_1_A = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB1_1_A = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB2_1_A = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)
        self.DB0_2_A = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB1_2_A = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB0_3_A = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)
        # decoder_B
        self.DB0_1_B = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB1_1_B = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB2_1_B = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)
        self.DB0_2_B = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB1_2_B = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB0_3_B = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)
        #=======================================================
        # classification head
        self.class_conv1 = block(nb_filter[3], nb_filter[3]*2, kernel_size, 2)#512, 7, 7
        self.class_conv2 = block(nb_filter[3]*2, nb_filter[3]*4, kernel_size, 2)#1024, 4, 4
        # self.class_conv3 = block(nb_filter[3]*4, nb_filter[3]*8, kernel_size, 2)#2048, 2, 2
        
        # self.fc1 = nn.Linear(nb_filter[3]*8*4, nb_filter[3]*4)#8192->1024
        self.fc1 = nn.Linear(nb_filter[3]*4, nb_filter[3]//2)#1024->128
        self.fc2 = nn.Linear(nb_filter[3]//2, nb_filter[3]//16)#128->16
        self.fc3 = nn.Linear(nb_filter[3]//16, 3)#16->3
        self.relu = nn.ReLU()
        
        
        # ==================================================================
        # new classification head using pretrained ResNet
        pretrained_resnet = models.resnet50(pretrained=True)
        # Freeze the parameters of the pre-trained ResNet
        for param in pretrained_resnet.parameters():
            param.requires_grad = False
    
        self.class_bn1 = pretrained_resnet.bn1#1, 64, 112, 112
        self.class_relu = pretrained_resnet.relu#1, 64, 112, 112
        self.class_maxpool = pretrained_resnet.maxpool#1, 64, 56, 56
        self.class_layer1 = pretrained_resnet.layer1#1, 256, 56, 56
        self.class_layer2 = pretrained_resnet.layer2#1, 512, 28, 28
        self.class_layer3 = pretrained_resnet.layer3#1, 1024, 14, 14
        self.class_layer4 = pretrained_resnet.layer4#1, 1024, 14, 14
        self.class_avgpool = pretrained_resnet.avgpool#1, 2048, 1, 1

        self.class_channel_reduction0 = nn.Conv2d(in_channels=64+nb_filter[0], out_channels=64, kernel_size=1)
        self.class_channel_reduction1 = nn.Conv2d(in_channels=256+nb_filter[1], out_channels=256, kernel_size=1)
        self.class_channel_reduction2 = nn.Conv2d(in_channels=512+nb_filter[2], out_channels=512, kernel_size=1)
        self.class_channel_reduction3 = nn.Conv2d(in_channels=1024+nb_filter[3], out_channels=1024, kernel_size=1)
        self.class_fc1 = nn.Linear(2048, 512)
        self.class_fc2 = nn.Linear(512, 64)
        self.class_fc3 = nn.Linear(64, 3)
        
        self.class_softmax = nn.Softmax(dim=1)




        
        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride, is_last=True)
    def encoder_A(self, input):
        x = self.conv0_A(input)
        x1_0 = self.EB1_0_A(x)
        x2_0 = self.EB2_0_A(self.pool(x1_0))
        x3_0 = self.EB3_0_A(self.pool(x2_0))
        x4_0 = self.EB4_0_A(self.pool(x3_0))
        # x5_0 = self.DB5_0(self.pool(x4_0))
        return [x1_0, x2_0, x3_0, x4_0]
    def encoder_B(self, input):
        x = self.conv0_B(input)
        x1_0 = self.EB1_0_B(x)
        x2_0 = self.EB2_0_B(self.pool(x1_0))
        x3_0 = self.EB3_0_B(self.pool(x2_0))
        x4_0 = self.EB4_0_B(self.pool(x3_0))
        # x5_0 = self.DB5_0(self.pool(x4_0))
        return [x1_0, x2_0, x3_0, x4_0]
    
    def fusion(self, en1, en2):
        if self.fusion_type==0:
            f0 = torch.cat([en1[0], en2[0]],1)
            f0 = self.fusion0(f0)
            f1 = torch.cat([en1[1], en2[1]],1)
            f1 = self.fusion1(f1)
            f2 = torch.cat([en1[2], en2[2]],1)
            f2 = self.fusion2(f2)
            f3 = torch.cat([en1[3], en2[3]],1)
            f3 = self.fusion3(f3)
            # print(f0.shape)
            # print(f1.shape)
            # print(f2.shape)
            # print(f3.shape)
            
        else:
            f0 = self.fusion0(en1[0], en2[0])
            f1 = self.fusion1(en1[1], en2[1])
            f2 = self.fusion2(en1[2], en2[2])
            f3 = self.fusion3(en1[3], en2[3])
            # print(f0.shape)
            # print(f1.shape)
            # print(f2.shape)
            # print(f3.shape)
        return [f0, f1, f2, f3]

    def decoder_train_A(self, f_en):
        #X10 upsample to X01
        # up_f_en1=F.interpolate(f_en[1],size=f_en[0].shape[-2:])#torch.Size([8, 112, 111, 111])
        # print(f_en[1].shape, up_f_en1.shape)
        
        # X20 upsample to X11
        # up_f_en2=F.interpolate(f_en[2],size=f_en[1].shape[-2:])#torch.Size([8, 160, 55, 55])
        # print(f_en[2].shape, up_f_en2.shape)
        
        # X30 upsample to X21
        # up_f_en3=F.interpolate(f_en[3],size=f_en[2].shape[-2:])#torch.Size([8, 208, 27, 27])
        # print(f_en[3].shape, up_f_en3.shape)
        
        #X10 upsample to X01
        up_f_en1=self.upsample_conv10_A(f_en[1])
        
        # X20 upsample to X11
        up_f_en2=self.upsample_conv20_A(f_en[2])
        
        # X30 upsample to X21
        up_f_en3=self.upsample_conv30_A(f_en[3])
        
        #X01
        concat_0_1=torch.cat((f_en[0], up_f_en1), 1)
        x0_1 = self.DB0_1_A(concat_0_1)
        #X11
        concat_1_1=torch.cat([f_en[1], up_f_en2], 1)
        x1_1 = self.DB1_1_A(concat_1_1)
        
        #X11 upsample to X02
        # up_x1_1=F.interpolate(x1_1,size=f_en[0].shape[-2:])
        up_x1_1=self.upsample_conv11_A(x1_1)
        
        #concat X00, X01, X11 upsample(X02)->X02
        concat_0_2=torch.cat([f_en[0], x0_1, up_x1_1], 1)
        #X02
        x0_2 = self.DB0_2_A(concat_0_2)
        
        #concat X20 and X30 upsample->X21
        concat_2_1=torch.cat([f_en[2], up_f_en3], 1)
        #X21
        x2_1 = self.DB2_1_A(concat_2_1)
        #X21 upsample to X12
        # up_x2_1=F.interpolate(x2_1,size=f_en[1].shape[-2:])
        up_x2_1=self.upsample_conv21_A(x2_1)
        
        #concat X10, X11, X21 upsample(X12)->X12
        concat_1_2=torch.cat([f_en[1], x1_1, up_x2_1], 1)
        #X12
        x1_2 = self.DB1_2_A(concat_1_2)
        
        #X12 upsample to X03
        # up_x1_2=F.interpolate(x1_2,size=f_en[0].shape[-2:])
        up_x1_2=self.upsample_conv12_A(x1_2)
        
        #concat X00, X01, X02, X12 upsample ->X03
        concat_0_3=torch.cat([f_en[0], x0_1, x0_2, up_x1_2], 1)
        #X03
        x0_3 = self.DB0_3_A(concat_0_3)
        
        if self.deepsupervision:
            output1 = self.conv1(x0_1)
            output2 = self.conv2(x0_2)
            output3 = self.conv3(x0_3)
            # output4 = self.conv4(x1_4)
            return torch.cat([output1[None,:,:,:,:], output2[None,:,:,:,:], output3[None,:,:,:,:]],0)
        else:
            output = self.conv_out(x0_3)
            return output
    
    def decoder_train_B(self, f_en):
        # up_f_en1=F.interpolate(f_en[1],size=f_en[0].shape[-2:])#torch.Size([8, 112, 111, 111])
        # up_f_en2=F.interpolate(f_en[2],size=f_en[1].shape[-2:])#torch.Size([8, 160, 55, 55])
        # up_f_en3=F.interpolate(f_en[3],size=f_en[2].shape[-2:])#torch.Size([8, 208, 27, 27])
        
        up_f_en1=self.upsample_conv10_B(f_en[1])
        up_f_en2=self.upsample_conv20_B(f_en[2])
        up_f_en3=self.upsample_conv30_B(f_en[3])
        

        concat_0_1=torch.cat((f_en[0], up_f_en1), 1)
        x0_1 = self.DB0_1_B(concat_0_1)
        
        concat_1_1=torch.cat([f_en[1], up_f_en2], 1)
        x1_1 = self.DB1_1_B(concat_1_1)
        
        # up_x1_1=F.interpolate(x1_1,size=f_en[0].shape[-2:])
        up_x1_1=self.upsample_conv11_B(x1_1)
        
        
        concat_0_2=torch.cat([f_en[0], x0_1, up_x1_1], 1)
        x0_2 = self.DB0_2_B(concat_0_2)
        
        concat_2_1=torch.cat([f_en[2], up_f_en3], 1)
        x2_1 = self.DB2_1_B(concat_2_1)
        
        # up_x2_1=F.interpolate(x2_1,size=f_en[1].shape[-2:])
        up_x2_1=self.upsample_conv21_B(x2_1)
        
        concat_1_2=torch.cat([f_en[1], x1_1, up_x2_1], 1)
        x1_2 = self.DB1_2_B(concat_1_2)
        
        # up_x1_2=F.interpolate(x1_2,size=f_en[0].shape[-2:])
        up_x1_2=self.upsample_conv12_B(x1_2)
        
        concat_0_3=torch.cat([f_en[0], x0_1, x0_2, up_x1_2], 1)
        x0_3 = self.DB0_3_B(concat_0_3)
        
        if self.deepsupervision:
            output1 = self.conv1(x0_1)
            output2 = self.conv2(x0_2)
            output3 = self.conv3(x0_3)
            # output4 = self.conv4(x1_4)
            return torch.cat([output1[None,:,:,:,:], output2[None,:,:,:,:], output3[None,:,:,:,:]],0)
        else:
            output = self.conv_out(x0_3)
            return output
    def classification_head(self,x):
        
        # print(f_en[0].shape)#64, 112, 112
        # print(f_en[1].shape)#96, 56, 56
        # print(f_en[2].shape)#128, 28, 28
        # print(f_en[3].shape)#256, 14, 14
        x = self.relu(self.class_conv1(x))#512, 4, 4
        # print(x.shape)
        x = self.relu(self.class_conv2(x))#1024, 1, 1
        # print(x.shape)
        # x = self.relu(self.class_conv3(x))#2048, 2, 2
        # print(x.shape)
        x = x.view(x.size(0), -1)#1024*1*1 #2048*2*2
        # print(x.shape)
        # x = self.relu(self.fc1(x))#8192->1024
        # print(x.shape)
        x = self.relu(self.fc1(x))#1024->128
        # print(x.shape)
        x = self.relu(self.fc2(x))#128->16
        # print(x.shape)
        x = self.fc3(x)#128->16
        # print(x.shape)
        
        return x
    def classification_transfer(self, fusion_features, output, path_feature_maps):
        # print(fusion_features[0].shape, output.shape)

        # x=torch.cat((fusion_features[0], output), dim=1)#1, 64, 112, 112 + 1, 64, 112, 112
        # x=self.class_channel_reduction0(x)#1, 64, 112, 112
        x=fusion_features[0]
        x=self.class_bn1(x) #1, 64, 112, 112
        x=self.class_relu(x) #1, 64, 112, 112
        x=self.class_maxpool(x) #1, 64, 56, 56
        x=self.class_layer1(x) #1, 256, 56, 56
        
        # plt.figure(figsize=(32, 32))
        # for i in range(x.shape[1]):
        #     plt.subplot(16,16,i+1)
        #     plt.imshow(x.cpu().detach().numpy()[0, i])
        # plt.show()
        # plt.savefig(path_feature_maps)
        
        x=torch.cat((fusion_features[1], x), dim=1)#1, 256, 56, 56+ 1, 96, 56, 56
        x=self.class_channel_reduction1(x)#1, 256, 56, 56
        x=self.class_layer2(x) #1, 512, 28, 28
        x=torch.cat((fusion_features[2], x), dim=1)#1, 512, 28, 28 + 1, 128, 28, 28
        x=self.class_channel_reduction2(x)#1, 512, 28, 28
        x=self.class_layer3(x) #1, 1024, 14, 14
        x=torch.cat((fusion_features[3], x), dim=1)#1, 1024, 14, 14 + 1, 256, 14, 14
        x=self.class_channel_reduction3(x)#1, 1024, 14, 14
        x=self.class_layer4(x) #1, 2048, 7, 7
        x=self.class_avgpool(x) #1, 2048, 1, 1
        # print(x.shape)
        x = torch.flatten(x, 1)
        # x=torch.squeeze(x, dim=-1)
        x=self.relu(self.class_fc1(x))
        # print(x.shape)
        x=self.relu(self.class_fc2(x))
        # print(x.shape)
        x=self.class_fc3(x)#2048->512->64->3
        # print(x.shape)
        x=self.class_softmax(x) 
        return x
    
    # def classification_transfer(self, fusion_features, output):

    #     x=torch.cat((output, fusion_features[0]), dim=1)#1, 64, 112, 112 + 1, 32, 112, 112
    #     x=self.class_channel_reduction0(x)#1, 64, 112, 112
    #     x=self.class_pretrained_bn1(x) #1, 64, 112, 112
    #     x=self.class_pretrained_relu(x) #1, 64, 112, 112
    #     x=self.class_pretrained_maxpool(x) #1, 64, 56, 56
    #     x=self.class_pretrained_layer1(x) #1, 256, 56, 56
        
    #     x=torch.cat((x, fusion_features[1]), dim=1)#1, 256, 56, 56+ 1, 128, 56, 56
    #     x=self.class_channel_reduction1(x)#1, 256, 56, 56
    #     x=self.class_pretrained_layer2(x) #1, 512, 28, 28
        
    #     x=torch.cat((x, fusion_features[2]), dim=1)#1, 512, 28, 28 + 1, 256, 28, 28
    #     x=self.class_channel_reduction2(x)#1, 512, 28, 28
    #     x=self.class_pretrained_layer3(x) #1, 1024, 14, 14
        
    #     x=torch.cat((x, fusion_features[3]), dim=1)#1, 1024, 14, 14 + 1, 512, 14, 14
    #     x=self.class_channel_reduction3(x)#1, 1024, 14, 14
    #     x=self.class_pretrained_layer4(x) #1, 2048, 7, 7
    #     x=self.class_pretrained_avgpool(x) #1, 2048, 1, 1
        
    #     x=self.class_fc3(self.class_fc2(self.class_fc1(x)))#2048->512->64->3
    #     return x

    
    def forward(self, x1, x2=torch.empty(1,1)):#, path_feature_maps
        if self.fusion_op:
            # encoder
            x1_en = self.encoder_A(x1)
            x2_en = self.encoder_B(x2)
            # fusion
            f = self.fusion(x1_en, x2_en)
            # decoder
            o = self.decoder_train_A(f)
            
            if self.classification:
                # o_class=self.classification_head(f[3])

                o_class=self.classification_transfer(f, o)#, path_feature_maps
                return o, o_class
            else:
                return o
        else:
            x1_en = self.encoder_A(x1)
            x2_en = self.encoder_B(x1)
            output1 = self.decoder_train_A(x1_en)
            output2 = self.decoder_train_B(x2_en)
            return output1, output2
        

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#3
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class Cross_WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2 , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C), which maps query
            y: input features with shape of (num_windows*B, N, C), which maps key and value
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class Cross_SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1_A = norm_layer(dim)
        self.norm1_B = norm_layer(dim)
        self.attn_A = Cross_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn_B = Cross_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        
        self.drop_path_A = nn.Identity()#DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_B = nn.Identity()#DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_A = norm_layer(dim)
        self.norm2_B = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_A = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_B = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, y, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut_A = x
        shortcut_B = y
        x = self.norm1_A(x)
        y = self.norm1_B(y)
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        y_windows = window_partition(shifted_y, self.window_size)  # nW*B, window_size, window_size, C
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows_A = self.attn_A(x_windows, y_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
            attn_windows_B = self.attn_B(y_windows, x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows_A = self.attn_A(x_windows, y_windows, mask=self.calculate_mask(x_size).to(x.device))
            attn_windows_B = self.attn_B(y_windows, x_windows, mask=self.calculate_mask(x_size).to(y.device))

        # merge windows
        attn_windows_A = attn_windows_A.view(-1, self.window_size, self.window_size, C)
        attn_windows_B = attn_windows_B.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows_A, self.window_size, H, W)  # B H' W' C
        shifted_y = window_reverse(attn_windows_B, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            y = torch.roll(shifted_y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            y = shifted_y
        x = x.view(B, H * W, C)
        y = y.view(B, H * W, C)

        # FFN
        x = shortcut_A + self.drop_path_A(x)
        x = x + self.drop_path_A(self.mlp_A(self.norm2_A(x)))

        y = shortcut_B + self.drop_path_B(y)
        y = y + self.drop_path_B(self.mlp_B(self.norm2_B(y)))
        return x, y

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale**2*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # print(x.shape)
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinUNet(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=112, patch_size=2, in_chans=1, num_classes=1,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", fusion_op=True, fusion_type=0, up_x4=True,**kwargs):
        super().__init__()


        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.patch_size = patch_size
        self.fusion_op = fusion_op
        self.fusion_type = fusion_type
        scale_kv=8
        scale_head=4


        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # ======================================================
        # Build encoder and bottleneck layers
        # ->A
        # ======================================================
        self.layers_A = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers_A.append(layer)
        # ======================================================
        # ->B
        # ======================================================
        self.layers_B = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers_B.append(layer)
        # ======================================================    
        # Fusion
        # ======================================================
        
        self.layers_fusion = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Cross_SwinTransformerBlock(dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                num_heads=num_heads[i_layer], window_size=window_size,
                shift_size=0 if (i_layer % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                #drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],#drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            self.layers_fusion.append(layer)
        
        self.fusion_proj = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.Conv2d(in_channels=int(embed_dim * 2 ** i_layer)*2,out_channels=int(embed_dim * 2 ** i_layer),kernel_size=1,bias=False)
            
            self.fusion_proj.append(layer)
            
        self.layers_fusion_strategy = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if self.fusion_type==1:
                layer=ChannelSpatialAttention()
            elif self.fusion_type==2:
                layer=SelfAttention(int(embed_dim * 2 ** i_layer), int(embed_dim * 2 ** i_layer)//scale_kv, int(embed_dim * 2 ** i_layer)//(scale_kv*scale_head), int(embed_dim * 2 ** i_layer)//scale_kv)
            elif self.fusion_type==3:
                layer=CrossAttention(int(embed_dim * 2 ** i_layer), int(embed_dim * 2 ** i_layer)//scale_kv, int(embed_dim * 2 ** i_layer)//(scale_kv*scale_head), int(embed_dim * 2 ** i_layer)//scale_kv)
                
            self.layers_fusion_strategy.append(layer)

        # ======================================================
        # Build decoder layers
        # ->A
        # ======================================================
        self.layers_up_A = nn.ModuleList()
        self.concat_back_dim_A = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_up_A.append(layer_up)
            self.concat_back_dim_A.append(concat_linear)
        # ======================================================
        # ->B
        # ======================================================
        self.layers_up_B = nn.ModuleList()
        self.concat_back_dim_B = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_up_B.append(layer_up)
            self.concat_back_dim_B.append(concat_linear)
        # ======================================================
        
        self.norm = norm_layer(self.num_features)
        self.norm_up= norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            # print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=patch_size,dim=embed_dim)
            self.output_A = nn.Conv2d(in_channels=embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)
            self.output_B = nn.Conv2d(in_channels=embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)
            


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #Encoder and Bottleneck
    def forward_features_A(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        for layer in self.layers_A:
            x_downsample.append(x)
            x = layer(x)
        x = self.norm(x)  # B L C
        return x, x_downsample
    
    def forward_features_B(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        for layer in self.layers_B:
            x_downsample.append(x)
            x = layer(x)
        x = self.norm(x)  # B L C
        return x, x_downsample
    
    def feature_fusion(self,x_downsample, y_downsample):
        xy_downsample = []
        for i in range(len(self.layers_fusion)):
            layer = self.layers_fusion[i]
            
            
            
            H = int(math.sqrt(x_downsample[i].shape[1]))
            x_size= H,H

                  
            x_cross, y_cross = layer(x_downsample[i], y_downsample[i], x_size)
            
            # print(i,": x_cross, y_cross",x_cross.shape, y_cross.shape)
            x_cross = x_cross.view(x_downsample[i].shape[0],H,H,-1)
            x_cross = x_cross.permute(0,3,1,2)
            y_cross = y_cross.view(x_downsample[i].shape[0],H,H,-1)
            y_cross = y_cross.permute(0,3,1,2)
            # print(i,": x_cross, y_cross",x_cross.shape, y_cross.shape)
            
            if self.fusion_type==0:
                proj = self.fusion_proj[i]
                xy_cross = torch.cat([x_cross, y_cross],1)
                xy_proj = proj(xy_cross)
            else:
                fusion_strategy=self.layers_fusion_strategy[i]
                xy_proj=fusion_strategy(x_cross, y_cross)

            
            
            #print(i, ": xy_proj", xy_proj.shape)
            
            xy_proj = xy_proj.permute(0,2,3,1)
            xy_proj = xy_proj.view(x_downsample[i].shape[0],x_downsample[i].shape[1],x_downsample[i].shape[2])
            xy_downsample.append(xy_proj)
            #print(i, ": xy_proj", xy_proj.shape)
            
        return xy_downsample

    #Dencoder and Skip connection
    def forward_up_features_A(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up_A):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x,x_downsample[self.num_layers-1-inx]],-1)
                x = self.concat_back_dim_A[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)  # B L C
        return x
    
    def forward_up_features_B(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up_B):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x,x_downsample[self.num_layers-1-inx]],-1)
                x = self.concat_back_dim_B[inx](x)
                x = layer_up(x)
        x = self.norm_up(x)  # B L C
        return x

    def up_x4_A(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"
        if self.final_upsample=="expand_first":
            x = self.up(x)
            x = x.view(B,self.patch_size*H,self.patch_size*W,-1)
            x = x.permute(0,3,1,2) #B,C,H,W
            x = self.output_A(x)
        return x
    def up_x4_B(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"
        if self.final_upsample=="expand_first":
            x = self.up(x)
            x = x.view(B,self.patch_size*H,self.patch_size*W,-1)
            x = x.permute(0,3,1,2) #B,C,H,W
            x = self.output_B(x)
        return x

    def forward(self, x, y=torch.empty(1,1)):
        
        if self.fusion_op:
            x, x_downsample = self.forward_features_A(x)
            y, y_downsample = self.forward_features_B(y)
            xy_downsample = self.feature_fusion(x_downsample, y_downsample)
            xy = self.forward_up_features_A(xy_downsample[-1],xy_downsample)
            output = self.up_x4_A(xy)
            return output
        else:    
            x_1, x_downsample_1 = self.forward_features_A(x)
            x_2, x_downsample_2 = self.forward_features_B(x)
            
            x_1 = self.forward_up_features_A(x_1, x_downsample_1)
            x_2 = self.forward_up_features_B(x_2, x_downsample_2)
            
            output1 = self.up_x4_A(x_1)
            output2 = self.up_x4_B(x_2)
            return output1, output2
            

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

if __name__ == "__main__":
    x = torch.randn(1,1,112,112)
    x=x.cuda()
    y = torch.randn(1,1,112,112)
    y=y.cuda()

    img_size=112
    patch_size=2
    in_chans=1
    out_chans=in_chans
    num_classes=1
    embed_dim=96
    depths=[2, 2, 2, 2]
    depths_decoder=[1, 2, 2, 2]
    num_heads=[3, 6, 12, 24]
    window_size=7
    mlp_ratio=4.
    qkv_bias=True
    qk_scale=None
    drop_rate=0.
    attn_drop_rate=0.
    drop_path_rate=0.1
    norm_layer=nn.LayerNorm
    ape=True
    patch_norm=True
    use_checkpoint=False    
    
    
    # swinunet=SwinUNet(img_size=img_size,
    #                     patch_size=patch_size,
    #                     in_chans=in_chans,
    #                     num_classes=out_chans,
    #                     embed_dim=embed_dim,
    #                     depths=depths,
    #                     num_heads=num_heads,
    #                     window_size=window_size,
    #                     mlp_ratio=mlp_ratio,
    #                     qkv_bias=qkv_bias,
    #                     qk_scale=qk_scale,
    #                     drop_rate=drop_rate,
    #                     drop_path_rate=drop_path_rate,
    #                     ape=ape,
    #                     patch_norm=patch_norm,
    #                     use_checkpoint=use_checkpoint, 
    #                     fusion_op=True, 
    #                     fusion_type=3).cuda()
    
    # output=swinunet(x,y)
    
    input_nc = 1
    output_nc = 1
    #nb_filter = [64, 112, 160, 208, 256]
    nb_filter = [64, 96, 128, 256]
    deepsupervision = False
    
    model_x2y = UNetPP(nb_filter, input_nc, output_nc, deepsupervision, fusion_type=3, scale_head=4, fusion_op=True).cuda()
    output=model_x2y(x,y)
    print(output.shape)
    
    model_y2x = UNetPP(nb_filter, input_nc, output_nc, deepsupervision, fusion_type=0, scale_head=4, fusion_op=False).cuda()
    output1, output2=model_y2x(x)
    print(output1.shape, output2.shape)
    