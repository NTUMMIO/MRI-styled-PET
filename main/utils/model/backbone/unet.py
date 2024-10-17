import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion.fusion_strategy import channel_spatial

from .fusion.efficient_attention import EfficientAttention, LinearAttention, HydraAttention, HydraAttention_v2, CrossAttention

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


# NestFuse network - light, no desnse
class NestFuse(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True, fusion_type=0, scale_head=4, fusion_op=True):
        super(NestFuse, self).__init__()
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
        self.DB1_0_A = block(output_filter, nb_filter[0], kernel_size, 1)
        self.DB2_0_A = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.DB3_0_A = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.DB4_0_A = block(nb_filter[2], nb_filter[3], kernel_size, 1)
        # encoder_B
        self.conv0_B = ConvLayer(input_nc, output_filter, 1, stride)
        self.DB1_0_B = block(output_filter, nb_filter[0], kernel_size, 1)
        self.DB2_0_B = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.DB3_0_B = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.DB4_0_B = block(nb_filter[2], nb_filter[3], kernel_size, 1)
        self.fusion_op = fusion_op
        # =========================================================
        
        #scale_kv=16
        scale_kv=8

        if fusion_type==1:
            self.fusion0_e1=EfficientAttention(nb_filter[0], nb_filter[0]//scale_kv, nb_filter[0]//(scale_kv*scale_head), nb_filter[0]//scale_kv)
            self.fusion1_e1=EfficientAttention(nb_filter[1], nb_filter[1]//scale_kv, nb_filter[1]//(scale_kv*scale_head), nb_filter[1]//scale_kv)
            self.fusion2_e1=EfficientAttention(nb_filter[2], nb_filter[2]//scale_kv, nb_filter[2]//(scale_kv*scale_head), nb_filter[2]//scale_kv)
            self.fusion3_e1=EfficientAttention(nb_filter[3], nb_filter[3]//scale_kv, nb_filter[3]//(scale_kv*scale_head), nb_filter[3]//scale_kv)

            self.fusion0_e2=EfficientAttention(nb_filter[0], nb_filter[0]//scale_kv, nb_filter[0]//(scale_kv*scale_head), nb_filter[0]//scale_kv)
            self.fusion1_e2=EfficientAttention(nb_filter[1], nb_filter[1]//scale_kv, nb_filter[1]//(scale_kv*scale_head), nb_filter[1]//scale_kv)
            self.fusion2_e2=EfficientAttention(nb_filter[2], nb_filter[2]//scale_kv, nb_filter[2]//(scale_kv*scale_head), nb_filter[2]//scale_kv)
            self.fusion3_e2=EfficientAttention(nb_filter[3], nb_filter[3]//scale_kv, nb_filter[3]//(scale_kv*scale_head), nb_filter[3]//scale_kv)

        elif fusion_type==2:
            self.fusion0_e1=LinearAttention(nb_filter[0], nb_filter[0]//scale_kv, nb_filter[0]//(scale_kv*scale_head), nb_filter[0]//scale_kv)
            self.fusion1_e1=LinearAttention(nb_filter[1], nb_filter[1]//scale_kv, nb_filter[1]//(scale_kv*scale_head), nb_filter[1]//scale_kv)
            self.fusion2_e1=LinearAttention(nb_filter[2], nb_filter[2]//scale_kv, nb_filter[2]//(scale_kv*scale_head), nb_filter[2]//scale_kv)
            self.fusion3_e1=LinearAttention(nb_filter[3], nb_filter[3]//scale_kv, nb_filter[3]//(scale_kv*scale_head), nb_filter[3]//scale_kv)

            self.fusion0_e2=LinearAttention(nb_filter[0], nb_filter[0]//scale_kv, nb_filter[0]//(scale_kv*scale_head), nb_filter[0]//scale_kv)
            self.fusion1_e2=LinearAttention(nb_filter[1], nb_filter[1]//scale_kv, nb_filter[1]//(scale_kv*scale_head), nb_filter[1]//scale_kv)
            self.fusion2_e2=LinearAttention(nb_filter[2], nb_filter[2]//scale_kv, nb_filter[2]//(scale_kv*scale_head), nb_filter[2]//scale_kv)
            self.fusion3_e2=LinearAttention(nb_filter[3], nb_filter[3]//scale_kv, nb_filter[3]//(scale_kv*scale_head), nb_filter[3]//scale_kv)

        elif fusion_type==3:
            self.fusion0_e1=HydraAttention(nb_filter[0], nb_filter[0]//scale_kv, nb_filter[0]//(scale_kv*scale_head), nb_filter[0]//scale_kv)
            self.fusion1_e1=HydraAttention(nb_filter[1], nb_filter[1]//scale_kv, nb_filter[1]//(scale_kv*scale_head), nb_filter[1]//scale_kv)
            self.fusion2_e1=HydraAttention(nb_filter[2], nb_filter[2]//scale_kv, nb_filter[2]//(scale_kv*scale_head), nb_filter[2]//scale_kv)
            self.fusion3_e1=HydraAttention(nb_filter[3], nb_filter[3]//scale_kv, nb_filter[3]//(scale_kv*scale_head), nb_filter[3]//scale_kv)

            self.fusion0_e2=HydraAttention(nb_filter[0], nb_filter[0]//scale_kv, nb_filter[0]//(scale_kv*scale_head), nb_filter[0]//scale_kv)
            self.fusion1_e2=HydraAttention(nb_filter[1], nb_filter[1]//scale_kv, nb_filter[1]//(scale_kv*scale_head), nb_filter[1]//scale_kv)
            self.fusion2_e2=HydraAttention(nb_filter[2], nb_filter[2]//scale_kv, nb_filter[2]//(scale_kv*scale_head), nb_filter[2]//scale_kv)
            self.fusion3_e2=HydraAttention(nb_filter[3], nb_filter[3]//scale_kv, nb_filter[3]//(scale_kv*scale_head), nb_filter[3]//scale_kv)
            
        elif fusion_type==4:

            self.fusion0_e1=HydraAttention_v2(nb_filter[0], nb_filter[0]//scale_kv, nb_filter[0]//(scale_kv*scale_head), nb_filter[0]//scale_kv)
            self.fusion1_e1=HydraAttention_v2(nb_filter[1], nb_filter[1]//scale_kv, nb_filter[1]//(scale_kv*scale_head), nb_filter[1]//scale_kv)
            self.fusion2_e1=HydraAttention_v2(nb_filter[2], nb_filter[2]//scale_kv, nb_filter[2]//(scale_kv*scale_head), nb_filter[2]//scale_kv)
            self.fusion3_e1=HydraAttention_v2(nb_filter[3], nb_filter[3]//scale_kv, nb_filter[3]//(scale_kv*scale_head), nb_filter[3]//scale_kv)

            self.fusion0_e2=HydraAttention_v2(nb_filter[0], nb_filter[0]//scale_kv, nb_filter[0]//(scale_kv*scale_head), nb_filter[0]//scale_kv)
            self.fusion1_e2=HydraAttention_v2(nb_filter[1], nb_filter[1]//scale_kv, nb_filter[1]//(scale_kv*scale_head), nb_filter[1]//scale_kv)
            self.fusion2_e2=HydraAttention_v2(nb_filter[2], nb_filter[2]//scale_kv, nb_filter[2]//(scale_kv*scale_head), nb_filter[2]//scale_kv)
            self.fusion3_e2=HydraAttention_v2(nb_filter[3], nb_filter[3]//scale_kv, nb_filter[3]//(scale_kv*scale_head), nb_filter[3]//scale_kv)
            
        elif fusion_type==5:
            self.fusion0=CrossAttention(nb_filter[0], nb_filter[0]//scale_kv, nb_filter[0]//(scale_kv*scale_head), nb_filter[0]//scale_kv)
            self.fusion1=CrossAttention(nb_filter[1], nb_filter[1]//scale_kv, nb_filter[1]//(scale_kv*scale_head), nb_filter[1]//scale_kv)
            self.fusion2=CrossAttention(nb_filter[2], nb_filter[2]//scale_kv, nb_filter[2]//(scale_kv*scale_head), nb_filter[2]//scale_kv)
            self.fusion3=CrossAttention(nb_filter[3], nb_filter[3]//scale_kv, nb_filter[3]//(scale_kv*scale_head), nb_filter[3]//scale_kv)

        #=======================================================
        # decoder_A
        self.DB1_1_A = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1_A = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1_A = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)
        self.DB1_2_A = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2_A = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB1_3_A = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)
        # decoder_B
        self.DB1_1_B = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1_B = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1_B = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)
        self.DB1_2_B = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2_B = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB1_3_B = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)
        #=======================================================
        
        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride, is_last=True)
    def encoder_A(self, input):
        x = self.conv0_A(input)
        x1_0 = self.DB1_0_A(x)
        x2_0 = self.DB2_0_A(self.pool(x1_0))
        x3_0 = self.DB3_0_A(self.pool(x2_0))
        x4_0 = self.DB4_0_A(self.pool(x3_0))
        # x5_0 = self.DB5_0(self.pool(x4_0))
        return [x1_0, x2_0, x3_0, x4_0]
    def encoder_B(self, input):
        x = self.conv0_B(input)
        x1_0 = self.DB1_0_B(x)
        x2_0 = self.DB2_0_B(self.pool(x1_0))
        x3_0 = self.DB3_0_B(self.pool(x2_0))
        x4_0 = self.DB4_0_B(self.pool(x3_0))
        # x5_0 = self.DB5_0(self.pool(x4_0))
        return [x1_0, x2_0, x3_0, x4_0]
    def fusion(self, en1, en2):
        if self.fusion_type==0:
            self.fusion_function = channel_spatial

            f0 = self.fusion_function(en1[0], en2[0])
            f1 = self.fusion_function(en1[1], en2[1])
            f2 = self.fusion_function(en1[2], en2[2])
            f3 = self.fusion_function(en1[3], en2[3])
            
        elif self.fusion_type==5:
            f0 = self.fusion0(en1[0], en2[0])
            f1 = self.fusion1(en1[1], en2[1])
            f2 = self.fusion2(en1[2], en2[2])
            f3 = self.fusion3(en1[3], en2[3])
        else:
            f0=( self.fusion0_e1(en1[0])+self.fusion0_e2(en2[0]) ) / 2
            f1=( self.fusion1_e1(en1[1])+self.fusion1_e2(en2[1]) ) / 2
            f2=( self.fusion2_e1(en1[2])+self.fusion2_e2(en2[2]) ) / 2
            f3=( self.fusion3_e1(en1[3])+self.fusion3_e2(en2[3]) ) / 2

        return [f0, f1, f2, f3]

    def decoder_train_A(self, f_en):
        up_f_en1=F.interpolate(f_en[1],size=f_en[0].shape[-2:])#torch.Size([8, 112, 111, 111])
        up_f_en2=F.interpolate(f_en[2],size=f_en[1].shape[-2:])#torch.Size([8, 160, 55, 55])
        up_f_en3=F.interpolate(f_en[3],size=f_en[2].shape[-2:])#torch.Size([8, 208, 27, 27])

        concat_1_1=torch.cat((f_en[0], up_f_en1), 1)
        x1_1 = self.DB1_1_A(concat_1_1)
        
        concat_2_1=torch.cat([f_en[1], up_f_en2], 1)
        x2_1 = self.DB2_1_A(concat_2_1)
        
        up_x2_1=F.interpolate(x2_1,size=f_en[0].shape[-2:])
        
        concat_1_2=torch.cat([f_en[0], x1_1, up_x2_1], 1)
        x1_2 = self.DB1_2_A(concat_1_2)
        
        concat_3_1=torch.cat([f_en[2], up_f_en3], 1)
        x3_1 = self.DB3_1_A(concat_3_1)
        
        up_x3_1=F.interpolate(x3_1,size=f_en[1].shape[-2:])
        concat_2_2=torch.cat([f_en[1], x2_1, up_x3_1], 1)
        x2_2 = self.DB2_2_A(concat_2_2)
        
        up_x2_2=F.interpolate(x2_2,size=f_en[0].shape[-2:])
        concat_1_3=torch.cat([f_en[0], x1_1, x1_2, up_x2_2], 1)
        x1_3 = self.DB1_3_A(concat_1_3)
        
        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            # output4 = self.conv4(x1_4)
            return torch.cat([output1[None,:,:,:,:], output2[None,:,:,:,:], output3[None,:,:,:,:]],0)
        else:
            output = self.conv_out(x1_3)
            return output
    
    def decoder_train_B(self, f_en):
        
        #X10 upsample to X01
        up_f_en1=F.interpolate(f_en[1],size=f_en[0].shape[-2:])#torch.Size([8, 112, 111, 111])
        #X20 upsample to X11
        up_f_en2=F.interpolate(f_en[2],size=f_en[1].shape[-2:])#torch.Size([8, 160, 55, 55])
        #X40 upsample to X21
        up_f_en3=F.interpolate(f_en[3],size=f_en[2].shape[-2:])#torch.Size([8, 208, 27, 27])

        concat_1_1=torch.cat((f_en[0], up_f_en1), 1)
        x1_1 = self.DB1_1_B(concat_1_1)
        
        concat_2_1=torch.cat([f_en[1], up_f_en2], 1)
        x2_1 = self.DB2_1_B(concat_2_1)
        
        up_x2_1=F.interpolate(x2_1,size=f_en[0].shape[-2:])
        
        concat_1_2=torch.cat([f_en[0], x1_1, up_x2_1], 1)
        x1_2 = self.DB1_2_B(concat_1_2)
        
        concat_3_1=torch.cat([f_en[2], up_f_en3], 1)
        x3_1 = self.DB3_1_B(concat_3_1)
        
        up_x3_1=F.interpolate(x3_1,size=f_en[1].shape[-2:])
        concat_2_2=torch.cat([f_en[1], x2_1, up_x3_1], 1)
        x2_2 = self.DB2_2_B(concat_2_2)
        
        up_x2_2=F.interpolate(x2_2,size=f_en[0].shape[-2:])
        concat_1_3=torch.cat([f_en[0], x1_1, x1_2, up_x2_2], 1)
        x1_3 = self.DB1_3_B(concat_1_3)
        
        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            # output4 = self.conv4(x1_4)
            return torch.cat([output1[None,:,:,:,:], output2[None,:,:,:,:], output3[None,:,:,:,:]],0)
        else:
            output = self.conv_out(x1_3)
            return output

    # def decoder_train(self, f_en):


    #     up_f_en1=F.interpolate(f_en[1],size=f_en[0].shape[-2:])#torch.Size([8, 112, 111, 111])
    #     up_f_en2=F.interpolate(f_en[2],size=f_en[1].shape[-2:])#torch.Size([8, 160, 55, 55])
    #     up_f_en3=F.interpolate(f_en[3],size=f_en[2].shape[-2:])#torch.Size([8, 208, 27, 27])
        
        
    #     concat_1_1=torch.cat((f_en[0], up_f_en1), 1)
    #     x1_1 = self.DB1_1(concat_1_1)#scale_factor=2,
        
    #     concat_2_1=torch.cat([f_en[1], up_f_en2], 1)
    #     x2_1 = self.DB2_1(concat_2_1)
        
       
    #     up_x2_1=F.interpolate(x2_1,size=f_en[0].shape[-2:])
        
    #     concat_1_2=torch.cat([f_en[0], x1_1, up_x2_1], 1)
    #     x1_2 = self.DB1_2(concat_1_2)
        
    #     concat_3_1=torch.cat([f_en[2], up_f_en3], 1)
    #     x3_1 = self.DB3_1(concat_3_1)
        
        
    #     up_x3_1=F.interpolate(x3_1,size=f_en[1].shape[-2:])
    #     concat_2_2=torch.cat([f_en[1], x2_1, up_x3_1], 1)
    #     x2_2 = self.DB2_2(concat_2_2)
        
    #     up_x2_2=F.interpolate(x2_2,size=f_en[0].shape[-2:])
    #     concat_1_3=torch.cat([f_en[0], x1_1, x1_2, up_x2_2], 1)
    #     x1_3 = self.DB1_3(concat_1_3)
        
    #     '''print("up_f_en1.shape",up_f_en1.shape)
    #     print("up_f_en2.shape",up_f_en2.shape)
    #     print("up_f_en3.shape",up_f_en3.shape)
    #     #torch.Size([8, 112, 111, 111])
    #     #torch.Size([8, 160, 55, 55])
    #     #torch.Size([8, 208, 27, 27])
    #     print("concat_1_1.shape",concat_1_1.shape)
    #     print("x1_1.shape",x1_1.shape)
    #     print("concat_2_1.shape",concat_2_1.shape)
    #     print("x2_1.shape",x2_1.shape)
    #     print("up_x2_1.shape",up_x2_1.shape)
    #     print("concat_1_2.shape",concat_1_2.shape)
    #     print(x1_2.shape)
    #     print("concat_3_1.shape",concat_3_1.shape)
    #     print("x3_1.shape",x3_1.shape)
    #     print(up_x3_1.shape)
    #     print(concat_2_2.shape)
    #     print(x2_2.shape)
    #     print(up_x2_2.shape)
    #     print(concat_1_3.shape)
    #     print(x1_3.shape)'''
        
        
    #     if self.deepsupervision:
    #         output1 = self.conv1(x1_1)
    #         output2 = self.conv2(x1_2)
    #         output3 = self.conv3(x1_3)
    #         # output4 = self.conv4(x1_4)
    #         return torch.cat([output1[None,:,:,:,:], output2[None,:,:,:,:], output3[None,:,:,:,:]],0)
    #     else:
    #         output = self.conv_out(x1_3)
    #         return output

    # def decoder_eval(self, f_en):

    #     x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))

    #     x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))
    #     x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))

    #     x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))
    #     x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], 1))

    #     x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], 1))

    #     if self.deepsupervision:
    #         output1 = self.conv1(x1_1)
    #         output2 = self.conv2(x1_2)
    #         output3 = self.conv3(x1_3)
    #         return torch.cat([output1[None,:,:,:,:], output2[None,:,:,:,:], output3[None,:,:,:,:]],0)
    #     else:
    #         output = self.conv_out(x1_3)
    #         return output
        
    def forward(self, x1, x2=torch.empty(1,1)):
        if self.fusion_op:
            # encoder
            x1_en = self.encoder_A(x1)
            x2_en = self.encoder_B(x2)
            # fusion
            f = self.fusion(x1_en, x2_en)
            # decoder
            o = self.decoder_train_A(f)
            return o
        else:
            x1_en = self.encoder_A(x1)
            x2_en = self.encoder_B(x1)
            output1 = self.decoder_train_A(x1_en)
            output2 = self.decoder_train_B(x2_en)
            return output1, output2
        