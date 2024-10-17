import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
EPSILON = 1e-5

def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class ChannelSpatialAttention(nn.Module):
    def __init__(self):
        super(ChannelSpatialAttention, self).__init__()

    def forward(self, tensor1, tensor2, p_type='attention_avg', spatial_type='mean'):
        f_channel = self.channel_fusion(tensor1, tensor2, p_type)
        f_spatial = self.spatial_fusion(tensor1, tensor2, spatial_type)

        tensor_f = (f_channel + f_spatial) / 2
        return tensor_f

    def channel_fusion(self, tensor1, tensor2, p_type='attention_avg'):
        shape = tensor1.size()
        global_p1 = self.channel_attention(tensor1, p_type)
        global_p2 = self.channel_attention(tensor2, p_type)

        global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
        global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

        global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
        global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

        tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

        return tensor_f

    def spatial_fusion(self, tensor1, tensor2, spatial_type='mean'):
        shape = tensor1.size()
        spatial1 = self.spatial_attention(tensor1, spatial_type)
        spatial2 = self.spatial_attention(tensor2, spatial_type)

        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

        tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

        return tensor_f

    def channel_attention(self, tensor, pooling_type='avg'):
        shape = tensor.size()
        pooling_function = F.avg_pool2d

        if pooling_type == 'attention_avg':
            pooling_function = F.avg_pool2d
        elif pooling_type == 'attention_max':
            pooling_function = F.max_pool2d
        elif pooling_type == 'attention_nuclear':
            pooling_function = self.nuclear_pooling

        global_p = pooling_function(tensor, kernel_size=shape[2:])
        return global_p

    def spatial_attention(self, tensor, spatial_type='sum'):
        spatial = []
        if spatial_type == 'mean':
            spatial = tensor.mean(dim=1, keepdim=True)
        elif spatial_type == 'sum':
            spatial = tensor.sum(dim=1, keepdim=True)
        return spatial

    def nuclear_pooling(self, tensor, kernel_size=None):
        shape = tensor.size()
        vectors = torch.zeros(1, shape[1], 1, 1).cuda()
        for i in range(shape[1]):
            u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
            s_sum = torch.sum(s)
            vectors[0, i, 0, 0] = s_sum
        return vectors

class ChannelSpatialAttention_v2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # self.feature_fusion=nn.Conv2d(in_channels=in_channels,out_channels=in_channels//2,kernel_size=1,bias=False)
        self.feature_fusion_channel=nn.Conv2d(in_channels=in_channels,out_channels=in_channels//2,kernel_size=1,bias=False)
        self.feature_fusion_spatial=nn.Conv2d(in_channels=in_channels,out_channels=in_channels//2,kernel_size=1,bias=False)
        

    def forward(self, tensor1, tensor2, p_type='attention_avg', spatial_type='mean'):
        f_channel = self.channel_fusion(tensor1, tensor2, p_type)
        f_spatial = self.spatial_fusion(tensor1, tensor2, spatial_type)
        # tensor_f = self.feature_fusion(torch.cat([f_channel, f_spatial],1))
        
        tensor_f = (f_channel + f_spatial) / 2
        
        
        return tensor_f

    def channel_fusion(self, tensor1, tensor2, p_type='attention_avg'):
        shape = tensor1.size()
        global_p1 = self.channel_attention(tensor1, p_type)
        global_p2 = self.channel_attention(tensor2, p_type)

        global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
        global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

        global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
        global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])
        
        tensor_f = self.feature_fusion_channel(torch.cat([global_p_w1 * tensor1, global_p_w2 * tensor2],1))

        # tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

        return tensor_f

    def spatial_fusion(self, tensor1, tensor2, spatial_type='mean'):
        shape = tensor1.size()
        spatial1 = self.spatial_attention(tensor1, spatial_type)
        spatial2 = self.spatial_attention(tensor2, spatial_type)

        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
        
        tensor_f = self.feature_fusion_spatial(torch.cat([spatial_w1 * tensor1, spatial_w2 * tensor2],1))
        
        # tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

        return tensor_f

    def channel_attention(self, tensor, pooling_type='avg'):
        shape = tensor.size()
        pooling_function = F.avg_pool2d

        if pooling_type == 'attention_avg':
            pooling_function = F.avg_pool2d
        elif pooling_type == 'attention_max':
            pooling_function = F.max_pool2d
        elif pooling_type == 'attention_nuclear':
            pooling_function = self.nuclear_pooling

        global_p = pooling_function(tensor, kernel_size=shape[2:])
        return global_p

    def spatial_attention(self, tensor, spatial_type='sum'):
        spatial = []
        if spatial_type == 'mean':
            spatial = tensor.mean(dim=1, keepdim=True)
        elif spatial_type == 'sum':
            spatial = tensor.sum(dim=1, keepdim=True)
        return spatial

    def nuclear_pooling(self, tensor, kernel_size=None):
        shape = tensor.size()
        vectors = torch.zeros(1, shape[1], 1, 1).cuda()
        for i in range(shape[1]):
            u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
            s_sum = torch.sum(s)
            vectors[0, i, 0, 0] = s_sum
        return vectors
    
    
    
class SelfAttention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        
        self.l2_norm = l2_norm
        self.eps = eps

        
        self.queries1 = nn.Conv2d(in_channels, key_channels, 1)
        self.keys1 = nn.Conv2d(in_channels, key_channels, 1)
        self.values1 = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection1 = nn.Conv2d(value_channels, in_channels, 1)
        
        self.queries2 = nn.Conv2d(in_channels, key_channels, 1)
        self.keys2 = nn.Conv2d(in_channels, key_channels, 1)
        self.values2 = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection2 = nn.Conv2d(value_channels, in_channels, 1)
        
        self.catconv = nn.Conv2d(2*in_channels, in_channels, 3, padding='same')

    def forward(self, tensor1, tensor2):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = tensor1.shape
        Q1 = self.queries1(tensor1).view(batch_size, self.key_channels, width * height)
        K1 = self.keys1(tensor1).view(batch_size, self.key_channels, width * height)
        V1 = self.values1(tensor1).view(batch_size, self.value_channels, width * height)#V=[n, c_v, h*w]

        Q1 = self.l2_norm(Q1).permute(-3, -1, -2) #Q transpose=[n, h*w, c_k]
        K1 = self.l2_norm(K1) #K=[n, c_k, h*w]

        denominator1 = 1 / (width * height + torch.einsum("bnc, bc->bn", Q1, torch.sum(K1, dim=-1) + self.eps)) #[n, h*w]
        value_sum1 = torch.einsum("bcn->bc", V1).unsqueeze(-1) #[n, c_v]
        value_sum1 = value_sum1.expand(-1, self.value_channels, width * height) #[n, c_v, h*w]

        matrix1 = torch.einsum('bmn, bcn->bmc', K1, V1) #KtV=[n, c_k, c_v]
        numerator1 = value_sum1 + torch.einsum("bnm, bmc->bcn", Q1, matrix1)#[n, c_v, h*w]

        weight_value1= torch.einsum("bcn, bn->bcn", numerator1, denominator1)#[n, c_v, h*w]
        weight_value1 = weight_value1.view(batch_size, self.value_channels, height, width)
        
        attention_output1 = self.reprojection1(weight_value1)#[n, c_input, h*w]
        
        
        
        Q2 = self.queries2(tensor2).view(batch_size, self.key_channels, width * height)
        K2 = self.keys2(tensor2).view(batch_size, self.key_channels, width * height)
        V2 = self.values2(tensor2).view(batch_size, self.value_channels, width * height)#V=[n, c_v, h*w]

        Q2 = self.l2_norm(Q2).permute(-3, -1, -2) #Q transpose=[n, h*w, c_k]
        K2 = self.l2_norm(K2) #K=[n, c_k, h*w]

        denominator2 = 1 / (width * height + torch.einsum("bnc, bc->bn", Q2, torch.sum(K2, dim=-1) + self.eps)) #[n, h*w]
        value_sum2 = torch.einsum("bcn->bc", V2).unsqueeze(-1) #[n, c_v]
        value_sum2 = value_sum2.expand(-1, self.value_channels, width * height) #[n, c_v, h*w]

        matrix2 = torch.einsum('bmn, bcn->bmc', K2, V2) #KtV=[n, c_k, c_v]
        numerator2 = value_sum2 + torch.einsum("bnm, bmc->bcn", Q2, matrix2)#[n, c_v, h*w]

        weight_value2 = torch.einsum("bcn, bn->bcn", numerator2, denominator2)#[n, c_v, h*w]
        weight_value2 = weight_value2.view(batch_size, self.value_channels, height, width)
        
        attention_output2 = self.reprojection2(weight_value2)#[n, c_input, h*w]

        
        out = torch.cat((attention_output1, attention_output2),1)
        
        out = self.catconv(out)
        
        return out#attention_output

class CrossAttention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        
        self.l2_norm = l2_norm
        self.eps = eps

        
        self.queries1 = nn.Conv2d(in_channels, key_channels, 1)
        self.keys1 = nn.Conv2d(in_channels, key_channels, 1)
        self.values1 = nn.Conv2d(in_channels, value_channels, 1)
        
        self.queries2 = nn.Conv2d(in_channels, key_channels, 1)
        self.keys2 = nn.Conv2d(in_channels, key_channels, 1)
        self.values2 = nn.Conv2d(in_channels, value_channels, 1)
        
        self.reprojection1 = nn.Conv2d(value_channels, in_channels, 1)
        self.reprojection2 = nn.Conv2d(value_channels, in_channels, 1)
        
        self.catconv = nn.Conv2d(in_channels*2, in_channels, 1)
        
        #self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding='same')
        #self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding='same')

    def forward(self, tensor1, tensor2):
        # the efficient attention optimized ver.
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = tensor1.shape
        Q1 = self.queries1(tensor1).view(batch_size, self.key_channels, width * height)
        K1 = self.keys1(tensor1).view(batch_size, self.key_channels, width * height)
        V1 = self.values1(tensor1).view(batch_size, self.value_channels, width * height)#V=[n, c_v, h*w]

        Q1 = self.l2_norm(Q1).permute(-3, -1, -2) #Q transpose=[n, h*w, c_k]
        K1 = self.l2_norm(K1) #K=[n, c_k, h*w]

        Q2 = self.queries2(tensor2).view(batch_size, self.key_channels, width * height)
        K2 = self.keys2(tensor2).view(batch_size, self.key_channels, width * height)
        V2 = self.values2(tensor2).view(batch_size, self.value_channels, width * height)#V=[n, c_v, h*w]

        Q2 = self.l2_norm(Q2).permute(-3, -1, -2) #Q transpose=[n, h*w, c_k]
        K2 = self.l2_norm(K2)

        denominator2 = 1 / (width * height + torch.einsum("bnc, bc->bn", Q1, torch.sum(K2, dim=-1) + self.eps)) #[n, h*w]
        value_sum2 = torch.einsum("bcn->bc", V2).unsqueeze(-1) #[n, c_v]
        value_sum2 = value_sum2.expand(-1, self.value_channels, width * height) #[n, c_v, h*w]

        matrix2 = torch.einsum('bmn, bcn->bmc', K2, V2) #KtV=[n, c_k, c_v]
        numerator2 = value_sum2 + torch.einsum("bnm, bmc->bcn", Q2, matrix2)#[n, c_v, h*w]

        weight_value2 = torch.einsum("bcn, bn->bcn", numerator2, denominator2)#[n, c_v, h*w]
        weight_value2 = weight_value2.view(batch_size, self.value_channels, height, width)
        
        attention_output2 = self.reprojection1(weight_value2)#[n, c_input, h*w]
        
        
        denominator1 = 1 / (width * height + torch.einsum("bnc, bc->bn", Q2, torch.sum(K1, dim=-1) + self.eps)) #[n, h*w]
        value_sum1 = torch.einsum("bcn->bc", V1).unsqueeze(-1) #[n, c_v]
        value_sum1 = value_sum1.expand(-1, self.value_channels, width * height) #[n, c_v, h*w]

        matrix1 = torch.einsum('bmn, bcn->bmc', K1, V1) #KtV=[n, c_k, c_v]
        numerator1 = value_sum1 + torch.einsum("bnm, bmc->bcn", Q1, matrix1)#[n, c_v, h*w]

        weight_value1 = torch.einsum("bcn, bn->bcn", numerator1, denominator1)#[n, c_v, h*w]
        weight_value1 = weight_value1.view(batch_size, self.value_channels, height, width)
        
        attention_output1 = self.reprojection2(weight_value1)
        
        # the original attention 
        '''# Attention calculation (original version)
        # Removed the denominator computation and used matrix multiplication directly.
        attention_scores1 = torch.einsum("bmn, bnc->bmc", K1, Q2)  # [n, c_k, c_v]
        attention_scores2 = torch.einsum("bmn, bnc->bmc", K2, Q1)  # [n, c_k, c_v]

        # Apply softmax for attention weights (original version)
        attention_weights1 = torch.nn.functional.softmax(attention_scores1, dim=-1)
        attention_weights2 = torch.nn.functional.softmax(attention_scores2, dim=-1)

        # Context vector calculation (original version)
        context1 = torch.einsum("bmc, bcn->bmn", attention_weights1, V2)  # [n, c_v, h*w]
        context2 = torch.einsum("bmc, bcn->bmn", attention_weights2, V1)  # [n, c_v, h*w]

        # Reshape and apply final projections (same as before)
        weight_value1 = context1.view(batch_size, self.value_channels, height, width)
        attention_output1 = self.reprojection1(weight_value1)

        weight_value2 = context2.view(batch_size, self.value_channels, height, width)
        attention_output2 = self.reprojection2(weight_value2)'''

        
        out = torch.cat((attention_output1, attention_output2),1)
        
        out = self.catconv(out)
        
        #out = torch.add(torch.mul(self.conv2(self.conv1(attention_output)),x),x)
        
        return out#attention_output
    
    
class CrossAttention_v2(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        
        self.l2_norm = l2_norm
        self.eps = eps

        
        self.queries1 = nn.Conv2d(in_channels, key_channels, 1)
        self.keys1 = nn.Conv2d(in_channels, key_channels, 1)
        self.values1 = nn.Conv2d(in_channels, value_channels, 1)
        
        self.queries2 = nn.Conv2d(in_channels, key_channels, 1)
        self.keys2 = nn.Conv2d(in_channels, key_channels, 1)
        self.values2 = nn.Conv2d(in_channels, value_channels, 1)
        
        self.reprojection1 = nn.Conv2d(value_channels, in_channels, 1)
        self.reprojection2 = nn.Conv2d(value_channels, in_channels, 1)
        
        # self.catconv = nn.Conv2d(in_channels*2, in_channels, 1)
        
        #self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding='same')
        #self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding='same')

    def forward(self, tensor1, tensor2):
        # the efficient attention optimized ver.
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = tensor1.shape
        Q1 = self.queries1(tensor1).view(batch_size, self.key_channels, width * height)
        K1 = self.keys1(tensor1).view(batch_size, self.key_channels, width * height)
        V1 = self.values1(tensor1).view(batch_size, self.value_channels, width * height)#V=[n, c_v, h*w]

        Q1 = self.l2_norm(Q1).permute(-3, -1, -2) #Q transpose=[n, h*w, c_k]
        K1 = self.l2_norm(K1) #K=[n, c_k, h*w]

        Q2 = self.queries2(tensor2).view(batch_size, self.key_channels, width * height)
        K2 = self.keys2(tensor2).view(batch_size, self.key_channels, width * height)
        V2 = self.values2(tensor2).view(batch_size, self.value_channels, width * height)#V=[n, c_v, h*w]

        Q2 = self.l2_norm(Q2).permute(-3, -1, -2) #Q transpose=[n, h*w, c_k]
        K2 = self.l2_norm(K2)

        denominator2 = 1 / (width * height + torch.einsum("bnc, bc->bn", Q1, torch.sum(K2, dim=-1) + self.eps)) #[n, h*w]
        value_sum2 = torch.einsum("bcn->bc", V2).unsqueeze(-1) #[n, c_v]
        value_sum2 = value_sum2.expand(-1, self.value_channels, width * height) #[n, c_v, h*w]

        matrix2 = torch.einsum('bmn, bcn->bmc', K2, V2) #KtV=[n, c_k, c_v]
        numerator2 = value_sum2 + torch.einsum("bnm, bmc->bcn", Q2, matrix2)#[n, c_v, h*w]

        weight_value2 = torch.einsum("bcn, bn->bcn", numerator2, denominator2)#[n, c_v, h*w]
        weight_value2 = weight_value2.view(batch_size, self.value_channels, height, width)
        
        attention_output2 = self.reprojection1(weight_value2)#[n, c_input, h*w]
        
        
        denominator1 = 1 / (width * height + torch.einsum("bnc, bc->bn", Q2, torch.sum(K1, dim=-1) + self.eps)) #[n, h*w]
        value_sum1 = torch.einsum("bcn->bc", V1).unsqueeze(-1) #[n, c_v]
        value_sum1 = value_sum1.expand(-1, self.value_channels, width * height) #[n, c_v, h*w]

        matrix1 = torch.einsum('bmn, bcn->bmc', K1, V1) #KtV=[n, c_k, c_v]
        numerator1 = value_sum1 + torch.einsum("bnm, bmc->bcn", Q1, matrix1)#[n, c_v, h*w]

        weight_value1 = torch.einsum("bcn, bn->bcn", numerator1, denominator1)#[n, c_v, h*w]
        weight_value1 = weight_value1.view(batch_size, self.value_channels, height, width)
        
        attention_output1 = self.reprojection2(weight_value1)
        
        # the original attention 
        '''# Attention calculation (original version)
        # Removed the denominator computation and used matrix multiplication directly.
        attention_scores1 = torch.einsum("bmn, bnc->bmc", K1, Q2)  # [n, c_k, c_v]
        attention_scores2 = torch.einsum("bmn, bnc->bmc", K2, Q1)  # [n, c_k, c_v]

        # Apply softmax for attention weights (original version)
        attention_weights1 = torch.nn.functional.softmax(attention_scores1, dim=-1)
        attention_weights2 = torch.nn.functional.softmax(attention_scores2, dim=-1)

        # Context vector calculation (original version)
        context1 = torch.einsum("bmc, bcn->bmn", attention_weights1, V2)  # [n, c_v, h*w]
        context2 = torch.einsum("bmc, bcn->bmn", attention_weights2, V1)  # [n, c_v, h*w]

        # Reshape and apply final projections (same as before)
        weight_value1 = context1.view(batch_size, self.value_channels, height, width)
        attention_output1 = self.reprojection1(weight_value1)

        weight_value2 = context2.view(batch_size, self.value_channels, height, width)
        attention_output2 = self.reprojection2(weight_value2)'''

        
        # out = torch.cat((attention_output1, attention_output2),1)
        
        # out = self.catconv(out)
        
        out=attention_output1+attention_output2
        
        #out = torch.add(torch.mul(self.conv2(self.conv1(attention_output)),x),x)
        
        return out#attention_output