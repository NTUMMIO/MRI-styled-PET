import torch
from torch import nn
from torch.nn import functional as f

import time

class EfficientAttention(nn.Module):
    
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding='same')

    def forward(self, input_): 
        n, _, h, w = input_.size()
        print(input_.size())
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        #print(keys.shape, queries.shape, values.shape, head_key_channels.shape, head_value_channels.shape)
        print(self.key_channels,self.head_count)
        print(keys.shape, queries.shape, values.shape, head_key_channels, head_value_channels)
        
        attended_values = []
        for i in range(self.head_count):
            key = f.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = f.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)
            
            #print(context.shape, attended_value.shape)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        
        out = torch.add(torch.mul(self.conv2(self.conv1(reprojected_value)),input_),input_)
        
        #attention = reprojected_value + input_

        return out#attention

def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))

class LinearAttention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        
        self.l2_norm = l2_norm
        self.eps = eps

        
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding='same')

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.queries(x).view(batch_size, self.key_channels, width * height)
        K = self.keys(x).view(batch_size, self.key_channels, width * height)
        V = self.values(x).view(batch_size, self.value_channels, width * height)#V=[n, c_v, h*w]

        Q = self.l2_norm(Q).permute(-3, -1, -2) #Q transpose=[n, h*w, c_k]
        K = self.l2_norm(K) #K=[n, c_k, h*w]

        denominator = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)) #[n, h*w]
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1) #[n, c_v]
        value_sum = value_sum.expand(-1, self.value_channels, width * height) #[n, c_v, h*w]

        matrix = torch.einsum('bmn, bcn->bmc', K, V) #KtV=[n, c_k, c_v]
        numerator = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)#[n, c_v, h*w]

        weight_value = torch.einsum("bcn, bn->bcn", numerator, denominator)#[n, c_v, h*w]
        weight_value = weight_value.view(batch_size, self.value_channels, height, width)
        
        attention_output = self.reprojection(weight_value)#[n, c_input, h*w]
        
        out = torch.add(torch.mul(self.conv2(self.conv1(attention_output)),x),x)
        
        return out#attention_output
        

class HydraAttention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        
        self.l2_norm = l2_norm
        self.eps = eps

        
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding='same')

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.queries(x).view(batch_size, self.key_channels, width * height)
        K = self.keys(x).view(batch_size, self.key_channels, width * height)
        V = self.values(x).view(batch_size, self.value_channels, width * height)#V=[n, c_v, h*w]
        
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count


        Q = self.l2_norm(Q).permute(-3, -1, -2) #Q transpose=[n, h*w, c_k]
        K = self.l2_norm(K) #K=[n, c_k, h*w]

        #denominator = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)) #[n, h*w]+
        
        agg_dom=0
        for i in range(self.head_count):
            q = Q[:, :, i * head_key_channels: (i + 1) * head_key_channels] #[n, h*w, 1]
            k=torch.sum(K, dim=-1)
            k = k[:, i * head_key_channels: (i + 1) * head_key_channels] #[n, 1]
            dom = torch.einsum("bnc, bc->bn",q,k)#[n, h*w]
            agg_dom+=dom #sum of c_k
        denominator = 1 / (width * height + agg_dom + self.eps)#[n, h*w]
        #print("denominator.shape",denominator.shape)
    
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1) #[n, c_v]
        value_sum = value_sum.expand(-1, self.value_channels, width * height) #[n, c_v, h*w]
        
        agg_kv=0
        for i in range(width * height):
            k = K[:, :, i]#[n, c_k]
            v = V[:, :, i]#[n, c_v]
            #print(k.shape,v.shape)
            kv = torch.einsum('bm, bc->bmc', k, v)#[n, c_k, c_v]
            agg_kv+=kv#sum of n
            
        agg_qkv=0
        for i in range(self.head_count):
            q = Q[:, :, i * head_key_channels: (i + 1) * head_key_channels] #[n, h*w, 1]
            kv = agg_kv[:, i * head_key_channels: (i + 1) * head_key_channels, :] #[n, 1, c_v]
            qkv = torch.einsum("bnm, bmc->bcn",q,kv) #[n, c_v, h*w]
            agg_qkv+=qkv#sum of c_k
        numerator = value_sum + agg_qkv#[n, c_v, h*w]
        
        
        weight_value = torch.einsum("bcn, bn->bcn", numerator, denominator)#[n, c_v, h*w]
        weight_value = weight_value.view(batch_size, self.value_channels, height, width)
        
        attention_output = self.reprojection(weight_value)#[n, c_input, h*w]
        
        out = torch.add(torch.mul(self.conv2(self.conv1(attention_output)),x),x)
        
        '''matrix = torch.einsum('bmn, bcn->bmc', K, V) #KtV=[n, c_k, c_v]
        numerator = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)#[n, c_v, h*w]

        weight_value = torch.einsum("bcn, bn->bcn", numerator, denominator)#[n, c_v, h*w]
        weight_value = weight_value.view(batch_size, self.value_channels, height, width)
        
        attention_output = self.reprojection(weight_value)#[n, c_input, h*w]'''
        
        return out#attention_output
    

class HydraAttention_v2(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        
        self.l2_norm = l2_norm
        self.eps = eps

        
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding='same')

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.queries(x).view(batch_size, self.key_channels, width * height)
        K = self.keys(x).view(batch_size, self.key_channels, width * height)
        V = self.values(x).view(batch_size, self.value_channels, width * height)#V=[n, c_v, h*w]
        
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count


        Q = self.l2_norm(Q).permute(-3, -1, -2) #Q transpose=[n, h*w, c_k]
        K = self.l2_norm(K) #K=[n, c_k, h*w]
        
        denominator = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
    
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1) #[n, c_v]
        value_sum = value_sum.expand(-1, self.value_channels, width * height) #[n, c_v, h*w]
        
        
        
        '''agg_kv=[]
        for j in range(batch_size):
            x1_b=K[j]
            x2_b=V[j]
            agg_m=0
            for i in range(width*height):
                m=x1_b[:,i].unsqueeze(-1) @ x2_b[:,i].unsqueeze(0)
                agg_m+=m.unsqueeze(0)
            if j==0:
                agg_kv=agg_m
            else:
                agg_kv=torch.cat((agg_kv,agg_m),0)'''
        
        agg_kv = torch.einsum('bmn, bcn->bmc', K, V)
        
        
            
        agg_qkv=[]
        for j in range(batch_size):
            x1_b=Q[j]
            x2_b=agg_kv[j]
            agg_m=0
            for i in range(self.head_count):
                if head_key_channels>1:
                    m = x2_b[i * head_key_channels: (i + 1) * head_key_channels,:].transpose(1,0) @ x1_b[:,i * head_key_channels: (i + 1) * head_key_channels].transpose(1,0)
                else:
                    m = x2_b[i,:].unsqueeze(-1) @ x1_b[:,i].unsqueeze(0) 
                agg_m+=m.unsqueeze(0)
            if j==0:
                agg_qkv=agg_m
            else:
                agg_qkv=torch.cat((agg_qkv,agg_m),0)
        
        numerator = value_sum + agg_qkv#[n, c_v, h*w]
        
        
        weight_value = torch.einsum("bcn, bn->bcn", numerator, denominator)#[n, c_v, h*w]
        weight_value = weight_value.view(batch_size, self.value_channels, height, width)
        
        attention_output = self.reprojection(weight_value)#[n, c_input, h*w]
        
        out = torch.add(torch.mul(self.conv2(self.conv1(attention_output)),x),x)

        
        return output
    
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
        
        out = torch.cat((attention_output1, attention_output2),1)
        
        out = self.catconv(out)
        
        #out = torch.add(torch.mul(self.conv2(self.conv1(attention_output)),x),x)
        
        return out#attention_output

if __name__=="__main__":
    #x = torch.Tensor([[[[1, 0.5, 0], [1, 0.5, 0], [1, 0.5, 0]]],[[[1, 0.5, 0], [1, 0.5, 0], [1, 0.5, 0]]]])
    #print(x.shape)

    c=256
    x = torch.rand(16,c,111,111)
    
    '''k=int(c/16)
    v=int(c/16)
    h=int(c/16)'''

    k=c//16
    v=c//16
    h=c//16
    
    '''t = time.process_time()
    print(EfficientAttention(c, k, h, v)(x).shape)
    elapsed_time = time.process_time() - t    
    print(elapsed_time)'''
    
    
    #print(EfficientAttention(1, 111, 111, 111)(x).shape)
    t = time.process_time()
    print(LinearAttention(c, k, h, v)(x).shape)
    elapsed_time = time.process_time() - t    
    print(elapsed_time)
    
    t = time.process_time()
    print(LinearAttention(c, k, h*4, v)(x).shape)
    elapsed_time = time.process_time() - t    
    print(elapsed_time)
    
    t = time.process_time()
    print(HydraAttention(c, k, h, v)(x).shape)
    elapsed_time = time.process_time() - t    
    print(elapsed_time)
    
    t = time.process_time()
    print(HydraAttention(c, k, h*4, v)(x).shape)
    elapsed_time = time.process_time() - t    
    print(elapsed_time)

