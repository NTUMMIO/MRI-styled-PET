import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils
from .efficient_attention import EfficientAttention, LinearAttention, HydraAttention

import numpy as np
from torch.autograd import Variable

EPSILON = 1e-5

def efficient_attention(tensor1, tensor2, scale_head):
    # avg, max, nuclear

    _, c, _, _ =tensor1.shape

    k=111
    v=111
    h=111

    tensor1_attention=EfficientAttention(c, k, v, h).cuda()(tensor1)

    tensor2_attention=EfficientAttention(c, k, v, h).cuda()(tensor2)

    tensor_f = (tensor1_attention + tensor2_attention) / 2
    
    return tensor_f


def linear_attention(tensor1, tensor2, scale_head):
    # avg, max, nuclear

    _, c, _, _ =tensor1.shape

    k=c//16
    v=c//16
    h=c//16

    tensor1_attention=LinearAttention(c, k, h*scale_head, v).cuda()(tensor1)

    tensor2_attention=LinearAttention(c, k, h*scale_head, v).cuda()(tensor2)

    tensor_f = (tensor1_attention + tensor2_attention) / 2
    
    return tensor_f

def hydra_attention(tensor1, tensor2, scale_head):
    
    _, c, _, _ =tensor1.shape

    k=c//16
    v=c//16
    h=scale_head

    tensor1_attention=HydraAttention(c, k, h, v).cuda()(tensor1)

    tensor2_attention=HydraAttention(c, k, h*scale_head, v).cuda()(tensor2)

    tensor_f = (tensor1_attention + tensor2_attention) / 2
    
    return tensor_f


# attention fusion strategy, average based on weight maps
def channel_spatial(tensor1, tensor2, p_type='attention_avg'):
    # avg, max, nuclear
    f_channel = channel_fusion(tensor1, tensor2,  p_type)
    f_spatial = spatial_fusion(tensor1, tensor2)

    tensor_f = (f_channel + f_spatial) / 2
    return tensor_f



# select channel
def channel_fusion(tensor1, tensor2, p_type='attention_avg'):
    # global max pooling
    shape = tensor1.size()
    # calculate channel attention
    global_p1 = channel_attention(tensor1, p_type)
    global_p2 = channel_attention(tensor2, p_type)

    # get weight map
    global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
    global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

    tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensor_f


def spatial_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)

    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f


# channel attention
def channel_attention(tensor, pooling_type='avg'):
    # global pooling
    shape = tensor.size()
    pooling_function = F.avg_pool2d

    if pooling_type is 'attention_avg':
        pooling_function = F.avg_pool2d
    elif pooling_type is 'attention_max':
        pooling_function = F.max_pool2d
    elif pooling_type is 'attention_nuclear':
        pooling_function = nuclear_pooling
    global_p = pooling_function(tensor, kernel_size=shape[2:])
    return global_p


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    spatial = []
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial


# pooling function
def nuclear_pooling(tensor, kernel_size=None):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1).cuda()
    for i in range(shape[1]):
        u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0] = s_sum
    return vectors





def attention_fusion_weight(tensor1, tensor2, p_type):

    f_row_vector = row_vector_fusion(tensor1, tensor2, p_type)
    f_column_vector = column_vector_fusion(tensor1, tensor2, p_type)

    tensor_f = (f_row_vector + f_column_vector)

    return tensor_f


def row_vector_fusion(tensor1, tensor2, p_type):
    shape = tensor1.size()
    # calculate row vector attention
    row_vector_p1 = row_vector_attention(tensor1, p_type)
    row_vector_p2 = row_vector_attention(tensor2, p_type)

    # get weight map
    row_vector_p_w1 = torch.exp(row_vector_p1) / (torch.exp(row_vector_p1) + torch.exp(row_vector_p2) + EPSILON)
    row_vector_p_w2 = torch.exp(row_vector_p2) / (torch.exp(row_vector_p1) + torch.exp(row_vector_p2) + EPSILON)

    row_vector_p_w1 = row_vector_p_w1.repeat(1, 1, shape[2], shape[3])
    row_vector_p_w1 = row_vector_p_w1.cuda()
    row_vector_p_w2 = row_vector_p_w2.repeat(1, 1, shape[2], shape[3])
    row_vector_p_w2 = row_vector_p_w2.cuda()

    tensor_f = row_vector_p_w1 * tensor1 + row_vector_p_w2 * tensor2

    return tensor_f


def column_vector_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()
    # calculate column vector attention
    column_vector_1 = column_vector_attention(tensor1, spatial_type)
    column_vector_2 = column_vector_attention(tensor2, spatial_type)

    column_vector_w1 = torch.exp(column_vector_1) / (torch.exp(column_vector_1) + torch.exp(column_vector_2) + EPSILON)
    column_vector_w2 = torch.exp(column_vector_2) / (torch.exp(column_vector_1) + torch.exp(column_vector_2) + EPSILON)

    column_vector_w1 = column_vector_w1.repeat(1, shape[1], 1, 1)
    column_vector_w1 = column_vector_w1.cuda()
    column_vector_w2 = column_vector_w2.repeat(1, shape[1], 1, 1)
    column_vector_w2 = column_vector_w2.cuda()

    tensor_f = column_vector_w1 * tensor1 + column_vector_w2 * tensor2

    return tensor_f


# row vector_attention
def row_vector_attention(tensor, type="l1_mean"):
    shape = tensor.size()

    c = shape[1]
    h = shape[2]
    w = shape[3]
    row_vector = torch.zeros(1, c, 1, 1)
    if type is"l1_mean":
        row_vector = torch.norm(tensor, p=1, dim=[2, 3], keepdim=True) / (h * w)
    elif type is"l2_mean":
        row_vector = torch.norm(tensor, p=2, dim=[2, 3], keepdim=True) / (h * w)
    elif type is "linf":
            for i in range(c):
                tensor_1 = tensor[0,i,:,:]
                row_vector[0,i,0,0] = torch.max(tensor_1)
            ndarray = tensor.cpu().numpy()
            max = np.amax(ndarray,axis=(2,3))
            tensor = torch.from_numpy(max)
            row_vector = tensor.reshape(1,c,1,1)
            row_vector = row_vector.cuda()
    return row_vector


# # column vector attention
def column_vector_attention(tensor, type='l1_mean'):

    shape = tensor.size()
    c = shape[1]
    h = shape[2]
    w = shape[3]
    column_vector = torch.zeros(1, 1, 1, 1)
    if type is 'l1_mean':
        column_vector = torch.norm(tensor, p=1, dim=[1], keepdim=True) / c
    elif type is"l2_mean":
        column_vector = torch.norm(tensor, p=2, dim=[1], keepdim=True) / c
    elif type is "linf":
        column_vector, indices = tensor.max(dim=1, keepdim=True)
        column_vector = column_vector / c
        column_vector = column_vector.cuda()
    return column_vector
