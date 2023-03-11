import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  #
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  #

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  #
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)  #
    num_points = x.size(2)  #
    x = x.view(batch_size, -1, num_points)  #
    if idx is None:
        idx = knn(x, k=k)  #
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  #

    idx = idx + idx_base  #

    idx = idx.view(-1)  #

    _, num_dims, _ = x.size()  #

    x = x.transpose(2,
                    1).contiguous()  #
    feature = x.view(batch_size * num_points, -1)  #
    feature = feature[idx, :]  #
    feature = feature.view(batch_size, num_points, k, num_dims)  #
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  #
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # [

    return feature

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0,2,1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist

def calc_distances(p0, points):
    p0=p0.unsqueeze(0)
    p0=p0.repeat(points.shape[0],1)

    dis=((p0 - points)**2).sum(axis=-1)
    return dis
def farthest_sampler(pts, k,node_xyz):
    for j in range(pts.shape[0]):# j in bach_size
        ran_idx = np.random.randint(len(pts))
        node_xyz[j,0,:] = pts[j,ran_idx,:]
        distances = calc_distances(node_xyz[j,0,:-1], pts[j,:,:-1])
        for i in range(1, k):
            idx=torch.argmax(distances,dim=-1).item()
            node_xyz[j,i,:] = pts[j,idx,:]
            distances2=calc_distances(node_xyz[j,i,:-1], pts[j,:,:-1])
            distances = torch.min(distances,distances2)
    return node_xyz

def index_points(points, idx):

    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1]*(len(view_shape)-1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # [B, S]
    new_points = points[batch_indices, idx, :]
    return new_points

def query_knn_point(k, query, pc):

    query = query.permute(0,2,1).unsqueeze(3)
    database = pc.permute(0,2,1).unsqueeze(2)
    dif=query - database

    norm = torch.norm(dif, dim=1, keepdim=False)

    knn_d, knn_ids = torch.topk(norm, k=k, dim=2, largest=False, sorted=True)
    knn_points = index_points(pc, knn_ids)
    centroids = torch.mean(knn_points, dim=2)
    centroids = centroids.unsqueeze(2).repeat(1,1,k,1)
    normed_knn_points = knn_points - centroids
    return normed_knn_points, knn_ids

def random_dilation_encoding(x_sample, x, k, n):

    xyz_sample = x_sample[:,:,:3] #
    xyz = x[:,:,:3] #
    feature = x[:,:,3:] #
    _, knn_idx = query_knn_point(int(k*n), xyz_sample, xyz) #
    rand_idx = torch.randperm(int(k*n))[:k] #
    dilation_idx = knn_idx[:,:,rand_idx]#
    dilation_xyz = index_points(xyz, dilation_idx)#
    dilation_feature = index_points(feature, dilation_idx)#
    xyz_expand = xyz_sample.unsqueeze(2).repeat(1,1,k,1)
    dilation_xyz_resi = dilation_xyz - xyz_expand
    dilation_xyz_dis = torch.norm(dilation_xyz_resi,dim=-1,keepdim=True)
    dilation_group = torch.cat((dilation_xyz_dis, dilation_xyz_resi),dim=-1)
    dilation_group = torch.cat((dilation_group, dilation_feature), dim=-1)
    dilation_group = dilation_group.permute(0,3,1,2)
    return dilation_group, dilation_xyz