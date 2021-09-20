#!/usr/bin/env python
# coding: utf-8

import torch
from PIL import Image
import numpy as np


def create_object_normal(size=100):
    a = np.zeros((size, size, size), dtype=np.bool)
    center = int(size/2)
    for i in range(0, size):
        for j in range(0, size):
            for k in range(0, size):
                if np.linalg.norm(np.array([i, j, k])-np.array([center, center, center])) < 25:
                    a[i, j, k] = True
    return a


def create_spike_big(size=100):
    s = int(size/2)
    a = np.zeros((size, size, size), dtype=np.bool)
    center = int(size/2)
    for i in range(0, size):
        for j in range(0, size):
            for k in range(0, size):
                if np.linalg.norm(np.array([i, j, k])-np.array([center, center, center])) < 25:
                    a[i, j, k] = True

    a[5:int(size/2), 49:51, s-1:s+1] = True
    return a


def create_spike_small(size=100):
    s = int(size/2)
    a = np.zeros((size, size, size), dtype=np.bool)
    center = int(size/2)
    for i in range(0, size):
        for j in range(0, size):
            for k in range(0, size):
                if np.linalg.norm(np.array([i, j, k])-np.array([center, center, center])) < 25:
                    a[i, j, k] = True
    a[5:int(size/2),49:51,s-1:s+1] = True
    a[49:51,10:size-10,s-1:s+1] = True
    a[int(size/2):size-10,49:51,s-1:s+1] = True
    a[s-1:s+1,49:51,10:size-10] = True

    return a


def RM(arr, w = 10, s = 10):
    (x0,y0,z0) = (0,0,0)
    cx = int(float(sum(torch.nonzero(arr)[:,0]))/float(torch.nonzero(arr).shape[0]))
    cy = int(float(sum(torch.nonzero(arr)[:,1]))/float(torch.nonzero(arr).shape[0]))
    cz = int(float(sum(torch.nonzero(arr)[:,2]))/float(torch.nonzero(arr).shape[0]))
    RI = 0.0
    (x,y,z) = arr.shape
    M = 0.0
    for x0 in range(0,x,s):
        for y0 in range(0,y,s):
            for z0 in range(0,z,s):
                N = 0.0
                D0 = 0.0
                if torch.nonzero(arr[x0:x0+w,y0:y0+w,z0:z0+w]).shape[0] > 0:
                    for i in range(x0,x0+w):
                        for j in range(y0,y0+w):
                            for k in range(z0,z0+w):
                                if i < x and j < y and k < z:
                                    if arr[i,j,k] == True:
                                        D0 += np.linalg.norm(np.array([i, j, k])-np.array([cx, cy, cz]))
                                        N += 1
                if N > 0:
                    D0 /= N
                N = 0.0
                R = 0.0
                if torch.nonzero(arr[x0:x0+w,y0:y0+w,z0:z0+w]).shape[0] > 0:
                    for i in range(x0,x0+w):
                        for j in range(y0,y0+w):
                            for k in range(z0,z0+w):
                                if x0+w < x and y0+w < y and z0+w < z: 
                                
                                    R += np.sqrt(np.square(np.linalg.norm(np.array([i, j, k])
                                                                          -np.array([cx, cy, cz])) - D0))
                                    N += 1
                if N > 0:
                    RI += R/N
                M += 1        
    if M > 0:
        RI /= M
    # print("Roughness Index:", RI)
    return RI 


def RM2D(arr, w = 3, s = 3):
    (x0,y0) = (0,0)
    cx = int(float(sum(torch.nonzero(arr)[:,0]))/float(torch.nonzero(arr).shape[0]))
    cy = int(float(sum(torch.nonzero(arr)[:,1]))/float(torch.nonzero(arr).shape[0]))
    #print(cx,cy)
    RI = 0.0
    (x,y) = arr.shape
    M = 0.0
    for x0 in range(0,x,s):
        for y0 in range(0,y,s):
            N = 0.0
            D0 = 0.0
            if torch.nonzero(arr[x0:x0+w,y0:y0+w]).shape[0] > 0:
                for i in range(x0,x0+w):
                    for j in range(y0,y0+w):
                        
                        if i < x and j < y:
                            if arr[i,j] == True:
                                D0 += np.linalg.norm(np.array([i, j])-np.array([cx, cy]))
                                N += 1
            if N > 0:
                D0 /= N
            N = 0.0
            R = 0.0
            if torch.nonzero(arr[x0:x0+w,y0:y0+w]).shape[0] > 0:
                for i in range(x0,x0+w):
                    for j in range(y0,y0+w):
                        if x0+w < x and y0+w < y: 
                            R += np.sqrt(np.square(np.linalg.norm(np.array([i, j])
                                                                  -np.array([cx, cy])) - D0))
                            N += 1
            if N > 0:
                RI += R/N
            M += 1
    if M > 0:
        RI /= M
    # print("Roughness Index:", RI)
    return RI

def RI2D(arr, w = 10, s = 10):
    w = int(arr.shape[0]*0.07)
    s = w
    (x0,y0) = (0,0)
    VM = torch.zeros(arr.shape[0],arr.shape[1],8)
    DM = torch.zeros(arr.shape[0],arr.shape[1])
    RI = torch.zeros(arr.shape[0],arr.shape[1])
    cx = int(float(sum(torch.nonzero(arr)[:,0]))/float(torch.nonzero(arr).shape[0]))
    cy = int(float(sum(torch.nonzero(arr)[:,1]))/float(torch.nonzero(arr).shape[0]))
    for i in range(0,arr.shape[0]):
        for j in range(0,arr.shape[1]):
            if arr[i,j] != 0:
                DM[i,j] = np.sqrt(np.square(i - cx) + np.square(j - cy))
    GM = np.gradient(DM)
    # print("center of gravity:")
    # print(cx,cy)
    for i in range(1,arr.shape[0] - 1):
        for j in range(1,arr.shape[1] - 1):
            if arr[i,j]:
                c = 0
                s = 0
                if arr[i-1,j-1]:
                    VM[i,j,0] = DM[i,j] - DM[i-1,j-1]
                    s += DM[i,j] - DM[i-1,j-1]
                    c += 1

                if arr[i,j-1]:
                    VM[i,j,1] = DM[i,j] - DM[i,j-1]
                    s += DM[i,j] - DM[i,j-1]
                    c += 1

                if arr[i+1,j-1]:
                    VM[i,j,2] = DM[i,j] - DM[i+1,j-1]
                    s += DM[i,j] - DM[i+1,j-1]
                    c += 1

                if arr[i-1,j]:
                    VM[i,j,3] = DM[i,j] - DM[i-1,j]
                    s += DM[i,j] - DM[i-1,j]
                    c += 1

                if arr[i+1,j]:
                    VM[i,j,4] = DM[i,j] - DM[i+1,j]
                    s += DM[i,j] - DM[i+1,j]
                    c += 1

                if arr[i-1,j+1]:
                    VM[i,j,5] = DM[i,j] - DM[i-1,j+1]
                    s += DM[i,j] - DM[i-1,j+1]
                    c += 1

                if arr[i,j+1]:
                    VM[i,j,6] = DM[i,j] - DM[i,j+1]
                    s += DM[i,j] - DM[i,j+1]
                    c += 1

                if arr[i+1, j+1]:
                    VM[i,j,7] = DM[i,j] - DM[i+1,j+1]
                    s += DM[i,j] - DM[i+1,j+1]
                    c += 1
                RI[i,j] = s/(c)

    return RI,DM


def RI2DNC(arr, w = 10, s = 10):
    w = int(arr.shape[0]*0.07)
    s = w
    (x0,y0) = (0,0)
    VM = torch.zeros(arr.shape[0],arr.shape[1],8)
    DM = torch.zeros(arr.shape[0],arr.shape[1])
    RI = torch.zeros(arr.shape[0],arr.shape[1])
    cx = int(float(sum(torch.nonzero(arr)[:,0]))/float(torch.nonzero(arr).shape[0]))
    cy = int(float(sum(torch.nonzero(arr)[:,1]))/float(torch.nonzero(arr).shape[0]))
    for i in range(0,arr.shape[0]):
        for j in range(0,arr.shape[1]):
            DM[i,j] = np.sqrt(np.square(i - cx) + np.square(j - cy))
    GM = np.gradient(DM)
    # print("center of gravity:")
    # print(cx,cy)
    for i in range(1,arr.shape[0] - 1):
        for j in range(1,arr.shape[1] - 1):
            if (arr[i,j]) == True:
                c = 0
                s = 0
                if arr[i-1,j-1]:
                    VM[i,j,0] = DM[i,j] - DM[i-1,j-1]
                    s += DM[i,j] - DM[i-1,j-1]
                    c += 1

                if arr[i,j-1]:
                    VM[i,j,1] = DM[i,j] - DM[i,j-1]
                    s += DM[i,j] - DM[i,j-1]
                    c += 1

                if arr[i+1,j-1]:
                    VM[i,j,2] = DM[i,j] - DM[i+1,j-1]
                    s += DM[i,j] - DM[i+1,j-1]
                    c += 1

                if arr[i-1,j]:
                    VM[i,j,3] = DM[i,j] - DM[i-1,j]
                    s += DM[i,j] - DM[i-1,j]
                    c += 1

                if arr[i+1,j]:
                    VM[i,j,4] = DM[i,j] - DM[i+1,j]
                    s += DM[i,j] - DM[i+1,j]
                    
                    c += 1

                if arr[i-1,j+1]:
                    VM[i,j,5] = DM[i,j] - DM[i-1,j+1]
                    s += DM[i,j] - DM[i-1,j+1]
                    c += 1

                if arr[i,j+1]:
                    VM[i,j,6] = DM[i,j] - DM[i,j+1]
                    s += DM[i,j] - DM[i,j+1]
                    c += 1

                if arr[i+1,j+1]:
                    VM[i,j,7] = DM[i,j] - DM[i+1,j+1]
                    s += DM[i,j] - DM[i+1,j+1]
                    c += 1
                    
                if (c >= 1) and (c <= 7):
                    RI[i,j] = s
    # if (isinstance(RI,float) or isinstance(RI,int)) and not isinstance(RI,list):
    #     print("Roughness Index:", RI)
    return RI,DM


def DM2D(arr):
    cx = int(float(sum(torch.nonzero(arr)[:,0]))/float(torch.nonzero(arr).shape[0]))
    cy = int(float(sum(torch.nonzero(arr)[:,1]))/float(torch.nonzero(arr).shape[0]))
    DM = torch.zeros(arr.shape[0],arr.shape[1])
    for i in range(0,arr.shape[0]):
        for j in range(0,arr.shape[1]):
            if arr[i,j] != 0:
                DM[i,j] = np.sqrt(np.square(i - cx) + np.square(j - cy))
                
    return DM


def DM3D(arr):
    cx = int(float(sum(torch.nonzero(arr)[:,0]))/float(torch.nonzero(arr).shape[0]))
    cy = int(float(sum(torch.nonzero(arr)[:,1]))/float(torch.nonzero(arr).shape[0]))
    cz = int(float(sum(torch.nonzero(arr)[:,2]))/float(torch.nonzero(arr).shape[0]))
    DM = torch.zeros(arr.shape[0],arr.shape[1],arr.shape[2])
    for i in range(0,arr.shape[0]):
        for j in range(0,arr.shape[1]):
            for k in range(0,arr.shape[2]):
                if arr[i,j,k] != 0:
                    DM[i,j,k] = np.sqrt(np.square(i - cx) + np.square(j - cy) + np.square(k - cz))
                
    return DM


def HDD(gt,pred):
    idx_gt = torch.nonzero(gt)
    idx_pred = torch.nonzero(pred)
    distances_gt_to_pred = torch.zeros(idx_gt.shape[0]).type(torch.float)
    distances_pred_to_gt = torch.zeros(idx_pred.shape[0]).type(torch.float)
    for i in range(0,idx_gt.shape[0]):
        distances_gt_to_pred[i] = torch.min(((idx_pred[:,0]-idx_gt[i,0])**2
                                                +(idx_pred[:,1]-idx_gt[i,1])**2)**0.5)
                
    for i in range(0,idx_pred.shape[0]):
        distances_pred_to_gt[i] = torch.min(((idx_gt[:,0]-idx_pred[i,0])**2
                                                +(idx_gt[:,1]-idx_pred[i,1])**2)**0.5)
    d_gt = torch.max(distances_gt_to_pred)
    d_pred = torch.max(distances_pred_to_gt)
    print("Hausdorff distance, HDD: ", torch.max(d_gt, d_pred).item())
    return torch.max(d_gt,d_pred)


def dilation(tensor, d=2, dim = 2):
    if dim == 2:
        k_shape = 2*d-1
        d_K = torch.ones(k_shape,k_shape).type(torch.float)
        d_K = d_K.repeat(1,1,1,1)
        dil_tensor= torch.nn.functional.conv2d(input=tensor.float(),weight=d_K,
                                               stride=(1,1), padding=(1,1))
        return dil_tensor
    if dim == 3:
        k_shape = 2*d-1
        d_K = torch.ones(k_shape,k_shape,k_shape).type(torch.float)
        d_K = d_K.repeat(1,1,1,1,1)
        dil_tensor= torch.nn.functional.conv3d(input=tensor.float(),weight=d_K,
                                               stride=(1,1,1), padding=(1,1,1))
        return dil_tensor


def dil(arr):
    Kx_weight = torch.tensor([[[128, 64],                                
                      [32, 16]],
                     [[8, 4],
                      [2, 1]]])
    Kx_weight = Kx_weight.repeat(1,1,1,1,1)
    nc = torch.nn.functional.conv3d(input=arr.unsqueeze(0).unsqueeze(0).float(),
                                    weight=Kx_weight.float(), stride=(1,1,1), padding=(1,1,1))
    be = (nc != 0) & (nc != 255)
    be = be.squeeze(0).squeeze(0)
    # plt.imshow(be[:,:,35].numpy().astype(np.float), cmap=plt.cm.gray)
    return be


def dil2D(arr):
    Kx_weight = torch.tensor(
                     [[8, 4],
                      [2, 1]])
    Kx_weight = Kx_weight.repeat(1,1,1,1)
    nc = torch.nn.functional.conv2d(input=arr.unsqueeze(0).unsqueeze(0).float(),
                                    weight=Kx_weight.float(), stride=(1,1), padding=(1,1))
    be = (nc != 0) & (nc != 15)
    be = dilation(be,1)
    be = be.squeeze(0).squeeze(0)
    # plt.imshow(be[:,:].numpy().astype(np.float), cmap=plt.cm.gray)
    return be    


def bitmp(img):    
    ary = np.array(img)
    r,g,b = np.split(ary,3,axis=2)
    r=r.reshape(-1)
    g=r.reshape(-1)
    b=r.reshape(-1)
    bitmap = list(map(lambda x: 0.299*x[0]+0.587*x[1]+0.114*x[2], 
    zip(r,g,b)))
    bitmap = np.array(bitmap).reshape([ary.shape[0], ary.shape[1]])
    bitmap = np.dot((bitmap > 128).astype(float),255)
    img_b = Image.fromarray(bitmap.astype(np.uint8))
    return img_b

#  EOF
