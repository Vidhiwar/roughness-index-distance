#!/usr/bin/env python
# coding: utf-8

import os
import time
import nibabel as nib
import matplotlib.pyplot as plt
from lib.functions import *
from tqdm import tqdm

work_dir = ''

start_time = time.time()
print("Starting process...")

i1 = torch.tensor(np.array(bitmp(Image.open(os.path.join(work_dir, 'data', "SD1.png")))))
i2 = torch.tensor(np.array(bitmp(Image.open(os.path.join(work_dir, 'data', "SD2.png")))))
f1 = torch.tensor(np.array(bitmp(Image.open(os.path.join(work_dir, 'data', "fig1.jpg")))))
f2 = torch.tensor(np.array(bitmp(Image.open(os.path.join(work_dir, 'data', "fig2.jpg")))))
f3 = torch.tensor(np.array(bitmp(Image.open(os.path.join(work_dir, 'data', "fig3.jpg")))))
p1 = torch.tensor(np.array(bitmp(Image.open(os.path.join(work_dir, 'data', "G1.jpg")))))
p2 = torch.tensor(np.array(bitmp(Image.open(os.path.join(work_dir, 'data', "S1.jpg")))))
p3 = torch.tensor(np.array(bitmp(Image.open(os.path.join(work_dir, 'data', "P1.jpg")))))

a, G = RI2D(dil2D(~(f1.bool())))
RI = a
if (isinstance(RI,float) or isinstance(RI,int)) and not isinstance(RI,list):
        print("Roughness Index:", RI)
a_, G_ = RI2DNC(~(f1.bool()))

g = np.absolute(a)
o = np.expand_dims(g, axis=2)
n = np.repeat(o*255, 3, axis=2).astype(np.uint8)
n_digits = 2
r = a
r = torch.round(r * 10**n_digits) / (10**n_digits)
r = torch.abs(r)

c = torch.unique(r, return_counts = True)[1]
u = torch.unique(r, return_counts = True)[0]

nb = []
nc = []
for i in tqdm(range(0, u.shape[0])):
    if u[i] > 0.001:
        nb.append(u[i])
        nc.append(c[i])
plt.close()
plt.plot(nb,nc)
plt.xlabel("Roughness Ratio")
plt.ylabel("Window size")
plt.savefig(os.path.join(work_dir,'output',
                         'RoughnessRatio-windowSize.png'))

print("------------------------------------------------------------------------------------------------------------")
print("")
print("")
print("")

a1 = create_object_normal()
a2 = create_spike_big()
a3 = create_spike_small()
nib.save(nib.Nifti1Image(a1.astype(np.int16),
                         affine=np.eye(4,4)), os.path.join(work_dir,'data','a1.nii.gz'))
nib.save(nib.Nifti1Image(a2.astype(np.int16),
                         affine=np.eye(4,4)), os.path.join(work_dir,'data','a2.nii.gz'))
nib.save(nib.Nifti1Image(a3.astype(np.int16),
                         affine=np.eye(4,4)), os.path.join(work_dir,'data','a3.nii.gz'))


HDD(dil(torch.from_numpy(a1)),dil(torch.from_numpy(a2)))
HDD(dil(torch.from_numpy(a1)),dil(torch.from_numpy(a3)))
HDD(dil2D(~(p1.bool())),dil2D(~(p3.bool())))
HDD(dil2D(~(p2.bool())),dil2D(~(p1.bool())))

print("Hausdorff Distance is Same!")
print("------------------------------------------------------------------------------------------------------------")
print("")
print("")
print("")


p = dil2D(i1.bool())
r1 = []
for w in tqdm(range(20, 60)):
    r = RM2D(p,w=w,s = w)
    r1.append(r)


p = dil2D(i2.bool())
r2 = []
for w in tqdm(range(20,60)):
    r = RM2D(p,w=w,s = w)
    r2.append(r)
plt.close()
plt.plot(np.arange(20,60),r1)
plt.ylabel('Roughness Ratio')
plt.xlabel('window size')
# plt.show()
plt.savefig(os.path.join(work_dir,'output',
                         'RoughnessRatio-WindowSize_3.png'))
plt.close()
plt.plot(np.arange(20,60),r2)
plt.ylabel('Roughness Ratio')
plt.xlabel('window size')
plt.savefig(os.path.join(work_dir,'output',
                         'RoughnessRatio-WindowSize_4.png'))
plt.close()
plt.plot(np.arange(20,60),r1,'r',
         label = 'Smooth Predicted Segmentation')
plt.plot(np.arange(20,60),r2,'b',
         label = 'Rough Predicted Segmentation')
plt.legend(loc = 'upper right')
plt.ylabel('Roughness Ratio')
plt.xlabel('window size')
plt.savefig(os.path.join(work_dir,'output',
                         'RoughnessRatio-WindowSize_compare.png'))

p11 = dil2D(~(f1.bool()))
r1 = []
p22 = dil2D(~(f2.bool()))
r2 = []
p33 = dil2D(~(f3.bool()))
r3 = []
for w in tqdm(range(20,60)):
    r11 = RM2D(p11,w=w,s = w)
    r1.append(r11)

    r22 = RM2D(p22,w=w,s = w)
    r2.append(r22)

    r33 = RM2D(p33,w=w,s = w)
    r3.append(r33)


p11 = dil2D((p1.bool()))
r1 = []
p22 = dil2D(~(p2.bool()))
r2 = []
p33 = dil2D(~(p3.bool()))
r3 = []
for w in tqdm(range(3,10)):
    r11 = RM2D(p11,w=w,s = w)
    r1.append(r11)
    r22 = RM2D(p22,w=w,s = w)
    r2.append(r22)
    r33 = RM2D(p33,w=w,s = w)
    r3.append(r33)

p11 = dil(torch.from_numpy(a1))
r1 = []
r1_n = []
r1_n2 = []
p22 = dil(torch.from_numpy(a2))
r2 = []
r2_n = []
r2_n2 = []
p33 = dil(torch.from_numpy(a3))
r3 = []
r3_n = []
r3_n2 = []
for w in tqdm(range(3,8)):
    r11 = RM(p11,w=w,s = w)
    r1.append(r11)
    r22 = RM(p22,w=w,s = w)
    r2.append(r22)
    r33 = RM(p33,w=w,s = w)
    r3.append(r33)

plt.close()
plt.plot(np.arange(3,8,1), r1,'r',label = 'Image(a)')
plt.plot(np.arange(3,8,1), r2,'b',label = 'Image(b)')
plt.plot(np.arange(3,8,1), r3,'g',label = 'Image(c)')
plt.legend(loc = 'upper right')
plt.ylabel('Roughness index')
plt.xlabel('window size')
plt.savefig(os.path.join(work_dir, 'output',
                         'RoughnessIndex-WindowSize.png'))

print("------------------------------------------------------------------------------------------------------------")
print("")
print("")
print("")


print("Total process Time:", time.time()-start_time, "Seconds")

# EOF