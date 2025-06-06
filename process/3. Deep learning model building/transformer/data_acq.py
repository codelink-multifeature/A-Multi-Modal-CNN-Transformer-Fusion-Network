import os
import numpy as np
from keras import *
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow import *
import numpy as np
from scipy import ndimage
def read_nifti_file(filepath):
    scan = nib.load(filepath)#Load and read the file name in nii format
    scan = scan.get_fdata()## Convert the original data to a matrix dataset of float type
    scan=scan.transpose(2,1,0)

    return scan

def normalize(volume):
    min = np.min(volume)
    max = np.max(volume)
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def process_scan(path):
    volume = read_nifti_file(path)
    volume = normalize(volume)
    #volume =volume[np.newaxis,:,:]
   # volume = resize_volume(volume)
    #volume=vgg16(volume)
    return volume
    


file_paths = []
title = []

pathA = r'D:\A'## The address of Folder A of Data (the initial total address)
pathB = r'D:\B'## The address of Folder B of Data (the initial total address)


def load(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)  
        if os.path.isdir(file_path): 
            load(file_path) 
        else:
            if os.path.splitext(file)[0] == '2D data':
                file_paths.append(os.path.join(file_path))
    return file_paths


data_partA = load(pathA)
data_partB = load(pathB)

normal_scan_paths = []
abnormal_scan_paths = []

for i in data_partA:
    if i[3:4] is 'A':
        normal_scan_paths.append(i)
    if i[3:4] is 'B':
        abnormal_scan_paths.append(i)


normal_scans = np.array([process_scan(path) for path in normal_scan_paths])

abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])

#normal_scans=np.squeeze(normal_scans, axis = None)
#abnormal_scans=np.squeeze(abnormal_scans, axis = None)

print(normal_scans.shape)
print(abnormal_scans .shape)
A=normal_scans.shape[0]
B=abnormal_scans.shape[0]
C=A+B
cancer= np.concatenate((normal_scans, abnormal_scans))

X = np.array(list(normal_scans) + list(abnormal_scans))
print(X.shape)

text=np.zeros((C, 2))
for i in range(A):
   text[i][0]=1

for i in range(B):

  text[i+A][1]=1


Processing_sequence = abnormal_scan_paths + normal_scan_paths
print("***************************")
print(Processing_sequence)

Processing_sequence_Patient_number = []
for k in Processing_sequence:
    a = k[5:12]

    Processing_sequence_Patient_number.append(a)
# print(Processing_sequence_Patient_number)
# partA is 0 ,partA is normal
# partB is 1 ,partB is abnormal
# first B then A

Processing_sequence_Patient_number_partA = []
Processing_sequence_Patient_number_partB = []

for i in normal_scan_paths:
    Processing_sequence_Patient_number_partA.append(i[5:][:8])
for i in abnormal_scan_paths:
    Processing_sequence_Patient_number_partB.append(i[5:][:8])
print(Processing_sequence_Patient_number_partA)
print(Processing_sequence_Patient_number_partB)
all_Patient_number = Processing_sequence_Patient_number_partB + Processing_sequence_Patient_number_partA
print(all_Patient_number)

