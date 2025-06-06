from http.client import FORBIDDEN
import os
from xml.etree.ElementTree import tostringlist
import numpy as np
import nibabel as nib 
from data import data_partA
from mask import data_partB
def read_nifti_file(filepath):

    scan = nib.load(filepath)#Load and read the file name in nii format

    scan = scan.get_fdata()
    return scan

def MAX1(tensor1,min1):
    i=0
    U=0
    ma=0
    c=0
    for u in range(len(tensor1)):
      c=0
      for i in range(len(tensor1)):
         if min1==tensor1[i]:
            c=1
            break
      if c==1:
        min1=min1+1
        continue
      else:
        ma=min1-1
        break
    return ma#Return the first circle
   

#for i in range(len(data_partA)):The loop can be set according to the demand

pathA=data_partA[2]#data
pathB=data_partB[2]#mask
print(pathB)    
print(pathA)   
nii_img = nib.load(pathA)
affine = nii_img.affine.copy()
hdr = nii_img.header.copy()

nii_img1 = nib.load(pathB)
affine1 = nii_img1.affine.copy()
hdr1 = nii_img1.header.copy()



mask_tensor = np.array([read_nifti_file(pathB) ])
yuan_tensor = np.array([read_nifti_file(pathA) ])

# Determine T, length, width and height
t,h,w,g=yuan_tensor.shape


# Determine the center point
tempL = np.nonzero(mask_tensor)


minx= np.min(tempL[1])
miny= np.min(tempL[2])
minz = np.min(tempL[3])

maxx= np.max(tempL[1])
maxy= np.max(tempL[2])
maxz= np.max(tempL[3])


maxx=MAX1(tempL[1],minx)
maxy=MAX1(tempL[2],miny)
maxz=MAX1(tempL[3],minz)

pyx=int((minx+maxx)/2)
pyy=int((miny+maxy)/2)
pyz=int((minz+maxz)/2)
# Determine the center point
  

# Cut into two-dimensional nii - tensor (originally a 4-dimensional nii image)
yuan_qieX= yuan_tensor[ 0,pyx,:,:] 
yuan_qieY= yuan_tensor[ 0,:,pyy,:] 
yuan_qieZ= yuan_tensor[ 0,:,:,pyz] 


mask_qieX= mask_tensor[ 0,pyx,:,:] 
mask_qieY= mask_tensor[ 0,:,pyy,:] 
mask_qieZ= mask_tensor[ 0,:,:,pyz] 

 # 32*32
img_qieX =yuan_qieX [max((pyy - 16), 1):min((pyy + 16),w ),max((pyz - 16), 1):min((pyz + 16), g)]
img_qieY =yuan_qieY [max((pyx - 16), 1):min((pyx + 16),h ),max((pyz - 16), 1):min((pyz + 16), g)]
img_qieZ =yuan_qieZ [max((pyx - 16), 1):min((pyx + 16),h ),max((pyy - 16), 1):min((pyy + 16), w)]


mask_qieX =mask_qieX [max((pyy - 16), 1):min((pyy + 16),w ),max((pyz - 16), 1):min((pyz + 16), g)]
mask_qieY =mask_qieY [max((pyx - 16), 1):min((pyx + 16),h ),max((pyz - 16), 1):min((pyz + 16), g)]
mask_qieZ =mask_qieZ [max((pyx - 16), 1):min((pyx + 16),h ),max((pyy - 16), 1):min((pyy + 16), w)]

img_qie= np.array((img_qieX,img_qieY,img_qieZ))

img_qie1= np.array((mask_qieX,mask_qieY,mask_qieZ))


new_niiX = nib.Nifti1Image(img_qie, affine, hdr)

new_niiX1 = nib.Nifti1Image(img_qie1, affine1, hdr1)


nib.save(new_niiX, "2D data" )#data

nib.save(new_niiX1, "2D mask" )#,mask