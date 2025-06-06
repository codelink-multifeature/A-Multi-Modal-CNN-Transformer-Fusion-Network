import os
import numpy as np
import nibabel as nib 
import imageio 
import os
import numpy as np
from collections import Counter


file_paths = []
pathA="D:\\A"
def load(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):  
            load(file_path)  
        else:
            if os.path.splitext(file)[0] == 'data':#The file name and extension have been separated, and it is determined whether the file is in nii format
               file_paths.append(os.path.join(file_path))#The file name of the second layer was added to the end of the originally blank list by append to the column, and a loop was performed for storage.


    return file_paths

i=0
data_partA = load(pathA)
print(len(data_partA))

