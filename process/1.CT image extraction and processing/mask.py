import os
import numpy as np
import nibabel as nib 
import imageio 
import os
import numpy as np
from collections import Counter


file_paths = []
pathB="D:\A"
def load(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path): 
            load(file_path)  
        else:
            if os.path.splitext(file)[0] == 'mask':#Separate the file name from the extension, and determine whether it is a nii format file remember it is all mask
               file_paths.append(os.path.join(file_path))#The file name of the second layer was added to the end of the originally blank list by append to the column, and a loop was performed for storage.


    return file_paths

data_partB = load(pathB)
