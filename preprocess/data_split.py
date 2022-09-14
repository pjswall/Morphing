import os 
import numpy as np
import shutil

root_dir = "crop_morph"
Dest1 = "ID_1/"
Dest2 = "ID_2/"

class_dir = ['ID_1','ID_2'] 

test_ratio = 0.20

for cls in class_dir:
    os.makedir(root_dir+'ID_1/'+cls)
    os.makedir(root_dir+'ID_2/'+cls)
    
    
src =  root_dir

allFileNames = os.listdir(src)
np.random.shuffle(allFilenames)

ID_1_filenames, ID_2_filenames = np.split(np.array(allFileNames),[int(len(allFileNames)* (1 - (test_ratio))), 
                                                           int(len(allFileNames)* (test_ratio))])

ID_1_filenames = [src +'/'+ name for name in ID_1_filenames.tolist()] 
ID_2_filenames = [src +'/'+ name for name in ID_2_filenames.tolist()]

print("Total images:" ,  len(allFileNames))
print("ID_1 images:" , len(ID_1_filenames))
print("ID_2 images:" , len(ID_2_filenames))

for name in ID_1_filenames:
    shutil.copy(name, Dest1 +'ID_1/' + cls)

for name in ID_2_filenames:
    shutil.copy(name, Dest2 +'ID_2/' + cls)


    