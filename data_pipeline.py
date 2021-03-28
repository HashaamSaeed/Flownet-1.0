import os
import shutil 
import cv2
import numpy as np
from tqdm import tqdm
import glob
#from Ops_Utils_flowlib_show import show_flow
#from imageio import imread







train_root = 'ChairsSDHom/ChairsSDHom/data/train'

test_root = 'ChairsSDHom/ChairsSDHom/data/test/'

img_t0_path = 't0'
img_t1_path = 't1'
img_flo_path = 'flow_flo'


def read_flow(filename):
    #if filename.endswith('.pfm') or filename.endswith('.PFM'):
     #   return readPFM(filename)[0][:,:,0:2]

    f = open(filename, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)




def get_train_files():
    img_t0_join= os.path.join(train_root,img_t0_path)
    images_t0_join = glob.glob(os.path.join(img_t0_join,'*.*'))

    img_t1_join= os.path.join(train_root,img_t1_path)
    images_t1_join = glob.glob(os.path.join(img_t1_join,'*.*'))

    img_flo_join = os.path.join(train_root,img_flo_path)
    images_flo = glob.glob(os.path.join(img_flo_join,'*.*'))

    return [os.path.basename(image_i_t0) for image_i_t0 in images_t0_join],[os.path.basename(image_i_t1) for image_i_t1 in images_t1_join],[os.path.basename(image_i_flo) for image_i_flo in images_flo]

all_batches = get_train_files()
#print(get_train_data())

path_t0_forloop = os.path.join(train_root,img_t0_path)
path_t1_forloop = os.path.join(train_root,img_t1_path)
path_flo_forloop = os.path.join(train_root,img_flo_path)


def get_train_data():
    img_cat_all = []
    flo_all = []

    for num in tqdm(range(len(all_batches[0]))):
        img_t0_path_here = os.path.join(path_t0_forloop,all_batches[0][num])
        img_t0_load = cv2.imread(img_t0_path_here)
        #print(img_t0_load.shape)
        

        img_t1_path_here  = os.path.join(path_t1_forloop,all_batches[1][num])
        img_t1_load = cv2.imread(img_t0_path_here)
        #print(img_t1_load.shape)

        img_cat = np.concatenate((img_t0_load,img_t1_load),axis=2)
        img_cat_all.append(img_cat)
        #print(img_cat.shape)
        
        flo_img_path_here  = os.path.join(path_flo_forloop,all_batches[2][num])
        flo_img_load = read_flow(flo_img_path_here)       
        flo_all.append(flo_img_load)
        #print(flo_img_load.shape)

    img_cat_all_numpy = np.asarray(img_cat_all)
    flo_all_numpy = np.asarray(flo_all)
    
    return img_cat_all_numpy , flo_all_numpy


