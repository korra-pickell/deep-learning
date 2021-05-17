from PIL import Image, ImageOps
import os, random
import numpy as np
from progress.bar import Bar
import h5py

save_dir = 'E:/DATA/NPY/relight/'

label = 'test'

if label == 'train':
    wdir_left = ['E:\\DATA\\Re-Lighting\\index_0\\left\\']
    wdir_right = ['E:\\DATA\\Re-Lighting\\index_0\\right\\']
else:
    wdir_left = ['E:\\DATA\\Re-Lighting\\demo 2\\left\\']
    wdir_right = ['E:\\DATA\\Re-Lighting\\demo 2\\right\\']

#ALL FILES ARE PULLED FROM EVERY FOLDER LISTED IN "WDIR"
starting_point = 0
cutoff = None

raw_images_left = [a for b in [[d+x for x in os.listdir(d)] for d in wdir_left] for a in b][starting_point:cutoff]
raw_images_right = [a for b in [[d+x for x in os.listdir(d)] for d in wdir_right] for a in b][starting_point:cutoff]

wkdir = os.path.dirname(os.path.realpath(__file__))

h,w = 512,512

channels = 3

#IMAGES WILL BE PROCESSED IN CHUNKS, WHICH WILL BE SAVED AND COMPRESSED AS .NPZ
chunk_length = 1000

bar = Bar('Processing',max = len(raw_images_left))

for chunk_index,chunk in enumerate([raw_images_left[i:i+chunk_length] for i in range(0,len(raw_images_left),chunk_length)]):
    x_data, y_data = [],[]
    for index,raw_image in enumerate(chunk):
        bar.next()
        xy = []

        image_left = Image.open(raw_image).resize((w,h))
        image_right = Image.open(raw_images_right[index]).resize((w,h))
        
        image_left_array = np.array(image_left)/127.5 - 1
        image_right_array = np.array(image_right)/127.5 - 1

        x_data.append(np.array(image_left_array).reshape(-1,w,h,3))
        y_data.append(np.array(image_right_array).reshape(-1,w,h,3))
    np.savez_compressed(save_dir+'/'+'relight-'+str(label)+'-'+str(w)+'-'+str(chunk_index)+'.npz',x=np.array(x_data),y=np.array(y_data))
    print('')
    print('SAVED: ' + str(index+1))

bar.finish()