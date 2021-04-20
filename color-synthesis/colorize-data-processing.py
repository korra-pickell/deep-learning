from PIL import Image, ImageOps
import os, random
import numpy as np
from progress.bar import Bar
import h5py

save_dir = ''

wdir = ['']

label = 'train'

#ALL FILES ARE PULLED FROM EVERY FOLDER LISTED IN "WDIR"
raw_images = [a for b in [[d+x for x in os.listdir(d)] for d in wdir] for a in b]
wkdir = os.path.dirname(os.path.realpath(__file__))

random.shuffle(raw_images)

h,w = 512,512

channels = 3

#IMAGES WILL BE PROCESSED IN CHUNKS, WHICH WILL BE SAVED AND COMPRESSED AS .NPZ
chunk_length = 100

bar = Bar('Processing',max = len(raw_images))

for chunk_index,chunk in enumerate([raw_images[i:i+chunk_length] for i in range(0,len(raw_images),chunk_length)]):
    x_data, y_data = [],[]
    for index,raw_image in enumerate(chunk):
        bar.next()
        xy = []

        #CROP LARGEST SQUARE FROM IMAGE
        image_original = Image.open(raw_image)
        if (image_original.size[0] < image_original.size[1]):
            image_original_square = image_original.resize((w,int(image_original.size[1]*(w/image_original.size[0])))).crop((0,0,w,h))
        else:
            image_original_square = image_original.resize((int(image_original.size[0]*(w/image_original.size[1])),h)).crop((0,0,w,h))
        
        #CONVERT TO GREYSCALE
        image_grey = ImageOps.grayscale(image_original_square)
        
        #For Training set, process flipped images in addition to normal orientation, doubling dataset
        if (label == 'train'):

            color_flipped = image_original_square.transpose(method=Image.FLIP_LEFT_RIGHT)
            grey_fipped = image_grey.transpose(method=Image.FLIP_LEFT_RIGHT)

            color_flipped_array =np.array(color_flipped)/127.5 - 1
            grey_flipped_array = np.array(grey_fipped)
            grey_flipped_array = np.stack((grey_flipped_array,)*3,axis=-1).astype('int32').reshape(w,h,channels)/127.5 - 1

            x_data.append(np.array(grey_flipped_array).reshape(-1,w,h,3))
            y_data.append(np.array(color_flipped_array).reshape(-1,w,h,3))

        color = np.array(image_original_square)/127.5 - 1
        grey = np.array(image_grey)
        grey = np.stack((grey,)*3,axis=-1).astype('int32').reshape(w,h,channels)/127.5 - 1

        x_data.append(np.array(grey).reshape(-1,w,h,3))
        y_data.append(np.array(color).reshape(-1,w,h,3))
    np.savez_compressed(save_dir+'/'+'outpaint-'+str(label)+'_x-'+str(w)+'-'+str(chunk_index)+'.npz',np.array(x_data))
    np.savez_compressed(save_dir+'/'+'outpaint-'+str(label)+'_y-'+str(w)+'-'+str(chunk_index)+'.npz',np.array(y_data))
    print('')
    print('SAVED: ' + str(index+1))

bar.finish()