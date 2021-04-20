from PIL import Image, ImageOps
import os, random
import numpy as np
from progress.bar import Bar
import h5py

save_dir = ''
compression = True

label = 'train'

#Four seperate file directories containing (noisy, normal) image data for both training and testing
if label == 'train':
    wdir_normal = ''
    wdir_noisy =  ''
else:
    wdir_normal = ''
    wdir_noisy =  ''


raw_images_normal = [x for x in os.listdir(wdir_normal)]

wkdir = os.path.dirname(os.path.realpath(__file__))

random.shuffle(raw_images_normal)

h,w = 512,512

channels = 3

chunk_length = 200

bar = Bar('Processing',max = len(raw_images_normal))

#Process images in chunks of 'chunk_length' and save those chunks individually, saves system resources

for chunk_index,chunk in enumerate([raw_images_normal[i:i+chunk_length] for i in range(0,len(raw_images_normal),chunk_length)]):
    x_data, y_data = [],[]
    for index,raw_image in enumerate(chunk):
        bar.next()
        xy = []

        image_original_normal = Image.open(wdir_normal+raw_image)
        image_original_noisy = Image.open(wdir_noisy+raw_image)
        
        #Crop largest squares from images
        if (image_original_normal.size[0] < image_original_normal.size[1]):
            image_original_normal_square = image_original_normal.resize((w,int(image_original_normal.size[1]*(w/image_original_normal.size[0])))).crop((0,0,w,h))
            image_original_noisy_square = image_original_noisy.resize((w,int(image_original_noisy.size[1]*(w/image_original_noisy.size[0])))).crop((0,0,w,h))
        else:
            image_original_normal_square = image_original_normal.resize((int(image_original_normal.size[0]*(w/image_original_normal.size[1])),h)).crop((0,0,w,h))
            image_original_noisy_square = image_original_noisy.resize((int(image_original_noisy.size[0]*(w/image_original_noisy.size[1])),h)).crop((0,0,w,h))

        normal_array = np.array(image_original_normal_square)/127.5 - 1
        noisy_array = np.array(image_original_noisy_square)/127.5 - 1

        #Flip images to double dataset size
        if (label == 'train'):
            normal_flipped = image_original_normal_square.transpose(method=Image.FLIP_LEFT_RIGHT)
            noisy_flipped = image_original_noisy_square.transpose(method=Image.FLIP_LEFT_RIGHT)

            normal_flipped_array = np.array(normal_flipped)/127.5 - 1
            noisy_flipped_array = np.array(noisy_flipped)/127.5 - 1

            x_data.append(np.array(noisy_flipped_array).reshape(-1,w,h,3))
            y_data.append(np.array(normal_flipped_array).reshape(-1,w,h,3))

        x_data.append(np.array(noisy_array).reshape(-1,w,h,3))
        y_data.append(np.array(normal_array).reshape(-1,w,h,3))

    if compression == True:
        np.savez_compressed(save_dir+'/'+str(label)+'-'+str(w)+'-'+str(chunk_index)+'.npz',x=np.array(x_data),y=np.array(y_data))
    else:
        np.save(save_dir+'/'+str(label)+'_x-'+str(w)+'-'+str(chunk_index)+'.npy',np.array(x_data))
        np.save(save_dir+'/'+str(label)+'_y-'+str(w)+'-'+str(chunk_index)+'.npy',np.array(y_data))
    print('')
    print('SAVED: ' + str(index+1))
bar.finish()