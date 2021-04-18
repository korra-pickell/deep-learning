from PIL import Image
import os, random
import numpy as np
from progress.bar import Bar

#Output Directory
save_dir = ''

#List of directories from which to pull images for processing
data_dir = ['E:/DATA/mountain-photos/']

#Differentiate between 'train' set and 'test' set
label = 'train'

#Compile all image paths into single list
raw_images = [a for b in [[d+x for x in os.listdir(d)] for d in data_dir] for a in b]
wkdir = os.path.dirname(os.path.realpath(__file__))

#Randomize order of images
random.shuffle(raw_images)

x_data, y_data = [],[]

#Specify the desired height and width of outputted image data
h,w = 512,512

#Specify the desired number of channels per image
channels = 1

cropping = {
    512: [80,432],
    256: [53,203],
    128: [32,96],
    64: [16,48],
    20: [4,16],
    32: [8,24]
}

#Create grey background on which to place the cropped photo
grey_image = Image.new('RGB',(w,h),(128,128,128))

bar = Bar('Processing',max = len(raw_images))

for index,raw_image in enumerate(raw_images[:6300]):
    bar.next()
    xy = []

    image_original = Image.open(raw_image)

    #Resize the photo with respect to its orientation
    if (image_original.size[0] < image_original.size[1]):
        image_original_square = image_original.resize((w,int(image_original.size[1]*(w/image_original.size[0])))).crop((0,0,w,h))#.convert('L')
    else:
        image_original_square = image_original.resize((int(image_original.size[0]*(w/image_original.size[1])),h)).crop((0,0,w,h))#.convert('L')
    
    #Crop photos to dimensions specified in 'cropping' array
    image_original_cropped = image_original_square.crop((cropping[w][0],0,cropping[w][1],h))
    grey_image.paste(image_original_cropped, (cropping[w][0], 0))
   
    #Convert images to Greyscale if desired
    #Model (x,y) = (crop,full)
    full = image_original_square.convert('L')
    crop = grey_image.convert('L')

    #Convert image data to NumPy array as values between 0-1
    full_array = np.array(image_original_square.convert('L'))/127.5 - 1
    crop_array = np.array(grey_image.convert('L'))/127.5 - 1
    
    #For training data, append both normal image data, as well as horizontally flipped image data, to double sample set
    if (label == 'train'):

        full_flipped = full.transpose(method=Image.FLIP_LEFT_RIGHT)
        crop_flipped = crop.transpose(method=Image.FLIP_LEFT_RIGHT)

        full_flipped_array = np.array(full_flipped.convert('L'))/127.5 - 1
        crop_flipped_array = np.array(crop_flipped.convert('L'))/127.5 - 1

        x_data.append(np.array(crop_flipped_array).reshape(-1,w,h,1))
        y_data.append(np.array(full_flipped_array).reshape(-1,w,h,1))

    x_data.append(np.array(crop_array).reshape(-1,w,h,1))
    y_data.append(np.array(full_array).reshape(-1,w,h,1))
    
bar.finish()

print('TRAIN DATA LENGTH:')
print(len(x_data),len(y_data))

chunk_length = 200

#Split data into chunks of length 'chunk_length' to be saved to seperate x and y files
data_chunks_x = [x_data[i:i+chunk_length] for i in range(0,len(x_data),chunk_length)]
data_chunks_y = [y_data[i:i+chunk_length] for i in range(0,len(y_data),chunk_length)]

for index,chunk in enumerate(data_chunks_x):
    np.save(save_dir+'/'+'outpaint-'+str(label)+'_x-'+str(w)+'-'+str(index)+'.npy',np.array(chunk))
    print('SAVED: ' + str(index+1))

for index,chunk in enumerate(data_chunks_y):
    np.save(save_dir+'/'+'outpaint-'+str(label)+'_y-'+str(w)+'-'+str(index)+'.npy',np.array(chunk))
    print('SAVED: ' + str(index+1))


#SINGLE FILE [X,Y]
#np.save(save_dir+'/'+'outpaint-'+str(label)+'-'+str(w)+'-'+str(channels)+'.npy',train_data)

#[X,Y] SPLIT INTO CHUNKS
'''
chunk_length = 2100

data_chunks = [train_data[i:i+chunk_length] for i in range(0,len(train_data),chunk_length)]

for index,chunk in enumerate(data_chunks):
    np.save(save_dir+'/'+'outpaint-'+str(label)+'-'+str(w)+'-'+str(channels)+'-'+str(index)+'.npy',chunk)
    print('SAVED: ' + str(index+1))
'''
