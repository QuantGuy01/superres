import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, UpSampling2D, Conv2DTranspose, Reshape, Dropout, LeakyReLU, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

import numpy as np
import imgaug.augmenters as iaa



run = wandb.init(project='superres')
config = run.config

config.num_epochs = 50
config.batch_size = 32
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256

val_dir = 'data/test'
train_dir = 'data/train'

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xzf - -C data", shell=True)

config.steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size

#sometimes = lambda aug: iaa.Sometimes(0.25, aug)

seq = iaa.Sequential(
    iaa.Fliplr(0.10), # horizontally flip 50% of the images
'''    [iaa.Sometimes(0.20,
      #iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
      iaa.OneOf([
          iaa.GaussianBlur(sigma=(0, 0.5)), # blur images with a sigma of 0 to 3.0
          iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
          #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
          #iaa.Affine(rotate=(-45, 45)), # rotate by -45 to +45 degrees
          #iaa.AddToHueAndSaturation((-10, 10)), # change hue and saturation
      ])
#        iaa.OneOf([
#          #iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
#          #iaa.GammaContrast(0.90,1.10)
#        ])
    )
]
'''
)

def image_generator(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    random.shuffle(input_filenames)
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            #print("Processing image: "+img+"\n")
            small_images[i] = np.array(Image.open(img)) / 255.0
            large_images[i] = np.array(
            Image.open(img.replace("-in.jpg", "-out.jpg"))
            #.convert('RGB')
        ) / 255.0
        #small_images_aug = seq.augment_images(small_images)
        #yield (small_images_aug, large_images)
        yield (small_images, large_images)
        counter += batch_size


def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


val_generator = image_generator(config.batch_size, val_dir)
in_sample_images, out_sample_images = next(val_generator)


class ImageLogger(Callback):
    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(in_sample_images)
        in_resized = []
        for arr in in_sample_images:
            # Simple upsampling
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)

vggmodel = VGG16(include_top=False,input_shape=(256,256,3))
def VGGloss(y_true, y_pred):  # Note the parameter order
    f_p = vggmodel(y_pred)
    f_t = vggmodel(y_true)
    lossvgg = K.mean(K.square(f_t - f_p))
    lossmse = K.mean(K.square(y_true - y_pred))
    losscombined = (lossvgg + (lossmse * 7))/8
    return losscombined
  
def identity_block(x, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    x -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    x -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    x_shortcut = x
    
    # First component of main path
    x = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'same', name = conv_name_base + '2a')(x)
    x = BatchNormalization(name = bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    
    # Second component of main path (≈3 lines)
    x = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(x)
    x = BatchNormalization(name = bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # Third component of main path (≈2 lines)
    x = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'same', name = conv_name_base + '2c')(x)
    x = BatchNormalization(name = bn_name_base + '2c')(x)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    
    
    return x

def convolutional_block(x, f, filters, stage, block, s = 1):
    """
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    x_shortcut = x


    ##### MAIN PATH #####
    # First component of main path 
    x = Conv2D(F1, (1, 1),  name = conv_name_base + '2a')(x)
    x = BatchNormalization(name = bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # Second component of main path (≈3 lines)
    x = Conv2D(filters = F2, kernel_size = (f, f), padding = 'same', name = conv_name_base + '2b')(x)
    x = BatchNormalization(name = bn_name_base + '2b')(x)
    x = Activation('relu')(x)


    # Third component of main path (≈2 lines)
    x = Conv2D(filters = F3, kernel_size = (1, 1), padding = 'same', name = conv_name_base + '2c')(x)
    x = BatchNormalization(name = bn_name_base + '2c')(x)


    ##### SHORTCUT PATH #### (≈2 lines)
    x_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), padding = 'same', name = conv_name_base + '1',)(x_shortcut)
    x_shortcut = BatchNormalization(name = bn_name_base + '1')(x_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    print(x.get_shape().as_list()) #64
    print(x_shortcut.get_shape().as_list()) #128
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    
    
    return x


def up_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):
    filters1, filters2, filters3 = filters
    
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
        
    up_conv_name_base = 'u_up' + str(stage) + block + '_branch'
    conv_name_base = 'u_res' + str(stage) + block + '_branch'
    bn_name_base = 'u_bn' + str(stage) + block + '_branch'

    x = UpSampling2D(size=(2, 2), name=up_conv_name_base + '2a')(input_tensor)

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(x)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = UpSampling2D(size=(2, 2), name=up_conv_name_base + '1')(input_tensor)
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(shortcut)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x
  
  
  
# This returns a tensor
input_tensor = Input(shape=(config.input_width, config.input_height,3))
x = input_tensor

#x = Dense(32*32*3)(x)
x = Conv2D(81*3*3, (5, 5), padding='same', activation='relu')(x)
x = Conv2D(81*3, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(81*3, (3, 3), padding='same', activation='relu')(x)#1
#x = BatchNormalization()(x)
x = Dropout(0.05)(x)
x = UpSampling2D()(x)
x = Conv2D(81*3, (3, 3), padding='same', activation='relu')(x)#2
x = Conv2D(81, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(81, (3, 3), padding='same', activation='relu')(x)
x = Dropout(0.05)(x)
#x = BatchNormalization()(x)
#x = Dropout(0.10)(x)
#shortcut = x
#x = BatchNormalization()(x)
#x = layers.add([x, shortcut])
x = UpSampling2D()(x)
x = Conv2D(27, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(27, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(9, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(9, (3, 3), padding='same', activation='relu')(x)
#shortcut = x
#x = layers.add([x, shortcut])
x = UpSampling2D()(x)
#x = Dropout(0.2)(x)
x = Conv2D(3, (3, 3), padding='same', activation='relu')(x)
#x = Conv2D(3, (3, 3), padding='same', activation='relu')(x)
#x = Conv2DTranspose(9, (3,3), strides=2, padding='same', activation='relu')(x)
#x = Conv2DTranspose(27, (3,3), strides=2, padding='same', activation='relu')(x)
#x = Conv2DTranspose(81, (3,3), strides=2, padding='same', activation='relu')(x)
#x = Conv2D(27, (3, 3), padding='same', activation='relu')(x)
#x = Conv2D(9, (3, 3), padding='same', activation='relu')(x)
#x = Conv2D(3, (3, 3), padding='same', activation='relu')(x)

#x = Dense(12,input_shape=(32,32,3))(x)
#x = Reshape((64,64,3))(x)
#x = Conv2DTranspose(9, (3,3), strides=2, padding='same', activation='relu')(x)
#x = Conv2DTranspose(3, (3,3), strides=2, padding='same', activation='relu')(x)



#x = Conv2D(3, (7, 7), padding='same', activation='relu',
#           input_shape=(config.input_width, config.input_height, 3))(x)
#x = Conv2D(9, (5, 5), padding='same', activation='relu')(x)
#x = Conv2D(27, (3, 3), padding='same', activation='relu')(x)
#x = Conv2D(81, (3, 3), padding='same', activation='relu')(x)
#x = Flatten()(x)
#a = Model(inputs=input_tensor, outputs=x)
#print(a.summary())
'''
x = up_conv_block(x, 3, [81,81,81], stage = 2, block='a')
x = identity_block(x, 3, [81,81,81], stage=2, block='b')
x = identity_block(x, 3, [81,81,81], stage=2, block='c')
#x = layers.add([x, shortcut])
''' 
'''
x = Conv2D(3, (3, 3), padding='same', activation='relu')(x)
x = UpSampling2D()(x)
x = Conv2D(3, (3, 3), padding='same', activation='relu')(x)
x = UpSampling2D()(x)
x = Conv2D(3, (3, 3), padding='same', activation='relu')(x)
x = UpSampling2D()(x)
x = Conv2D(3, (3, 3), padding='same', activation='relu')(x)
'''

'''
#x = UpSampling2D()(x)

# Stage 1
x = Conv2D(3, (7, 7), padding='same', name = 'conv1')(x)
x = BatchNormalization(name = 'bn_conv1')(x)
x = Activation('relu')(x)

#x = MaxPooling2D((3, 3), strides=(1, 1))(x)

#x = MaxPooling2D((3, 3), strides=(2, 2))(x)

# Stage 2
x = identity_block(x, 5, [3, 3, 3], stage=2, block='b')
x = identity_block(x, 5, [3, 3, 3], stage=2, block='c')
x = up_conv_block(x, 5, filters=[3,3,9], stage=2, block='a')

#x = Conv2D(256, (1, 1), padding='same', name = 'conv3')(x)
#x = BatchNormalization(name = 'bn_conv3')(x)
#x = Activation('relu')(x)

# Stage 3 (≈4 lines)
x = identity_block(x, 3, [9,9,9], stage=3, block='b')
x = identity_block(x, 3, [9,9,9], stage=3, block='c')
x = up_conv_block(x, 3, filters = [9,9,3], stage = 3, block='b')

# Stage 3 (≈4 lines)
x = up_conv_block(x, 3, filters = [3,3,3], stage = 4, block='c')

# AVGPOOL (≈1 line). Use "x = AveragePooling2D(...)(x)"
#x = AveragePooling2D((2,2), name="avg_pool")(x)
'''




'''
### END CODE HERE ###

# output layer
x = Flatten()(x)
x = Dense(classes, activation='softmax', name='fc' + str(classes))(x)
'''


#x = Conv2D(256, (1,1), padding='same')(x)
#x = Activation('relu')(x)



'''
shortcut = layers.Conv2D(3, (3, 3), padding='same')(input_tensor)
shortcut = layers.BatchNormalization()(shortcut)

x = Add()([x, shortcut])
x = BatchNormalization()(x)
'''

#x = Conv2D(3, (3, 3), padding='same', activation='relu')(x)

output_tensor = x

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=input_tensor, outputs=output_tensor)
print(model.summary())



# DONT ALTER metrics=[perceptual_distance]
opt=Adam(lr=0.0001) #lr=xxx
model.compile(optimizer=opt, loss='mse',
              metrics=[perceptual_distance])

model.fit_generator(image_generator(config.batch_size, train_dir),
                    steps_per_epoch=config.steps_per_epoch,
                    epochs=config.num_epochs, callbacks=[
                        ImageLogger(), WandbCallback()],
                    validation_steps=config.val_steps_per_epoch,
                    validation_data=val_generator)
