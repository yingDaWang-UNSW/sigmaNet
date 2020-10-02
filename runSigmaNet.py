'''
SigmaNet implementation - based on the CycleSR network.
primary models are a cycleGAN for noise control and an SR generator.
training can occur in forwards mode or in sigma mode
Forwards mode: CycleGAN[LQLR -> HQLR] ---> HQLR -> HQSR - learn the degradation and pass LGSR back into the cycleGAN
Sigma mode: CycleGAN[HQLR -> LQLR] -> HQSR - do noise removal separately to SR - no pixelwise feedback to noise removal
write this to train with loose coupling - more lines of code - cycleGAN, GSR, 

for the cycleGAN, use standard LSGAN+resnet w/ upsampling or Unet w/ upsampling
for srgan: use ESRGAN (relGAN+rrdb+rrfdb+gradientguidance), but start with EDSRGAN - make the network jointly learn idt and degraded lr
use radiomics losses instead of vgg losses - try to incorporate LPIPS and NIQE - but retrain them? - add fft losses
variational isnt needed because we can just synthesise weights/images between PSNR and GAN 
'''

#TODO: write up testing section if train if test.
#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import sigmaNetArgs
from utils import tifffile
# Helper libraries
from sys import stdout
import numpy as np
import os
from glob import glob
import time
import datetime
import scipy.io
import PIL
import PIL.Image
import pdb
from matplotlib import pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE
print(tf.__version__)

args=sigmaNetArgs.args() # args is global

args.iterNum = args.iterNum//args.batch_size
if args.mixedPrecision:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
else:
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

# detect hardware
if len(args.gpuIDs.split(','))<=1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
else:
    strategy = tf.distribute.MirroredStrategy()


# define the network
with strategy.scope():
    def loadtf2(image_file):
      """Loads the image and generates input and target image.
      Args:
        image_file: .jpeg file
      Returns:
        Input image, target image
      """
      image = tf.io.read_file(image_file)
      image = tf.image.decode_png(image)
      image = tf.cast(image, tf.float32)
      return image

    # quick test to see if load is correct
    #realLR = loadtf2(args.dataset_dir+'trainA/tomo00001.png')
    #realHR = loadtf2(args.dataset_dir+'trainB/tomo00001.png')
    #plt.figure()
    #plt.imshow(realLR/255.0)
    #plt.figure()
    #plt.imshow(realHR/255.0)
    #plt.show()

        
    def bicubic_kernel(x, a=0): # use hermite resampling a=0 to avoid black white inversion
      """https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic"""
      if abs(x) <= 1:
        return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
      elif 1 < abs(x) and abs(x) < 2:
        return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a 
      else:
        return 0

    def build_filter(factor):
      size = factor*4
      k = np.zeros((size))
      for i in range(size):
        x = (1/factor)*(i- np.floor(size/2) +0.5)
        k[i] = bicubic_kernel(x)
      k = k / np.sum(k)
      # make 2d
      k = np.outer(k, k.T)
      k = tf.constant(k, dtype=tf.float32, shape=(size, size, 1, 1))
      return k#tf.concat([k, k, k], axis=2)

    def apply_bicubic_downsample(x, filter, factor):
      """Downsample x by a factor of factor, using the filter built by build_filter()
      x: a rank 4 tensor with format NHWC
      filter: from build_filter(factor)
      factor: downsampling factor (ex: factor=2 means the output size is (h/2, w/2))
      """
      # using padding calculations from https://www.tensorflow.org/api_guides/python/nn#Convolution
      #x = (x+1)*127.5
      filter_height = factor*4
      filter_width = factor*4
      strides = factor
      pad_along_height = max(filter_height - strides, 0)
      pad_along_width = max(filter_width - strides, 0)
      # compute actual padding values for each side
      pad_top = pad_along_height // 2
      pad_bottom = pad_along_height - pad_top
      pad_left = pad_along_width // 2
      pad_right = pad_along_width - pad_left
      # apply mirror padding
      x = tf.pad(x, [[0,0], [pad_top,pad_bottom], [pad_left,pad_right], [0,0]], mode='REFLECT')
      # downsampling performed by strided conv
      x = tf.nn.depthwise_conv2d(x, filter=filter, strides=[1,strides,strides,1], padding='VALID')
      #x = x/127.5 - 1
      return x

    def resizetf2(image, height, width):
      k = build_filter(factor=2)
      image=tf.expand_dims(image,0)
      image = apply_bicubic_downsample(apply_bicubic_downsample(image, filter=k, factor=2), filter=k, factor=2)
      image=tf.expand_dims(tf.squeeze(image),2)
    #  image = tf.image.resize(image, [height, width],method=tf.image.ResizeMethod.BICUBIC)
      return image
      
    def random_croptf2(image, height, width):
      cropped_image = tf.image.random_crop(image, size=[height, width,1])
      return cropped_image
      
    def normalize(image):
      image = (image / 127.5) - 1
      return image

    @tf.function
    def loadRealLR(image_file):
      imageLR = loadtf2(image_file)
      imageLR = random_croptf2(imageLR, args.fine_size, args.fine_size)
      imageLR = normalize(imageLR)
      #imageLR = tf.squeeze(tf.stack([imageLR, imageLR, imageLR],2))
      return imageLR

    @tf.function
    def loadRealHRandBC(image_file):
      imageHR = loadtf2(image_file)
      imageHR = random_croptf2(imageHR, args.fine_size*args.scale, args.fine_size*args.scale)
      imageHR = normalize(imageHR)
      imageBC = resizetf2(imageHR, args.fine_size, args.fine_size)
      #imageHR = tf.squeeze(tf.stack([imageHR, imageHR, imageHR],2))
      #imageBC = tf.squeeze(tf.stack([imageBC, imageBC, imageBC],2))
      return imageHR, imageBC

    @tf.function
    def loadRealLRTest(image_file):
      imageLR = loadtf2(image_file)
      imageLR = normalize(imageLR)
      #imageLR = tf.squeeze(tf.stack([imageLR, imageLR, imageLR],2))
      return imageLR

    @tf.function
    def loadRealHRandBCTest(image_file):
      imageHR = loadtf2(image_file)
      imageHR = normalize(imageHR)
      imageBC = resizetf2(imageHR, args.fine_size, args.fine_size)
      #imageHR = tf.squeeze(tf.stack([imageHR, imageHR, imageHR],2))
      #imageBC = tf.squeeze(tf.stack([imageBC, imageBC, imageBC],2))
      return imageHR, imageBC
      
    #realLR = loadRealLR(args.dataset_dir+'trainA/tomo00002.png')
    #realHR, synLR = loadRealHRandBC(args.dataset_dir+'trainB/tomo00002.png')

    #plt.figure()
    #plt.imshow(np.uint8((realLR+1)*127.5))
    #plt.figure()
    #plt.imshow(np.uint8((realHR+1)*127.5))
    #plt.figure()
    #plt.imshow(np.uint8((synLR+1)*127.5))
    #tempLR = tf.image.resize(realHR, [192, 192],method=tf.image.ResizeMethod.BICUBIC)
    #plt.figure()
    #plt.imshow((np.uint8((tempLR+1)*127.5)))
    #plt.show()
    #pdb.sadlfkj

    # TF dataset structure
    realLR_dataset = tf.data.Dataset.list_files(args.dataset_dir+'trainA/*.png')
    realLR_dataset = realLR_dataset.map(loadRealLR, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(args.batch_size*5).batch(args.batch_size).prefetch(args.batch_size*5)

    realHR_and_synLR_dataset = tf.data.Dataset.list_files(args.dataset_dir+'trainB/*.png')
    realHR_and_synLR_dataset = realHR_and_synLR_dataset.map(loadRealHRandBC, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(args.batch_size*5).batch(args.batch_size).prefetch(args.batch_size*5)


    realLR_dataset_test = tf.data.Dataset.list_files(args.dataset_dir+'testA/*.png')
    realLR_dataset_test = realLR_dataset_test.map(loadRealLRTest, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(args.batch_size*5)
    realHR_and_synLR_dataset_test = tf.data.Dataset.list_files(args.dataset_dir+'testB/*.png')
    realHR_and_synLR_dataset_test = realHR_and_synLR_dataset_test.map(loadRealHRandBCTest, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(args.batch_size*5)

    #sampleRealLRTrain=next(iter(realLR_dataset))
    #sampleRealHRBCTrain=next(iter(realHR_and_synLR_dataset))
    #sampleRealLRTest=next(iter(realLR_dataset_test))
    #sampleRealHRBCTest=next(iter(realHR_and_synLR_dataset_test))


    # define architecture
    class InstanceNormalization(tf.keras.layers.Layer):
      """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

      def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

      def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

      def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
        
    def res_block_EDSR(x_in, filters, norm_type='instancenorm', apply_norm=False):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x_in)
        x = tf.keras.layers.Activation('relu')(x)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x = tf.keras.layers.BatchNormalization()(x)
            elif norm_type.lower() == 'instancenorm':
                x = InstanceNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
        x = tf.keras.layers.Add()([x_in, x])
        return x
        
    def res_block_RDB(x_in, filters, norm_type='instancenorm', apply_norm=False): # residual dense block
        outFilters = filters//2
        x1 = tf.keras.layers.Conv2D(outFilters, 3, padding='same')(x_in)
        x1 = tf.keras.layers.LeakyReLU(alpha=0.2)(x1)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x1 = tf.keras.layers.BatchNormalization()(x1)
            elif norm_type.lower() == 'instancenorm':
                x1 = InstanceNormalization()(x1)
        x2 = tf.keras.layers.Concatenate()([x_in,x1])
        x2 = tf.keras.layers.Conv2D(outFilters, 3, padding='same')(x2)
        x2 = tf.keras.layers.LeakyReLU(alpha=0.2)(x2)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x2 = tf.keras.layers.BatchNormalization()(x2)
            elif norm_type.lower() == 'instancenorm':
                x2 = InstanceNormalization()(x2)
        x3 = tf.keras.layers.Concatenate()([x_in,x1,x2])
        x3 = tf.keras.layers.Conv2D(outFilters, 3, padding='same')(x3)
        x3 = tf.keras.layers.LeakyReLU(alpha=0.2)(x3)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x3 = tf.keras.layers.BatchNormalization()(x3)
            elif norm_type.lower() == 'instancenorm':
                x3 = InstanceNormalization()(x3)
        x4 = tf.keras.layers.Concatenate()([x_in,x1,x2,x3])
        x4 = tf.keras.layers.Conv2D(outFilters, 3, padding='same')(x4)
        x4 = tf.keras.layers.LeakyReLU(alpha=0.2)(x4)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x4 = tf.keras.layers.BatchNormalization()(x4)
            elif norm_type.lower() == 'instancenorm':
                x4 = InstanceNormalization()(x4)
        x5 = tf.keras.layers.Concatenate()([x_in,x1,x2,x3,x4])
        x5 = tf.keras.layers.Conv2D(filters, 3, padding='same')(x5)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x5 = tf.keras.layers.BatchNormalization()(x5)
            elif norm_type.lower() == 'instancenorm':
                x5 = InstanceNormalization()(x5)
        return x5 * 0.2 + x_in
        
    def res_block_RRDB(x_in, filters, norm_type='instancenorm', apply_norm=False): # residual in residual dense block
        x = res_block_RDB(x_in, filters, norm_type, apply_norm)
        x = res_block_RDB(x, filters, norm_type, apply_norm)
        x = res_block_RDB(x, filters, norm_type, apply_norm)
        return x * 0.2 + x_in
    
    def res_block_RFB(x_in, filters, norm_type='instancenorm', apply_norm=False): # receptive field block
        outFilters = filters//4
        
        x1 = tf.keras.layers.Conv2D(outFilters, 1, padding='same')(x_in)
        x1 = tf.keras.layers.LeakyReLU(alpha=0.2)(x1)
        x1 = tf.keras.layers.Conv2D(outFilters, 3, padding='same')(x1)
        
        x2 = tf.keras.layers.Conv2D(outFilters, 1, padding='same')(x_in)
        x2 = tf.keras.layers.LeakyReLU(alpha=0.2)(x2)
        x2 = tf.keras.layers.Conv2D(outFilters, (1,3), padding='same')(x2)
        x2 = tf.keras.layers.LeakyReLU(alpha=0.2)(x2)
        x2 = tf.keras.layers.Conv2D(outFilters, 3, padding='same',dilation_rate=3)(x2)
        
        x3 = tf.keras.layers.Conv2D(outFilters, 1, padding='same')(x_in)
        x3 = tf.keras.layers.LeakyReLU(alpha=0.2)(x3)
        x3 = tf.keras.layers.Conv2D(outFilters, (3,1), padding='same')(x3)
        x3 = tf.keras.layers.LeakyReLU(alpha=0.2)(x3)
        x3 = tf.keras.layers.Conv2D(outFilters, 3, padding='same',dilation_rate=3)(x3)
        
        x4 = tf.keras.layers.Conv2D(outFilters//2, 1, padding='same')(x_in)
        x4 = tf.keras.layers.LeakyReLU(alpha=0.2)(x4)
        x4 = tf.keras.layers.Conv2D(outFilters//4*3, (1,3), padding='same')(x4)
        x4 = tf.keras.layers.LeakyReLU(alpha=0.2)(x4)
        x4 = tf.keras.layers.Conv2D(outFilters, (3,1), padding='same')(x4)
        x4 = tf.keras.layers.LeakyReLU(alpha=0.2)(x4)
        x4 = tf.keras.layers.Conv2D(outFilters, 3, padding='same',dilation_rate=5)(x4)
        
        x5 = tf.keras.layers.Concatenate()([x1,x2,x3,x4])
        x5 = tf.keras.layers.Conv2D(filters, 1, padding='same')(x5)
        x0 = tf.keras.layers.Conv2D(filters, 1, padding='same')(x_in)
        x5 = x5 + x0
        return x5
        
    def res_block_RFDB(x_in, filters, norm_type='instancenorm', apply_norm=False): # receptive field dense block
        outFilters = filters//2
        x1 = res_block_RFB(x_in, outFilters, norm_type='instancenorm', apply_norm=False)
        x1 = tf.keras.layers.LeakyReLU(alpha=0.2)(x1)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x1 = tf.keras.layers.BatchNormalization()(x1)
            elif norm_type.lower() == 'instancenorm':
                x1 = InstanceNormalization()(x1)
        x2 = tf.keras.layers.Concatenate()([x_in,x1])
        x2 = res_block_RFB(x2, outFilters, norm_type, apply_norm)
        x2 = tf.keras.layers.LeakyReLU(alpha=0.2)(x2)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x2 = tf.keras.layers.BatchNormalization()(x2)
            elif norm_type.lower() == 'instancenorm':
                x2 = InstanceNormalization()(x2)
        x3 = tf.keras.layers.Concatenate()([x_in,x1,x2])
        x3 = res_block_RFB(x3, outFilters, norm_type, apply_norm)
        x3 = tf.keras.layers.LeakyReLU(alpha=0.2)(x3)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x3 = tf.keras.layers.BatchNormalization()(x3)
            elif norm_type.lower() == 'instancenorm':
                x3 = InstanceNormalization()(x3)
        x4 = tf.keras.layers.Concatenate()([x_in,x1,x2,x3])
        x4 = res_block_RFB(x4, outFilters, norm_type, apply_norm)
        x4 = tf.keras.layers.LeakyReLU(alpha=0.2)(x4)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x4 = tf.keras.layers.BatchNormalization()(x4)
            elif norm_type.lower() == 'instancenorm':
                x4 = InstanceNormalization()(x4)
        x5 = tf.keras.layers.Concatenate()([x_in,x1,x2,x3,x4])
        x5 = res_block_RFB(x5, filters, norm_type, apply_norm)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x5 = tf.keras.layers.BatchNormalization()(x5)
            elif norm_type.lower() == 'instancenorm':
                x5 = InstanceNormalization()(x5)
        return x5 * 0.2 + x_in
        
    def res_block_RRFDB(x_in, filters, norm_type='instancenorm', apply_norm=False): # residual receptive field dense block
        x = res_block_RFDB(x_in, filters, norm_type, apply_norm)
        x = res_block_RFDB(x, filters, norm_type, apply_norm)
        x = res_block_RFDB(x, filters, norm_type, apply_norm)
        return x * 0.2 + x_in
        
    def upsampleRRFBRRDB(x, scale, num_filters, norm_type='instancenorm', apply_norm=False):
        def upsample_RFB(x, factor, **kwargs):
        
            x = res_block_RFB(x, num_filters, norm_type, apply_norm)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            
            if apply_norm:
                if norm_type.lower() == 'batchnorm':
                    x = tf.keras.layers.BatchNormalization()(x)
                elif norm_type.lower() == 'instancenorm':
                    x = InstanceNormalization()(x)
            return SubpixelConv2D(factor)(x)
        if scale == 2:
            x = upsample_RFB(x, 2, name='conv2d_1_scale_2_up')
        elif scale == 3:
            x = upsample_RFB(x, 3, name='conv2d_1_scale_3_up')
        elif scale == 4:
            x = upsample_RFB(x, 2, name='conv2d_1_scale_2_up')
            x = upsample_RFB(x, 2, name='conv2d_2_scale_2_up')
        elif scale == 8:
            x = upsample_RFB(x, 2, name='conv2d_1_scale_2_up')
            x = upsample_RFB(x, 2, name='conv2d_2_scale_2_up')
            x = upsample_RFB(x, 2, name='conv2d_3_scale_2_up')
        return x
        
    def upsampleEDSR(x, scale, num_filters, norm_type='instancenorm', apply_norm=False):
        def upsample_edsr(x, factor, **kwargs):
            x = tf.keras.layers.Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
            x = tf.keras.layers.Activation('relu')(x)
            if apply_norm:
                if norm_type.lower() == 'batchnorm':
                    x = tf.keras.layers.BatchNormalization()(x)
                elif norm_type.lower() == 'instancenorm':
                    x = InstanceNormalization()(x)
            return SubpixelConv2D(factor)(x)
        if scale == 2:
            x = upsample_edsr(x, 2, name='conv2d_1_scale_2_up')
        elif scale == 3:
            x = upsample_edsr(x, 3, name='conv2d_1_scale_3_up')
        elif scale == 4:
            x = upsample_edsr(x, 2, name='conv2d_1_scale_2_up')
            x = upsample_edsr(x, 2, name='conv2d_2_scale_2_up')
        elif scale == 8:
            x = upsample_edsr(x, 2, name='conv2d_1_scale_2_up')
            x = upsample_edsr(x, 2, name='conv2d_2_scale_2_up')
            x = upsample_edsr(x, 2, name='conv2d_3_scale_2_up')
        return x
        
    def downsampleEDSR(x, scale, num_filters, norm_type='instancenorm', apply_norm=False):
        def downsample_edsr(x, factor, **kwargs):
            x = SubpixelConv2DDown(factor)(x)
            x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', **kwargs)(x)
            x = tf.keras.layers.Activation('relu')(x)
            if apply_norm:
                if norm_type.lower() == 'batchnorm':
                    x = tf.keras.layers.BatchNormalization()(x)
                elif norm_type.lower() == 'instancenorm':
                    x = InstanceNormalization()(x)
            return x
        if scale == 2:
            x = downsample_edsr(x, 2, name='conv2d_1_scale_2_down')
        elif scale == 3:
            x = downsample_edsr(x, 3, name='conv2d_1_scale_3_down')
        elif scale == 4:
            x = downsample_edsr(x, 2, name='conv2d_1_scale_2_down')
            x = downsample_edsr(x, 2, name='conv2d_2_scale_2_down')
        elif scale == 8:
            x = downsample_edsr(x, 2, name='conv2d_1_scale_2_down')
            x = downsample_edsr(x, 2, name='conv2d_2_scale_2_down')
            x = downsample_edsr(x, 2, name='conv2d_3_scale_2_down')
        return x
        
    def downsample(filters, size, norm_type='instancenorm', apply_norm=False):
      """Downsamples an input.
      Conv2D => Batchnorm => LeakyRelu
      Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
        apply_norm: If True, adds the batchnorm layer
      Returns:
        Downsample Sequential Model
      """
      initializer = tf.random_normal_initializer(0., 0.02)

      result = tf.keras.Sequential()
      result.add(
          tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
      result.add(tf.keras.layers.LeakyReLU())
      if apply_norm:
        if norm_type.lower() == 'batchnorm':
          result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
          result.add(InstanceNormalization())
      return result
      
    def cyclegan_generator(args):
        x_in = tf.keras.layers.Input(shape=(None, None, 1))
        x = x_in
        x = b = downsampleEDSR(x, 4, args.ngf, norm_type='instancenorm', apply_norm=False)
        for i in range(8):
            b = res_block_EDSR(b, args.ngf, norm_type='instancenorm', apply_norm=False)
        b = tf.keras.layers.Conv2D(args.ngf, 3, padding='same')(b)
        x = tf.keras.layers.Concatenate()([x, b])
        x = upsampleEDSR(x, 4, args.ngf, norm_type='instancenorm', apply_norm=False)
        x = tf.keras.layers.Conv2D(1, 3, padding='same')(x)

        x = tf.keras.layers.Activation('tanh', dtype='float32')(x)
        return tf.keras.models.Model(x_in, x, name="cycleganGenerator")


    def discriminatorCGAN(norm_type='instancenorm', target=False):
      """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
      Args:
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
        target: Bool, indicating whether target image is an input or not.
      Returns:
        Discriminator model
      """

      initializer = tf.random_normal_initializer(0., 0.02)

      inp = tf.keras.layers.Input(shape=[None, None, 1], name='input_image')
      x = inp

      if target:
        tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
        x = tf.keras.layers.Concatenate()([inp, tar])  # (bs, 256, 256, channels*2)

      down1 = downsample(args.ndf, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
      down2 = downsample(args.ndf*2, 4, norm_type)(down1)  # (bs, 64, 64, 128)
      down3 = downsample(args.ndf*4, 4, norm_type)(down2)  # (bs, 32, 32, 256)

      zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
      conv = tf.keras.layers.Conv2D(
          args.ngf*8, 4, strides=1, kernel_initializer=initializer,
          use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

      if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
      elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)

      leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

      zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

      last = tf.keras.layers.Conv2D(
          1, 4, strides=1,
          kernel_initializer=initializer, dtype='float32')(zero_pad2)  # (bs, 30, 30, 1)

      if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last, name="DiscrimCycle")
      else:
        return tf.keras.Model(inputs=inp, outputs=last, name="DiscrimCycle")

    def disc_block(x_in, filters):
        x = tf.keras.layers.Conv2D(filters, 3, 1, padding='same')(x_in)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(filters, 3, 2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return x
        

    def DiscriminatorSRGAN(args):
        #xIn = tf.keras.layers.Input(shape=[args.fine_size, args.fine_size, args.output_nc], name='Disc_Inputs')
        xIn = tf.keras.layers.Input(shape=[args.disc_size, args.disc_size, 1], name='Disc_Inputs')
        # shallow layers
        x = tf.keras.layers.Conv2D(args.ndf, 3, 1, padding='same')(xIn)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(args.ndf, 3, 2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        numDiscBlocks=3
        for i in range(numDiscBlocks):
            x = disc_block(x, args.ndf*(2**(i+1)))
        #xOut = tf.keras.layers.Conv2D(1, 3, 1, padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        xOut = tf.keras.layers.Dense(1, dtype='float32')(x)
        '''
        h = lrelu(conv2d(image, options.df_dim, ks=3, s=1, name='dInitConv'))
        h = lrelu(batchnormSR(conv2d(h, options.df_dim, ks=3, s=s, name='dUpConv')))
        for i in range(numDiscBlocks):
            expon=2**(i+1)
            h = lrelu(batchnormSR(conv2d(h, options.df_dim*expon, ks=3, s=1, name=f'dBlock{i+1}Conv')))
            h = lrelu(batchnormSR(conv2d(h, options.df_dim*expon, ks=3, s=2, name=f'dBlock{i+1}UpConv')))
        h = conv2d(h, 1, ks=3, s=1, name='d_h3_pred')
        #h = lrelu(denselayer(slim.flatten(h), 1024, name="dFC1"))
        #h = denselayer(h, 1, name="dFCout")
        return h
        
        '''  
        return tf.keras.Model(inputs=[xIn], outputs=xOut, name="DiscrimSR")
        
    def SubpixelConv2D(scale, **kwargs):
        return  tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale), **kwargs)
        
    def SubpixelConv2DDown(scale, **kwargs):
        return  tf.keras.layers.Lambda(lambda x: tf.nn.space_to_depth(x, scale), **kwargs)

    def edsr(scale, num_filters=64, num_res_blocks=8):
        x_in = tf.keras.layers.Input(shape=(None, None, 1))
        x = x_in
        x = b = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
        for i in range(num_res_blocks):
            b = res_block_EDSR(b, num_filters, norm_type='instancenorm', apply_norm=False)
        b = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(b)
        x = tf.keras.layers.Add()([x, b])

        x = upsampleEDSR(x, scale, num_filters, norm_type='instancenorm', apply_norm=False)
        
        x = tf.keras.layers.Conv2D(1, 3, padding='same')(x)
        x = tf.keras.layers.Activation('tanh', dtype='float32')(x)
        
        return tf.keras.models.Model(x_in, x, name="EDSR")

    def RRFDB_RRDB_SRGAN(scale, num_filters=64, num_res_blocks=8, num_res_rfb_blocks=8):
        x_in = tf.keras.layers.Input(shape=(None, None, 1))
        x = x_in
        x = b = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
        for i in range(num_res_blocks):
            b = res_block_RRDB(b, num_filters, norm_type='instancenorm', apply_norm=False)
        for i in range(num_res_rfb_blocks):
            b = res_block_RRFDB(b, num_filters, norm_type='instancenorm', apply_norm=False)
        x = tf.keras.layers.Add()([x, b])
        x = res_block_RFB(x, num_filters, norm_type='instancenorm', apply_norm=False)
        x = upsampleRRFBRRDB(x, scale, num_filters, norm_type='instancenorm', apply_norm=False)
        x = res_block_RFB(x, num_filters, norm_type='instancenorm', apply_norm=False)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Conv2D(1, 3, padding='same')(x)
        x = tf.keras.layers.Activation('tanh', dtype='float32')(x)
        return tf.keras.models.Model(x_in, x, name="RRFDB-RRDB")
        
    def RRDB_SRGAN(scale, num_filters=64, num_res_blocks=8):
        x_in = tf.keras.layers.Input(shape=(None, None, 1))
        x = x_in
        x = b = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
        for i in range(num_res_blocks):
            b = res_block_RRDB(b, num_filters, norm_type='instancenorm', apply_norm=False)
            
        b = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(b)
        x = tf.keras.layers.Add()([x, b])

        x = upsampleEDSR(x, scale, num_filters, norm_type='instancenorm', apply_norm=False)
        x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(1, 3, padding='same')(x)
        x = tf.keras.layers.Activation('tanh', dtype='float32')(x)
        return tf.keras.models.Model(x_in, x, name="RRFDB-RRDB")

    #generatorAB = cyclegan_generator(args)
    #generatorBA = cyclegan_generator(args)

    #discriminatorA = discriminator(norm_type='instancenorm', target=False)
    #discriminatorB = discriminator(norm_type='instancenorm', target=False)

    #generatorSR = edsr_generator(scale=4, num_filters=64, num_res_blocks=16)
    #discriminatorSR = DiscriminatorSRGAN(args)

    #tf.keras.utils.plot_model(generatorAB, to_file='cyclegan.png', show_shapes=True, dpi=64)
    #tf.keras.utils.plot_model(discriminatorA, to_file='lsgan.png', show_shapes=True, dpi=64)
    #tf.keras.utils.plot_model(generatorSR, to_file='edsr.png', show_shapes=True, dpi=64)
    #tf.keras.utils.plot_model(discriminatorSR, to_file='scganTL.png', show_shapes=True, dpi=64)

    # final sanity checks for IO and network design
    #sampleRealLRTrain=next(iter(realLR_dataset))
    #sampleRealHRBCTrain=next(iter(realHR_and_synLR_dataset))
    #sampleRealLRTest=next(iter(realLR_dataset_test))
    #sampleRealHRBCTest=next(iter(realHR_and_synLR_dataset_test))

    #plt.subplot(131)
    #plt.title('realLR')
    #plt.imshow(np.squeeze(sampleRealLRTest))

    #plt.subplot(132)
    #plt.title('synLR')
    #plt.imshow(np.squeeze(sampleRealHRBCTest[1]))

    #plt.subplot(133)
    #plt.title('realHR')
    #plt.imshow(np.squeeze(sampleRealHRBCTest[0]))


    # standard losses
    def meanAbsoluteError(labels, predictions):
        per_example_loss = tf.reduce_mean(tf.abs(labels-predictions), axis = [1,2])
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=args.batch_size)
   
    def meanSquaredError(labels, predictions):
        per_example_MSE = tf.reduce_mean(((labels-predictions))**2, axis = [1,2])
        return tf.nn.compute_average_loss(per_example_MSE, global_batch_size=args.batch_size)
       
    def sigmoidCrossEntropy(labels, logits):
        per_example_sxe = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), axis = 1)
        return tf.nn.compute_average_loss(per_example_sxe, global_batch_size=args.batch_size)
        
    def colourGLCMLossL2(labels, predictions):
        numLevels = 8
        span = 1#scaleFactor
        per_example_glcm=0
        for i in range(args.output_nc):
            per_example_glcm += tf.reduce_mean((compute8WayGLCM(labels[:,:,:,i], numLevels, span) - compute8WayGLCM(predictions[:,:,:,i], numLevels, span))**2)
        return tf.nn.compute_average_loss(per_example_glcm, global_batch_size=args.batch_size)
        
    # special gan losses
    def lsganLoss(disc_real_output, disc_generated_output):
        real_loss = meanSquaredError(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = meanSquaredError(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss # optimal foolery is when this equals 1, 50 real, 50 fake
        return 0.5*total_disc_loss

    def rellsganLoss(disc_real_output, disc_generated_output):
        real_loss = meanSquaredError(tf.ones_like(disc_real_output), disc_real_output-tf.reduce_mean(disc_generated_output))
        generated_loss = meanSquaredError(tf.zeros_like(disc_generated_output), disc_generated_output-tf.reduce_mean(disc_real_output))
        total_disc_loss = real_loss + generated_loss # optimal foolery is when this equals 1, 50 real, 50 fake
        return 0.5*total_disc_loss

    def advLsganLoss(disc_generated_output):
        adversarial_loss = meanSquaredError(tf.ones_like(disc_generated_output), disc_generated_output)
        return adversarial_loss
        
    def reladvLsganLoss(disc_real_output, disc_generated_output):
        real_loss = meanSquaredError(tf.zeros_like(disc_real_output), disc_real_output-tf.reduce_mean(disc_generated_output))
        generated_loss = meanSquaredError(tf.ones_like(disc_generated_output), disc_generated_output-tf.reduce_mean(disc_real_output))
        total_disc_loss = real_loss + generated_loss # optimal foolery is when this equals 1, 50 real, 50 fake
        return 0.5*total_disc_loss
#        adversarial_loss = meanSquaredError(tf.ones_like(disc_generated_output), disc_generated_output)
#        return adversarial_loss

    def scganLoss(disc_real_output, disc_generated_output):
        real_loss = sigmoidCrossEntropy(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = sigmoidCrossEntropy(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss # optimal foolery is when this equals 1, 50 real, 50 fake
        return 0.5*total_disc_loss
        
    def advScganLoss(disc_generated_output):
        adversarial_loss = sigmoidCrossEntropy(tf.ones_like(disc_generated_output), disc_generated_output)
        return adversarial_loss
        
    def rel_scganLoss(disc_real_output, disc_generated_output):
    
        real_loss = sigmoidCrossEntropy(tf.ones_like(disc_real_output), disc_real_output-tf.reduce_mean(disc_generated_output))
        generated_loss = sigmoidCrossEntropy(tf.zeros_like(disc_generated_output), disc_generated_output-tf.reduce_mean(disc_real_output))
        
        total_disc_loss = 0.5*(real_loss + generated_loss) # optimal foolery is when this equals 1, 50 real, 50 fake
        return total_disc_loss
        
    def rel_advScganLoss(disc_real_output, disc_generated_output):
    
        real_loss = sigmoidCrossEntropy(tf.ones_like(disc_generated_output), disc_generated_output-tf.reduce_mean(disc_real_output))
        generated_loss = sigmoidCrossEntropy(tf.zeros_like(disc_real_output), disc_real_output-tf.reduce_mean(disc_generated_output))
        
        adversarial_loss =  0.5*(real_loss + generated_loss)
        return adversarial_loss
        
    def createGenerator(args):
        # model and optimizer must be created under `strategy.scope`.
        generator = cyclegan_generator(args)
        generator.summary(200)
        optimizerGenerator = tf.keras.optimizers.Adam(lr=args.lr)
        optimizerGenerator = mixed_precision.LossScaleOptimizer(optimizerGenerator, loss_scale='dynamic')

        return generator, optimizerGenerator
        
    def createDiscriminator(args):    
        discriminator = discriminatorCGAN(norm_type='instancenorm', target=False)
        discriminator.summary(200)
        optimizerDiscriminator = tf.keras.optimizers.Adam(lr=args.lr)    
        optimizerDiscriminator = mixed_precision.LossScaleOptimizer(optimizerDiscriminator, loss_scale='dynamic')     
        return discriminator, optimizerDiscriminator
    
    def createSRGenerator(args):
        if args.generatorType == 'edsr':
            generator = edsr(scale=4, num_filters=args.ngsrf, num_res_blocks=args.numResBlocks)
        elif args.generatorType == 'rrfdb-rrdb':
            generator = RRFDB_RRDB_SRGAN(scale=4, num_filters=args.ngsrf, num_res_blocks=args.numResBlocks, num_res_rfb_blocks=args.numResRFBBlocks)
        elif args.generatorType == 'rrdb':
            generator = RRDB_SRGAN(scale=4, num_filters=args.ngsrf, num_res_blocks=args.numResBlocks)
        generator.summary(200)
        optimizerGenerator = tf.keras.optimizers.Adam(lr=args.lr)
        optimizerGenerator = mixed_precision.LossScaleOptimizer(optimizerGenerator, loss_scale='dynamic')
        return generator, optimizerGenerator
    
    def createSRDiscriminator(args):
        if args.ganFlag:
            discriminator = DiscriminatorSRGAN(args)
            optimizerDiscriminator = tf.keras.optimizers.Adam(lr=args.lr)         
        else:
            a = tf.keras.layers.Input(shape=(1,))
            b = a
            discriminator = tf.keras.models.Model(inputs=a, outputs=b)
            optimizerDiscriminator = tf.keras.optimizers.Adam(lr=args.lr)                
        optimizerDiscriminator = mixed_precision.LossScaleOptimizer(optimizerDiscriminator, loss_scale='dynamic')     
        discriminator.summary(200)
        return discriminator, optimizerDiscriminator
    
    generatorAB, optimizerGeneratorAB = createGenerator(args)
    generatorBA, optimizerGeneratorBA = createGenerator(args)
    discriminatorA, optimizerDiscriminatorA = createDiscriminator(args)
    discriminatorB, optimizerDiscriminatorB = createDiscriminator(args)
    generatorSR, optimizerGeneratorSR = createSRGenerator(args)
    discriminatorSR, optimizerDiscriminatorSR = createSRDiscriminator(args)
    
    if args.sigmaType == 'gamma':
        generatorSRC, optimizerGeneratorSRC = createSRGenerator(args)
        discriminatorSRC, optimizerDiscriminatorSRC = createSRDiscriminator(args)
    
    # define the actions taken per iteration (calc grads and make an optim step)
    def train_step(realLRBatch, HRandBCBatch):
        A = realLRBatch
        C, B =  HRandBCBatch
        # train
        with tf.GradientTape(persistent=True) as tape:
            # run a cycle on the cycleGAN
            B_fake = generatorAB(A, training = True)
            A_cycle = generatorBA(B_fake, training = True)
            
            A_fake = generatorBA(B, training = True)
            B_cycle = generatorAB(A_fake, training = True)
            
            # run the identity through the generators
            A_same = generatorBA(A, training = True)
            B_same = generatorAB(B, training = True)
            
            # get the discriminator logits
            disc_real_A = discriminatorA(A, training=True)
            disc_fake_A = discriminatorA(A_fake, training=True)
            disc_real_B = discriminatorB(B, training=True)
            disc_fake_B = discriminatorB(B_fake, training=True)
                       
            ## calculate the cycle and idt loss (pixelwise)
            if args.cyclePixelwiseLoss == 'L1':
                cycleLoss = meanAbsoluteError(A, A_cycle) + meanAbsoluteError(B, B_cycle) #mse would also work here
                idtABLoss = meanAbsoluteError(A, A_same)
                idtBALoss = meanAbsoluteError(B, B_same)
            elif args.cyclePixelwiseLoss == 'L2':
                cycleLoss = meanSquaredError(A, A_cycle) + meanSquaredError(B, B_cycle) #mae would also work here
                idtABLoss = meanSquaredError(A, A_same)
                idtBALoss = meanSquaredError(B, B_same)
                
            ## calculate the adversarial/discriminator losses (cross entropy)
            if args.cycleDiscLoss == 'LS':
                advABLoss = advLsganLoss(disc_fake_B)
                advBALoss = advLsganLoss(disc_fake_A)
                discALoss = lsganLoss(disc_real_A, disc_fake_A)
                discBLoss = lsganLoss(disc_real_B, disc_fake_B)
            elif args.cycleDiscLoss == 'SC':
                advABLoss = advScganLoss(disc_fake_B)
                advBALoss = advScganLoss(disc_fake_A)
                discALoss = scganLoss(disc_real_A, disc_fake_A)
                discBLoss = scganLoss(disc_real_B, disc_fake_B)
            elif args.cycleDiscLoss == 'RelSC':
                advABLoss = rel_advScganLoss(disc_real_B, disc_fake_B)
                advBALoss = rel_advScganLoss(disc_real_A, disc_fake_A)
                discALoss = rel_scganLoss(disc_real_A, disc_fake_A)
                discBLoss = rel_scganLoss(disc_real_B, disc_fake_B)
            elif args.cycleDiscLoss == 'RelLS':
                advABLoss = reladvLsganLoss(disc_real_B, disc_fake_B)
                advBALoss = reladvLsganLoss(disc_real_A, disc_fake_A)
                discALoss = rellsganLoss(disc_real_A, disc_fake_A)
                discBLoss = rellsganLoss(disc_real_B, disc_fake_B)
            totalGABLoss = cycleLoss + 0.5*idtABLoss + args.cycleAdv_lambda*advABLoss
            totalGBALoss = cycleLoss + 0.5*idtBALoss + args.cycleAdv_lambda*advBALoss
            
            totalDALoss = discALoss
            totalDBLoss = discBLoss
            
            ## run and calculate sr losses
            # run the SRGAN in sigma mode
            if args.sigmaType == 'sigma' or args.sigmaType == 'omega' or args.sigmaType == 'gamma':
                C_sr = generatorSR(A_fake, training=True)
            elif args.sigmaType == 'delta':
                C_sr = generatorSR(B, training=True)
            
            if args.srPixelwiseLoss == 'L1':
                gsrLoss = meanAbsoluteError(C, C_sr)
            elif args.srPixelwiseLoss == 'L2':
                gsrLoss = meanSquaredError(C, C_sr)
                
            if args.ganFlag:
                disc_real_C = discriminatorSR(C[:,0:args.disc_size,0:args.disc_size,:], training=True)
                disc_fake_C = discriminatorSR(C_sr[:,0:args.disc_size,0:args.disc_size,:], training=True)

                if args.srDiscLoss == 'LS':
                    advsrLoss = advLsganLoss(disc_fake_C)
                    dsrLoss = lsganLoss(disc_real_C, disc_fake_C)
                elif args.srDiscLoss == 'SC':
                    advsrLoss = advScganLoss(disc_fake_C)
                    dsrLoss = scganLoss(disc_real_C, disc_fake_C)
                elif args.srDiscLoss == 'RelSC':
                    advsrLoss = rel_advScganLoss(disc_real_C, disc_fake_C)
                    dsrLoss = rel_scganLoss(disc_real_C, disc_fake_C)
            else:
                advsrLoss = 0
                dsrLoss = 0
           
           ## if omega, also pass the clean image through SR and the cycle clean image and calculate extra losses
            if args.sigmaType == 'omega':
                C_clean = generatorSR(B, training=True)
                if args.srPixelwiseLoss == 'L1':
                    gsrLoss = gsrLoss + meanAbsoluteError(C, C_clean)
                elif args.srPixelwiseLoss == 'L2':
                    gsrLoss = gsrLoss + meanSquaredError(C, C_clean)                
                if args.ganFlag:
                    disc_fake_CC = discriminatorSR(C_clean[:,0:args.disc_size,0:args.disc_size,:], training=True)

                    if args.srDiscLoss == 'LS':
                        advsrLoss = advsrLoss + advLsganLoss(disc_fake_CC)
                        dsrLoss = dsrLoss + lsganLoss(disc_real_C, disc_fake_CC)
                    elif args.srDiscLoss == 'SC':
                        advsrLoss = advsrLoss + advScganLoss(disc_fake_CC)
                        dsrLoss = dsrLoss + scganLoss(disc_real_C, disc_fake_CC)
                    elif args.srDiscLoss == 'RelSC':
                        advsrLoss = advsrLoss + rel_advScganLoss(disc_real_C, disc_fake_CC)
                        dsrLoss = dsrLoss + rel_scganLoss(disc_real_C, disc_fake_CC)
                else:
                    advsrLoss = 0
                    dsrLoss = 0
                    
                C_clean_cycle = generatorSR(B_cycle, training=True)
                if args.srPixelwiseLoss == 'L1':
                    gsrLoss = gsrLoss + meanAbsoluteError(C, C_clean_cycle)
                elif args.srPixelwiseLoss == 'L2':
                    gsrLoss = gsrLoss + meanSquaredError(C, C_clean_cycle)                
                if args.ganFlag:
                    disc_fake_CCC = discriminatorSR(C_clean_cycle[:,0:args.disc_size,0:args.disc_size,:], training=True)

                    if args.srDiscLoss == 'LS':
                        advsrLoss = advsrLoss + advLsganLoss(disc_fake_CCC)
                        dsrLoss = dsrLoss + lsganLoss(disc_real_C, disc_fake_CCC)
                    elif args.srDiscLoss == 'SC':
                        advsrLoss = advsrLoss + advScganLoss(disc_fake_CCC)
                        dsrLoss = dsrLoss + scganLoss(disc_real_C, disc_fake_CCC)
                    elif args.srDiscLoss == 'RelSC':
                        advsrLoss = advsrLoss + rel_advScganLoss(disc_real_C, disc_fake_CCC)
                        dsrLoss = dsrLoss + rel_scganLoss(disc_real_C, disc_fake_CCC)
                else:
                    advsrLoss = 0
                    dsrLoss = 0
           
            totalGsrLoss = gsrLoss + args.srAdv_lambda*advsrLoss
            totalDsrLoss = dsrLoss
            
            ## sigma coupling
            if args.sigmaCouplingFlag:
                totalGABLoss = totalGABLoss + args.sigmaCoupling_lambda*totalGsrLoss
                totalGBALoss = totalGBALoss + args.sigmaCoupling_lambda*totalGsrLoss


            totalGABLossScal = optimizerGeneratorAB.get_scaled_loss(totalGABLoss)
            totalGBALossScal = optimizerGeneratorBA.get_scaled_loss(totalGBALoss)
            totalDALossScal = optimizerDiscriminatorA.get_scaled_loss(totalDALoss)
            totalDBLossScal = optimizerDiscriminatorB.get_scaled_loss(totalDBLoss)
            totalGsrLossScal = optimizerGeneratorSR.get_scaled_loss(totalGsrLoss)
            if args.ganFlag:
                totalDsrLossScal = optimizerDiscriminatorSR.get_scaled_loss(totalDsrLoss)
        # calculate gradients
        gradGAB = tape.gradient(totalGABLossScal, generatorAB.trainable_variables)
        gradGBA = tape.gradient(totalGBALossScal, generatorBA.trainable_variables)
        gradDA = tape.gradient(totalDALossScal, discriminatorA.trainable_variables)
        gradDB = tape.gradient(totalDBLossScal, discriminatorB.trainable_variables)
        gradGsr = tape.gradient(totalGsrLossScal, generatorSR.trainable_variables)
        if args.ganFlag:
            gradDsr = tape.gradient(totalDsrLossScal, discriminatorSR.trainable_variables)
        # unscale gradients
        gradGAB = optimizerGeneratorAB.get_unscaled_gradients(gradGAB)
        gradGBA = optimizerGeneratorBA.get_unscaled_gradients(gradGBA)
        gradDA = optimizerDiscriminatorA.get_unscaled_gradients(gradDA)
        gradDB = optimizerDiscriminatorB.get_unscaled_gradients(gradDB)
        gradGsr = optimizerGeneratorSR.get_unscaled_gradients(gradGsr)
        if args.ganFlag:
            gradDsr = optimizerDiscriminatorSR.get_unscaled_gradients(gradDsr)
        # apply gradients
        optimizerGeneratorAB.apply_gradients(zip(gradGAB,generatorAB.trainable_variables))
        optimizerGeneratorBA.apply_gradients(zip(gradGBA,generatorBA.trainable_variables))
        optimizerDiscriminatorA.apply_gradients(zip(gradDA,discriminatorA.trainable_variables))
        optimizerDiscriminatorB.apply_gradients(zip(gradDB,discriminatorB.trainable_variables))
        optimizerGeneratorSR.apply_gradients(zip(gradGsr,generatorSR.trainable_variables))
        if args.ganFlag:
            optimizerDiscriminatorSR.apply_gradients(zip(gradDsr,discriminatorSR.trainable_variables))

        return totalGABLoss, advABLoss, totalGBALoss, advBALoss, totalDALoss, totalDBLoss, totalGsrLoss, advsrLoss, totalDsrLoss

    @tf.function
    def distributed_train_step(realLRBatch, HRandBCBatch):
        PRGABL, PRADVABL, PRGBAL, PRADVBAL, PRDAL, PRDBL, PRGSRL, PRADVSRL, PRDSRL = strategy.run(train_step, args=(realLRBatch, HRandBCBatch,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, PRGABL, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, PRADVABL, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, PRGBAL, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, PRADVBAL, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, PRDAL, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, PRDBL, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, PRGSRL, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, PRADVSRL, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, PRDSRL, axis=None)

    trainingDir=f"./{args.checkpoint_dir}/{args.modelName}/"
    if args.continue_train or args.phase == 'test': # restore the weights if requested, or if testing
        print(f'Loading checkpoints from {trainingDir}')
        generatorAB=tf.keras.models.load_model(f'{trainingDir}/GAB')
        generatorBA=tf.keras.models.load_model(f'{trainingDir}/GBA')
        discriminatorA=tf.keras.models.load_model(f'{trainingDir}/DA')
        discriminatorB=tf.keras.models.load_model(f'{trainingDir}/DB')
        generatorSR=tf.keras.models.load_model(f'{trainingDir}/GSR')
        if args.ganFlag:
            discriminatorSR=tf.keras.models.load_model(f'{trainingDir}/DSR')
    # run
    if args.phase == 'train':
        EPOCHS = args.epoch
        valoutDir = args.dataset_dir.split('/')[-2]
        # Create a checkpoint directory to store the checkpoints.
        rightNow=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        trainOutputDir=f'./training_outputs/{rightNow}-distNN-{valoutDir}/'
        if not os.path.exists(trainingDir):
            os.mkdir(trainingDir)
        os.mkdir(trainOutputDir)
        train_template = ("\rEpoch %4d, Iter %4d, Time %4.4f, Speed %4.4f its/s, GABL: %4.4f, ADVABL: %4.4f, GBAL: %4.4f, ADVBAL: %4.4f, DAL: %4.4f, DBL: %4.4f, GSRL: %4.4f, ADVSRL: %4.4f, DSRL: %4.4f")
        test_template = ("\rIter %4d, Test PSNR-A: %4.4f, PSNR-B: %4.4f, PSNR-SR: %4.4f, PSNR-SRC: %4.4f, PSNR-SRCC: %4.4f")
        
        realLR_dataset_dist = strategy.experimental_distribute_dataset(realLR_dataset)
        realHR_and_synLR_dataset_dist = strategy.experimental_distribute_dataset(realHR_and_synLR_dataset)
#        realLR_dataset_test
#        realHR_and_synLR_dataset_test  
        start_time = time.time()
        for epoch in range(EPOCHS):
            # TRAIN LOOP
            lastTime=time.time()
            if args.lrType == 'halfLife':
                lr=args.lr * 0.5**(epoch/args.epoch_step) # add cosine annealing later
                optimizerGeneratorAB.learning_rate = lr
                optimizerGeneratorBA.learning_rate = lr
                optimizerDiscriminatorA.learning_rate = lr
                optimizerDiscriminatorB.learning_rate = lr
                optimizerGeneratorSR.learning_rate = lr
                optimizerDiscriminatorSR.learning_rate = lr
            totGABL = 0
            totADVABL = 0
            totGBAL = 0
            totADVBAL = 0
            totDAL = 0
            totDBL = 0
            totGSRL = 0
            totADVSRL = 0
            totDSRL = 0
            num_batches = 0
            print(f'Learning Rate: {lr:.4e}')
            for x, y in zip(realLR_dataset_dist, realHR_and_synLR_dataset_dist):
                GABL, ADVABL, GBAL, ADVBAL, DAL, DBL, GSRL, ADVSRL, DSRL = distributed_train_step(x,y)
                totGABL += GABL
                totGBAL += GBAL
                totDAL += DAL
                totDBL += DBL
                totGSRL += GSRL
                totDSRL += DSRL
                totADVABL += ADVABL
                totADVBAL += ADVBAL
                totADVSRL += ADVSRL
                num_batches += 1
                currentTime=time.time()
                stdout.write(train_template % (epoch+1, num_batches, currentTime-start_time, 1/(currentTime-lastTime),GABL, ADVABL, GBAL, ADVBAL, DAL, DBL, GSRL, ADVSRL, DSRL))
                stdout.flush()
                lastTime=currentTime
                if num_batches==args.iterNum:
                    break
                
            stdout.write("\n")
            totGABL /= num_batches
            totGBAL /= num_batches
            totDAL /= num_batches
            totDBL /= num_batches
            totGSRL /= num_batches
            totDSRL /= num_batches
            totADVABL /= num_batches
            totADVBAL /= num_batches
            totADVSRL /= num_batches
            print('Mean Epoch Performance: GABL: %4.4f, ADVABL: %4.4f, GBAL: %4.4f, ADVBAL: %4.4f, DAL: %4.4f, DBL: %4.4f, GSRL: %4.4f, ADVSRL: %4.4f, DSRL: %4.4f' % (totGABL, totADVABL, totGBAL, totADVBAL, totDAL, totDBL, totGSRL, totADVSRL, totDSRL))
            
            if np.mod(epoch+1, args.print_freq) == 0 or epoch == 0:
                # validation LOOP
                valPSNRA=0.0
                valPSNRB=0.0
                valPSNRC=0.0
                valPSNRCC=0.0
                valPSNRCCC=0.0
                
                numTestBatches=0
                os.mkdir(f'./{trainOutputDir}/epoch-{epoch+1}/')
                for A, BC in zip(realLR_dataset_test, realHR_and_synLR_dataset_test):
                    B = BC[1]
                    C = BC[0]
                    A = A[:,0:A.shape[1]//4*4,0:A.shape[2]//4*4,:]
                    B = B[:,0:B.shape[1]//4*4,0:B.shape[2]//4*4,:]
                    C = C[:,0:C.shape[1]//16*16,0:C.shape[2]//16*16,:]

                    fakeB = generatorAB(A, training=False)
                    cycleA = generatorBA(fakeB, training=False) 
                    fakeA = generatorBA(B, training=False)
                    cycleB = generatorAB(fakeA,training=False)
                    

                    fakeC = generatorSR(fakeA, training=False)
                    fakeC_clean = generatorSR(B, training=False)
                    fakeC_clean_cycle = generatorSR(cycleB, training=False)
                    A = np.asarray(A)
                    B = np.asarray(B)
                    C = np.asarray(C)
                    
                    fakeB = np.asarray(fakeB)
                    cycleA = np.asarray(cycleA)
                    fakeA = np.asarray(fakeA)
                    cycleB = np.asarray(cycleB)
                    fakeC = np.asarray(fakeC)
                    fakeC_clean = np.asarray(fakeC_clean)
                    fakeC_clean_cycle = np.asarray(fakeC_clean_cycle)
                    
                    psnrA=10*np.log10(2*2/np.mean((A-cycleA)**2))
                    psnrB=10*np.log10(2*2/np.mean((B-cycleB)**2))
                    psnrC=10*np.log10(2*2/np.mean((C-fakeC)**2))
                    psnrCC=10*np.log10(2*2/np.mean((C-fakeC_clean)**2))
                    psnrCCC=10*np.log10(2*2/np.mean((C-fakeC_clean_cycle)**2))
                    
                    valPSNRA += psnrA
                    valPSNRB += psnrB
                    valPSNRC += psnrC
                    valPSNRCC += psnrCC
                    valPSNRCCC += psnrCCC
                    numTestBatches += 1
                    
                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-A.tif'
                    A=(A+1)*127.5
                    tifffile.imsave(image_path, np.array(np.squeeze(A.astype('uint8')), dtype='uint8'))
                    
                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-B.tif'
                    B=(B+1)*127.5
                    tifffile.imsave(image_path, np.array(np.squeeze(B.astype('uint8')), dtype='uint8'))
                    
                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-C.tif'
                    C=(C+1)*127.5
                    tifffile.imsave(image_path, np.array(np.squeeze(C.astype('uint8')), dtype='uint8'))
                                        
                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-AB.tif'
                    fakeB=(fakeB+1)*127.5
                    tifffile.imsave(image_path, np.array(np.squeeze(fakeB.astype('uint8')), dtype='uint8'))
                                        
                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-ABA.tif'
                    cycleA=(cycleA+1)*127.5
                    tifffile.imsave(image_path, np.array(np.squeeze(cycleA.astype('uint8')), dtype='uint8'))
                                        
                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}BA.tif'
                    fakeA=(fakeA+1)*127.5
                    tifffile.imsave(image_path, np.array(np.squeeze(fakeA.astype('uint8')), dtype='uint8'))
                                        
                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}BAB.tif'
                    cycleB=(cycleB+1)*127.5
                    tifffile.imsave(image_path, np.array(np.squeeze(cycleB.astype('uint8')), dtype='uint8'))

                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-CSR.tif'
                    fakeC=(fakeC+1)*127.5
                    tifffile.imsave(image_path, np.array(np.squeeze(fakeC.astype('uint8')), dtype='uint8'))
                    
                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-CSRC.tif'
                    fakeC_clean=(fakeC_clean+1)*127.5
                    tifffile.imsave(image_path, np.array(np.squeeze(fakeC_clean.astype('uint8')), dtype='uint8'))
                    
                    image_path = f'./{trainOutputDir}/epoch-{epoch+1}/{numTestBatches}-CSRCC.tif'
                    fakeC_clean_cycle=(fakeC_clean_cycle+1)*127.5
                    tifffile.imsave(image_path, np.array(np.squeeze(fakeC_clean_cycle.astype('uint8')), dtype='uint8'))
                    
                    stdout.write(test_template %(numTestBatches, psnrA, psnrB, psnrC, psnrCC, psnrCCC))
                    stdout.flush()
                    if numTestBatches == args.valNum:
                        break
                valPSNRA /= numTestBatches
                valPSNRB /= numTestBatches
                valPSNRC /= numTestBatches
                valPSNRCC /= numTestBatches
                valPSNRCCC /= numTestBatches
                stdout.write("\n")
                print(f'Mean Validation PSNR-A: {valPSNRA}, PSNR-B: {valPSNRB}, PSNR-SR: {valPSNRC}, PSNR-SRC: {valPSNRCC}, PSNR-SRCC: {valPSNRCCC}')
            if (epoch+1) % args.save_freq == 0:
                #checkpoint.save(checkpoint_prefix)
                print('Saving network weights')
                generatorAB.save(f'{trainingDir}/GAB')
                generatorBA.save(f'{trainingDir}/GBA')
                discriminatorA.save(f'{trainingDir}/DA')
                discriminatorB.save(f'{trainingDir}/DB')
                generatorSR.save(f'{trainingDir}/GSR')
                if args.ganFlag:
                    discriminatorSR.save(f'{trainingDir}/DSR')

    elif args.phase == 'test':
#        # test within scope?
        testFiles = glob(args.test_dir+'/*.mat')
#        for testFile in testFiles:
#            if args.nDims == 2: # read png images
#                img = Image.open(testFile)
#                if img.mode != 'RGB': #makes it triple channel
#                    img = img.convert('RGB')
#                img = np.array(img, dtype='uint8')
#            elif args.nDims >= 3: # read .mat files
#                arrays = {}
#                f = scipy.io.loadmat(testFile)
#                for k, v in f.items():
#                    arrays[k] = np.array(v)
#                fileName=testFile.split('/')[-1].split('.')[0]
#                img=arrays[fileName]
#            img=np.array(img, dtype='float32')
#            img=np.expand_dims(np.expand_dims(img[0:300,0:300,0:300],0),4)
#            predicted = generator(img, training=False)
#            predicted = np.squeeze(np.asarray(predicted))
#            image_path = f'./{args.test_dir}/{fileName}-B.mat'
#            print(f'Generated prediction {fileName}')
#            scipy.io.savemat(image_path, {fileName: predicted})

