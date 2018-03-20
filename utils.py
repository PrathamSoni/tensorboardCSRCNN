"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
"""7-1-2 load h5"""
def read_data(path):
  """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """  
  print 'check1' 
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))

    label = np.array(hf.get('label'))

    return data, label

"""7-1-1-2"""
def preprocess(path, scale=3):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation
  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path, is_grayscale=True)
  label_ = modcrop(image, scale)#7-1-1-2-1 crop image for sclaing

  # Must be normalized
  image = image / 255.
  label_ = label_ / 255.

  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)#down-scale
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)#up-scale
 
  return input_, label_

"""7-1-1-1 generating image path for training/testing"""
def prepare_data(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  if FLAGS.is_train:
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")#test folder: Test/Set5/
    data = glob.glob(os.path.join(data_dir, "*.bmp"))

  #return a list of absolute image paths
  return data

"""7-1-1-3"""
def make_data(sess, data, label):
  """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
  if FLAGS.is_train:
    savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
  else:
    savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)
    hf.close()
  print 'check2'

def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

"""7-1-1-2-1"""
def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

"""7-1-1 input setup"""
def input_setup(sess, config):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  # Load data path
  if config.is_train:
    data = prepare_data(sess, dataset="Train")#7-1-1-1
  else:
    data = prepare_data(sess, dataset="Test")

  sub_input_sequence = []
  sub_label_sequence = []
  padding =  abs(config.image_size - config.label_size) / 2 # 6
  
  #if training
  if config.is_train:
    for i in xrange(len(data)):
      #preprocess each image
      input_, label_ = preprocess(data[i], config.scale)#7-1-1-2
      #get image size
      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        h, w = input_.shape
      #generate patches
      for x in range(0, h-config.image_size+1, config.stride):
        for y in range(0, w-config.image_size+1, config.stride):
           # We create the inputs and labels.
           # We take care to create all surrounding areas of the patch.

           # left/up
           sub_input1 = input_[x - config.image_size:x, y - config.image_size:y]  # [33 x 33]

           # right/up
           sub_input2 = input_[x + config.image_size:x + 2 * config.image_size, y - config.image_size:y]  # [33 x 33]

           # center/up
           sub_input3 = input_[x:x + config.image_size, y - config.image_size:y]  # [33 x 33]

           # left/center
           sub_input4 = input_[x - config.image_size:x, y:y + config.image_size]  # [33 x 33]

           # center/center
           sub_input5 = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
           sub_label = label_[int(x + padding):int(x + padding + config.label_size), int(y + padding):int(y + padding + config.label_size)]  # [21 x 21]

           # right/center
           sub_input6 = input_[x + config.image_size:x + 2 * config.image_size, y:y + config.image_size]  # [33 x 33]

           # center/bottom
	   sub_input7 = input_[x:x + config.image_size, y + config.image_size:y + 2 * config.image_size]  # [33 x 33]

           # left/bottom
	   sub_input8 = input_[x - config.image_size:x,y + config.image_size:y + 2 * config.image_size]  # [33 x 33]

	   # right/bottom
	   sub_input9 = input_[x + config.image_size:x + 2 * config.image_size, y + config.image_size:y + 2 * config.image_size]  # [33 x 33]

	   # Make channel value
	   # reshape image/label from 2d to 3d
 	   # Temp array to create higher channel input
	   temp_input = np.empty((config.c_dim, config.image_size, config.image_size))
 
	   listOfInputs = [sub_input1, sub_input2, sub_input3, sub_input4, sub_input5, sub_input6, sub_input7, sub_input8, sub_input9]
                   
	   # nested for loops to stack the high channel inputs
 
	   #edge cases
	   if ((x -config.image_size)<0 or (x + config.image_size)>0 or (y -config.image_size)<0 or (y + config.image_size)>0):
	     for i in range(0, config.c_dim):
	        temp_input[i] = sub_input5

	   #main block
	   else:
	     for i in range(0, config.c_dim):
	       temp_input[i] = listOfInputs[i]

	   # sets input and label
	   sub_input = temp_input.reshape([config.image_size, config.image_size, config.c_dim])
	   # label is still 1 channel
	   sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
	   # append to list
	   sub_input_sequence.append(sub_input)
	   sub_label_sequence.append(sub_label)
  else:#test
    input_, label_ = preprocess(data[0], config.scale)# !!!here, only the third image is returned for testing #test_image_path

    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      h, w = input_.shape

    # Numbers of sub-images in height and width of image are needed to compute merge operation.
    nx = ny = 0 
    for x in range(0, h-config.image_size+1, config.stride):
      nx += 1; ny = 0
      for y in range(0, w-config.image_size+1, config.stride):
        # We create the inputs and labels.
	# We take care to create all surrounding areas of the patch.

	# left/up
	sub_input1 = input_[x - config.image_size:x, y - config.image_size:y]  # [33 x 33]

	# right/up
	sub_input2 = input_[x + config.image_size:x + 2 * config.image_size,
                                 y - config.image_size:y]  # [33 x 33]

	# center/up
	sub_input3 = input_[x:x + config.image_size, y - config.image_size:y]  # [33 x 33]

	# left/center
	sub_input4 = input_[x - config.image_size:x, y:y + config.image_size]  # [33 x 33]

	# center/center
	sub_input5 = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
	sub_label = label_[int(x + padding):int(x + padding + config.label_size),
                                int(y + padding):int(y + padding + config.label_size)]  # [21 x 21]

	# right/center
	sub_input6 = input_[x + config.image_size:x + 2 * config.image_size,
                                 y:y + config.image_size]  # [33 x 33]

	# center/bottom
	sub_input7 = input_[x:x + config.image_size,
                                 y + config.image_size:y + 2 * config.image_size]  # [33 x 33]

	# left/bottom
	sub_input8 = input_[x - config.image_size:x,
                                 y + config.image_size:y + 2 * config.image_size]  # [33 x 33]

	# right/bottom
	sub_input9 = input_[x + config.image_size:x + 2 * config.image_size,
                                 y + config.image_size:y + 2 * config.image_size]  # [33 x 33]

	# Make channel value
	# reshape image/label from 2d to 3d
	# Temp array to create higher channel input
	temp_input = np.empty((config.c_dim, config.image_size, config.image_size))

	listOfInputs = [sub_input1, sub_input2, sub_input3, sub_input4, sub_input5, sub_input6, sub_input7, sub_input8, sub_input9]
	# nested for loops to stack the high channel inputs

	#edge cases
	if ((x -config.image_size)<0 or (x + 2*config.image_size)>w or (y -config.image_size)<0 or (y + 2*config.image_size)>h):
	  for i in range(0, config.c_dim):
	    temp_input[i] = sub_input5

	#main block
	else:
	  for i in range(0, config.c_dim):
	    temp_input[i] = listOfInputs[i]

	# sets input and label
	sub_input = temp_input.reshape([config.image_size, config.image_size, config.c_dim])
	
	# label is still 1 channel
	sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
	
	# append to list
	sub_input_sequence.append(sub_input)
	sub_label_sequence.append(sub_label)
       

  """
  len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
  (sub_input_sequence[0]).shape : (33, 33, 1)
  """
  # Make list to numpy array. With this transform
  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
  arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]

  make_data(sess, arrdata, arrlabel)#7-1-1-3, save training/testing data as h5

  if not config.is_train:
    return nx, ny
    
def imsave(image, path):
  return scipy.misc.imsave(path, image)
"""7-1-2 merge patches into an image"""
def merge(images, size):
  h, w = images.shape[1], images.shape[2]

  img = np.zeros((h*size[0], w*size[1], 1))
  for idx, image in enumerate(images):
    print image.shape
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img
