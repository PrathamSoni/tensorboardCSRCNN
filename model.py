from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge
)
import math
import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

class SRCNN(object):
  """6-1 init SRCNN and setup hyperparameters"""
  def __init__(self, 
               sess, 
               image_size=33,
               label_size=11, 
               batch_size=128,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size			

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()
  """6-2 define model"""
  def build_model(self):
    #input
    self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
    #output
    self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, 1], name='labels')
    #weights
    self.weights = {
      'w1': tf.Variable(tf.random_normal([9, 9, 9, 64], stddev=1e-3), name='w1'),
      'w2': tf.Variable(tf.random_normal([5, 5, 64, 32], stddev=1e-3), name='w2'),
      'w3': tf.Variable(tf.random_normal([1, 1, 32, 9], stddev=1e-3), name='w3'),
      'w4': tf.Variable(tf.random_normal([5,5,9,1], stddev=1e-3), name='w4')
    }
    #bias
    self.biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([9]), name='b3'),
      'b4': tf.Variable(tf.zeros([1]), name='b4')
    }
    #prediction
    self.pred = self.model()

    # Loss function (MSE)
    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
    #to save best model
    self.saver = tf.train.Saver()
  """7-1 train/test"""
  def train(self, config):
    #data preprocessing
    if config.is_train:
      input_setup(self.sess, config)#7-1-1
    else:
      nx, ny = input_setup(self.sess, config)
    #build image path
    if config.is_train:     
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    else:
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")
    print('data_dir',data_dir)
    train_data, train_label = read_data(data_dir)#7-1-2 read image from h5py

    # Stochastic gradient descent with the standard backpropagation
    self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
#tf.train.MomentumOptimizer(config.learning_rate,config.momentum).minimize(self.loss)
###add momentum###tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)

    tf.global_variables_initializer().run()###remove DEPRECATED function###tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()
    #Try to load pretrained model from checkpoint_dir
    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
    #if training
    if config.is_train:
      print("Training...")
      epoch_loss = 0
      average_loss = 0	
      for ep in xrange(config.epoch):#for each epoch
        # Run by batch images
        batch_idxs = len(train_data) // config.batch_size#TODO: check data loader of tensorflow and shuffle training data in each epoch
        for idx in xrange(0, batch_idxs):
          batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]
          
          counter += 1
          _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})#update weights and biases 
	  epoch_loss += err
	  average_loss = epoch_loss / float(170)
          PSNR=10*math.log10(1/average_loss) 
          if counter % 10 == 0:#display training loss for every 10 batches
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err))

          if counter % 170 == 0:#save model for every 500 batches. Note: final model may not be saved!!!
            self.save(config.checkpoint_dir, counter)
          if counter % 170 == 0:
	       with open('data.txt', 'a') as file:
	           file.write(str(average_loss) + " , " + str(PSNR)+ "\n")
		   #file.write(str(average_loss) + "\n")
		   epoch_loss = 0
	           average_loss = 0
  
    else:
      print("Testing...")

      result = self.pred.eval({self.images: train_data, self.labels: train_label})#test
      
      result = merge(result, [nx, ny])#7-1-2
      print('test output size',result.shape)
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      print('image_path',image_path)
      image_path = os.path.join(image_path, "test_image.png")
      ###plot image here
      # check https://stackoverflow.com/questions/35286540/display-an-image-with-python/35286615
      imsave(result, image_path)#save image

  def model(self):

    conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'])
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID') + self.biases['b2'])
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='VALID') + self.biases['b3'])
    conv4 = tf.nn.conv2d(conv3, self.weights['w4'], strides=[1,1,1,1], padding='VALID') + self.biases['b4']

    return conv4

  def save(self, checkpoint_dir, step):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    print('checkpoint_dir',checkpoint_dir)#print folder path out
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('model_checkpoint_path',ckpt.model_checkpoint_path)#model path
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
