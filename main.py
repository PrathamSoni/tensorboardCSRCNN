from model import SRCNN
from utils import input_setup

import numpy as np
import tensorflow as tf

import pprint
import os

"""1.configuration"""
flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Number of epoch [100]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 33, "The size of label to produce [33]")
flags.DEFINE_integer("test_batch_size", 65536, "The size of batch images for testing [65536]") 
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_float("momentum",0.9,"The momentum of SGD [0.9]")###add momentum for better training performance
flags.DEFINE_integer("c_dim", 9, "Dimension of image color. [9]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
flags.DEFINE_string("test_image_path","path/to/your/image","Path of your image to test")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  """3.print configurations"""
  pp.pprint(flags.FLAGS.__flags)
  """4.check/create folders"""
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  """5.begin tf session"""
  with tf.Session() as sess:
    """6.init srcnn model"""
    srcnn = SRCNN(sess, 
                  image_size=FLAGS.image_size, 
                  label_size=FLAGS.label_size, 
                  batch_size=FLAGS.batch_size,
		  learning_rate=FLAGS.learning_rate,
                  c_dim=FLAGS.c_dim, 
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir)
    """7.start to train/test"""
    srcnn.train(FLAGS)
    
if __name__ == '__main__':
  """2.call main function"""
  tf.app.run()

