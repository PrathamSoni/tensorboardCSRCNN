from model import SRCNN
import tensorflow as tf
import os

"""1.configuration"""
flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Number of epoch [100]")
flags.DEFINE_integer("batch_size", 256, "The size of batch images [128]")
flags.DEFINE_integer("test_batch_size", 1024, "The size of batch images for testing") 
flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 33, "The size of label to produce [33]")

flags.DEFINE_float("learning_rate", 3e-4, "The learning rate of gradient descent algorithm [1e-4]")
#flags.DEFINE_float("momentum",0.9,"The momentum of SGD [0.9]")###add momentum for better training performance
flags.DEFINE_integer("c_dim", 9, "Dimension of image color. [9]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("trn_folderpath", "Train/general", "Name of sample directory [sample]")
flags.DEFINE_string("tst_folderpath", "Test/Set5", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
flags.DEFINE_string("new_image_path","path/to/your/image","Path of your image to test")
flags.DEFINE_string("summary_dir","save_model","Path of your save directory for tensorboard")
flags.DEFINE_boolean("make_patch",False,"generate patches if h5 already exists [False]")

FLAGS = flags.FLAGS

def main(_):
    """3.print configurations"""
    print('tf version:',tf.__version__)
    print('tf setup:')
    for k,v in FLAGS.flag_values_dict().items():
        print(k,v)
    """4.check/create folders"""
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    """5.begin tf session"""
    with tf.Session() as sess:
        """6.init srcnn model"""
        srcnn = SRCNN(sess, FLAGS)
        """7.start to train/test"""
        if(FLAGS.is_train):
            srcnn.train()
        else:
            srcnn.test()
    
if __name__ == '__main__':
    """2.call main function"""
    tf.app.run()
    

