from utils import (
  read_data, 
  input_setup, 
  imsave,
  preprocess
)
import glob
import numpy as np
import gc
from functools import reduce
import math
import time
import os
import tensorflow as tf

class SRCNN(object):
    """6-1 init SRCNN and setup hyperparameters"""
    def __init__(self, 
               sess, 
               config):

        self.sess = sess
        self.config=config
        self.build_model()
        
    """6-2 define model"""
    def build_model(self):
        #input
        self.images = tf.placeholder(tf.float32, [None, self.config.image_size, self.config.image_size, self.config.c_dim], name='images')
        #output
        self.labels = tf.placeholder(tf.float32, [None, self.config.label_size, self.config.label_size, 1], name='labels')
        #weights
        self.weights = {
          'w1': tf.Variable(tf.truncated_normal([9, 9, self.config.c_dim, 64], stddev=1e-3, seed=111),name='w1'),
          'w2': tf.Variable(tf.truncated_normal([5, 5, 64, 32], stddev=1e-3, seed=222),name='w2'),
          'w3': tf.Variable(tf.truncated_normal([5, 5, 32, 1], stddev=1e-3, seed=333),name='w3'),
          }
        #bias
        self.biases = {
          'b1': tf.Variable(tf.constant(0.1,shape=[64]), name='b1'),
          'b2': tf.Variable(tf.constant(0.1,shape=[32]), name='b2'),
          'b3': tf.Variable(tf.constant(0.1,shape=[1]), name='b3'),
          }
        #prediction
        self.pred = self.model()
        # Loss function (MSE) #avg per sample
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        #to save best model
        self.saver = tf.train.Saver()
        
    """7-1 train/test"""
    def input_parser(self,img_path):
        img,lbl=preprocess(img_path)
        img=np.asarray([img]*self.config.c_dim).astype(np.float32)
        img=np.transpose(img,(1,2,0))#channel at tail
        return img,lbl
    def test(self):
        #load new images in a folder
        try:
            self.load(self.checkpoint_dir)
            print(" [*] Load SUCCESS")
        except:
            print(" [!] Load failed...")
            return
        new_data_dir=self.config.new_image_path
        print('new_data_dir',new_data_dir)
        new_data=tf.constant(glob.glob(os.path.join(new_data_dir, "*.bmp")))
        new_data_pathlist=self.sess.run(new_data)
        new_data_loader = tf.data.Dataset.from_tensor_slices(new_data)
        new_data_loader = new_data_loader.map(self.input_parser,num_threads=4)#path to img,lbl
        new_data_loader = new_data_loader.batch(batch_size=self.config.batch_size)
        iterator = tf.data.Iterator.from_structure(new_data_loader.output_types,new_data_loader.output_shapes)
        next_batch=iterator.get_next()
        new_init_op = iterator.make_initializer(new_data_loader)
        
        result=list()
        self.sess.run(new_init_op)
        total_mse=0.
        img_count=0.
        batch_count=0.
        start_time=time.time()
        while True:
            try:
                X,y=self.sess.run(next_batch)
                y_pred = self.pred.eval({self.images: X, self.labels: y})
                result.append(y_pred)
                total_mse+=tf.reduce_mean(tf.squared_difference(y_pred, y))
                batch_count+=1
            except tf.errors.OutOfRangeError:#all images passes
                break
        averge_mse=total_mse/batch_count
        PSNR=-10*math.log10(averge_mse)
        print("time: [%4.2f], \ntesting loss: [%.8f], \nPSNR: [%.4f]" % (time.time()-start_time, averge_mse,PSNR))
        
        #save
            #flatten
        result=reduce(lambda x,y: x+y,result)
        assert(len(result)==len(new_data_pathlist))
        for i in len(result):
            img=self.sess.run(result[i])
            imsave(img,new_data_pathlist[i].replace('.bmp','.SR.bmp'))
                    
    def train(self):
        #data preprocessing
        if(input_setup(self.sess, self.config)):#7-1-1
            print('generating patches...')
        else:
            print('found existing h5 files...')

        #build image path  
        trn_data_dir = os.path.join(self.config.checkpoint_dir,'train.c'+str(self.config.c_dim)+'.h5')
        print('trn_data_dir',trn_data_dir)
        X_train,y_train=read_data(trn_data_dir)
        trn_data_loader=tf.data.Dataset.from_tensor_slices((X_train,y_train))
        trn_data_loader = trn_data_loader.shuffle(buffer_size=X_train.shape[0])#smaller buffer size?
        trn_data_loader = trn_data_loader.batch(batch_size=self.config.batch_size)

        
        tst_data_dir = os.path.join(self.config.checkpoint_dir,'test.c'+str(self.config.c_dim)+'.h5')
        print('tst_data_dir',tst_data_dir)
        X_test,y_test=read_data(trn_data_dir)#7-1-2 read image from h5py
        tst_data_loader=tf.data.Dataset.from_tensor_slices((X_test,y_test))
        tst_data_loader = tst_data_loader.batch(batch_size=self.config.test_batch_size)
        
        #data description
        print('X_train.shape',X_train.shape)
        print('y_train.shape',y_train.shape)
        del X_train,y_train,X_test,y_test
        gc.collect()
        #iter
        iterator = tf.data.Iterator.from_structure(tst_data_loader.output_types,tst_data_loader.output_shapes)
        next_batch=iterator.get_next()
   
        trn_init_op = iterator.make_initializer(trn_data_loader)
        tst_init_op=iterator.make_initializer(tst_data_loader)
        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
  
    
        tf.global_variables_initializer().run()###remove DEPRECATED function###tf.initialize_all_variables().run()
    
        #Try to load pretrained model from checkpoint_dir
        if self.load(self.config.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        #if training
        print("Training...")
        best_PSNR=0.
        best_ep=0.
        for ep in range(self.config.epoch):#for each epoch
            epoch_loss = 0.
            average_loss = 0.
            batch_count=0.
            start_time = time.time()
            self.sess.run(trn_init_op)#need to init for each epoch
            while True:#for each batch
        # Run by batch images
                try:
                    X,y = self.sess.run(next_batch)
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: X, self.labels: y})#update weights and biases 
                    #print('err',err)
                    epoch_loss += err
                    batch_count+=1
                    #print('epoch_loss',epoch_loss)
                except tf.errors.OutOfRangeError:#all images passes
                    break
            
            average_loss = epoch_loss / batch_count #per sample
            #print(self.sess.run(average_loss))
            PSNR=-10*math.log10(average_loss)
            print("Epoch: [%2d], \n\ttime: [%4.2f], \n\ttraining loss: [%.8f], \n\tPSNR: [%.4f]" % (ep, time.time()-start_time, average_loss,PSNR))
            
            #valid
            epoch_loss = 0.
            average_loss = 0.
            batch_count=0.
            start_time = time.time()
            self.sess.run(tst_init_op)
            while True:
                try:
                    X,y=self.sess.run(next_batch)
                    err = self.sess.run(self.loss, feed_dict={self.images: X, self.labels: y})#only compute err
                    epoch_loss += err
                    batch_count+=1
                except tf.errors.OutOfRangeError:#all images passes
                    break
            average_loss = epoch_loss / batch_count #per sample
            PSNR=-10*math.log10(average_loss) 
            print("\n\ttime: [%4.2f], \n\ttesting loss: [%.4f], \n\tPSNR: [%.4f]\n\n" % (time.time()-start_time, average_loss,PSNR))
            
            #save
            if PSNR>best_PSNR:
                self.save(self.config.checkpoint_dir,ep)
                best_ep=ep
                best_PSNR=PSNR
        print('best ep',best_ep)
        print('best PSNR',best_PSNR)

    def model(self):
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='SAME') + self.biases['b3']
        out = tf.clip_by_value(conv3,0.0,1.0)
        return out

    def save(self, checkpoint_dir, step):
        model_name = "CASRCNN_C"+str(self.config.c_dim)+".model"
        model_dir = "%s_%s" % ("srcnn", self.config.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("srcnn", self.config.label_size)
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
