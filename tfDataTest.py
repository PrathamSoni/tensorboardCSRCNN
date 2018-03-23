#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 02:59:09 2018

@author: ht
"""

import tensorflow as tf
import numpy as np

X_test=np.ones((7,3))
y_test=np.zeros((7,1))
tst_data_loader=tf.data.Dataset.from_tensor_slices((X_test,y_test))
tst_data_loader = tst_data_loader.batch(batch_size=5)
iterator = tf.data.Iterator.from_structure(tst_data_loader.output_types,tst_data_loader.output_shapes)
next_batch=iterator.get_next()
tst_init_op=iterator.make_initializer(tst_data_loader)

with tf.Session() as sess:
    for i in range(2):
        sess.run(tst_init_op)
        while True:
            try:
                X,y = sess.run(next_batch)
                print(X)
                print(y)
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break