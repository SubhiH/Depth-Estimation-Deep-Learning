#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 07:38:54 2017

@author: Soubhi
"""

import numpy as np;
from PIL import Image
import glob
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy.misc import imread, imresize
import tensorflow as tf
import timeit

import sys;

slim = tf.contrib.slim


#Main Vars
img_width = 298;
img_height= 218;

img_output_width = 37;
img_output_height = 27;


train_input = [];
train_output = []; 
                      
val_input = [];
val_output = [];               

test_input = [];
test_output = []; 

figure_num=1;

img_size_flat = img_width * img_height
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_width, img_height)
num_channels = 3


weights_dict = np.load("/home/soubhi/Desktop/DL Project/bvlc_alexnet.npy", encoding = 'bytes').item()

              
#Helper Functions
def plot_imgs(img1,img2,figure_num):
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    plt.figure(figure_num)
    plt.gca().axis('off')
    plt.subplot(1, 2, 1)
    plt.imshow(img1.astype('uint8'))
    plt.gca().axis('off')
    plt.subplot(1 ,2, 2)
    plt.imshow(img2.astype('uint8'))


def load_data():
    train_input=np.load('train_input.npy')
    train_output=np.load('train_output.npy')

    val_input=np.load('val_input.npy')
    val_output=np.load('val_output.npy')

    test_input=np.load('test_input.npy')
    test_output=np.load('test_output.npy')
    return train_input,train_output,val_input,val_output,test_input,test_output 

def get_next_batch(data,label,train_batch_size,batch_itr):
    
    if (batch_itr+1)*train_batch_size >= len(data) and \
    batch_itr*train_batch_size<len(data):
        batch_data = data[batch_itr*train_batch_size:len(data)];
        y_batch_data = label[batch_itr*train_batch_size:len(label)];
        batch_itr = 0;
    else:
        batch_data = data[batch_itr*train_batch_size:(batch_itr+1)*train_batch_size];
        y_batch_data = label[batch_itr*train_batch_size:(batch_itr+1)*train_batch_size];
        batch_itr+=1;
   
    return batch_data,y_batch_data;


def normalized_loss(y_preds,y_trues):
    N = len(y_preds);
    for i in range(N):
        y_pred_mean = tf.reduce_mean(y_preds[i])
        y_pred_var = np.var(y_preds[i])
        
        y_true_mean = tf.reduce_mean(y_trues[i])
        y_true_var = np.var(y_trues[i])
        
    
    
    
def RootMeanSquaredErrorLog(output, gt):
#    print output.shape;
    d = LogDepth(output / 10.0) * 10.0 - LogDepth(gt / 10.0) * 10.0
    diff = tf.sqrt(tf.reduce_mean(d * d))
    return diff

def LogDepth(depth):
	depth = tf.maximum(depth, 1.0 / 255.0)	
	return 0.179581 * tf.log(depth) + 1

def loss(y_preds,y_trues):
    N = y_preds.shape[0];#len(y_preds);
    rmsel=RootMeanSquaredErrorLog(y_preds,y_trues);
    return rmsel
#    squared_diff_image = tf.square(y_trues - y_preds)
##     Sum over all dimensions except the first (the batch-dimension).
#    print squared_diff_image
#    ssd_images = tf.reduce_sum(squared_diff_image, [1, 2])
##     Take mean ssd over batch.
#    error_images = tf.reduce_mean(ssd_images)
#    return error_images;


#using Slim
def model(inputs):
    
    with slim.arg_scope([slim.conv2d],
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005)):
        #conv1
        net = slim.conv2d(inputs, 96, [11, 11],4, padding='VALID',weights_initializer=tf.truncated_normal_initializer(stddev=0.01))    
        net = tf.nn.relu(net);
        net = tf.nn.local_response_normalization(net, depth_radius=5, bias=1, alpha=0.0001, beta= 0.75)
        #pool1
        net = slim.max_pool2d(net, [3, 3], 2,padding='VALID')
        
        #conv2
        net = slim.conv2d(net, 256, [5, 5],1, padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        net = tf.nn.relu(net);
        net = tf.nn.local_response_normalization(net, depth_radius=5, bias=1, alpha=0.0001, beta= 0.75)        
        
        #conv3
        net = slim.conv2d(net, 384, [3, 3],1, padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        net = tf.nn.relu(net);
        
        #conv4
        net = slim.conv2d(net, 384, [3, 3],1, padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        net = tf.nn.relu(net);
        
        #conv5
        net = slim.conv2d(net, 256, [3, 3],1, padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        net = tf.nn.relu(net);
                        
        #pool2
        net = slim.max_pool2d(net, [3,3], 2,padding='VALID')
        #fc1 in Conv Representation  
#        net = slim.conv2d(net, 1024, [17, 12],1, padding='VALID',weights_initializer=tf.contrib.layers.xavier_initializer())
        net = tf.reshape(net,[-1,17*12*256]);        
        net = slim.fully_connected(net, 1024,weights_initializer=tf.contrib.layers.xavier_initializer())
        
        #dropout
        net = slim.dropout(net, 0.5)
       
        #fc2
#        net = slim.conv2d(net, 999, [1, 1],1, padding='VALID',weights_initializer=tf.contrib.layers.xavier_initializer())
        net = slim.fully_connected(net, 999,weights_initializer=tf.truncated_normal_initializer(mean=0.5,stddev=0.01))
        
        return net;

#Load Data
train_input,train_output,val_input,val_output,test_input,test_output=load_data();

#Plot Some Data
plot_imgs(train_input[51],train_output[51],figure_num)
figure_num+=1;
plot_imgs(train_input[5],train_output[5],figure_num)
figure_num+=1;                                                        



#Data Preprocessing    
train_output = train_output.astype('float32');
train_output = train_output/255;
train_input = train_input.astype('float32');
train_input = train_input-127;



x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
y = tf.placeholder(tf.float32, shape=[None, img_output_width, img_output_height], name='y')
preds_y = tf.placeholder(tf.float32, shape=[None,img_output_width,img_output_height], name='preds_y')
x_image = tf.reshape(x, [-1, img_width, img_height, num_channels])
net = model(x_image);
preds_y =  tf.reshape(net, [-1,img_output_width,img_output_height])
with tf.name_scope("cost_function") as scope:
    cost = loss(preds_y,y)
    tf.summary.scalar("train_function", cost)
    val_error = loss(preds_y,y)
    tf.summary.scalar("val_error", val_error)
optimizer = tf.train.MomentumOptimizer(learning_rate=0.00005,momentum=0.9).minimize(cost)


# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()




def train(epoch,train_input,train_output,batch_size,display_step = 1):
    N = len(train_input);
    batch_itr = int(N/batch_size);
    model_path = "home/soubhi/Desktop/DL Project/model01.ckpt"
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init);             
        summary_writer = tf.summary.FileWriter('/home/soubhi/Desktop/DL Project',graph=sess.graph)

#        model_variables = slim.get_model_variables()
    #    for var in variables_to_restore:
        
        #Transfer Learning From AlexNet
        for v in tf.trainable_variables():
    #        print v.name;
            if v.name=="Conv/weights:0":
                v.assign(weights_dict['conv1'][0])
                print "weights0 assigned"
            if v.name=="Conv/biases:0":
                v.assign(weights_dict['conv1'][1])
                print "Biase0 assigned"
                
        for i in range(epoch):
            print("Epoch : ",i);
            for j in range(batch_itr):
                avg_cost=0;
                print("BATCH : ",j);
                x_batch,y_true_batch = get_next_batch(train_input,train_output,batch_size,j)                                  
                feed_dict_train = {x_image: x_batch,y: y_true_batch}
                sess.run([optimizer], feed_dict=feed_dict_train)
                c=sess.run(cost, feed_dict=feed_dict_train)
                
                # Compute average loss
#                avg_cost += c / batch_size
                print "Avg Cost : ",c;
                
                summary_str = sess.run(merged_summary_op, feed_dict=feed_dict_train)
                summary_writer.add_summary(summary_str, i*batch_itr + j)
                
                
            save_path = saver.save(sess, model_path)
            print("Model Epoch %d saved in file: %s" % (epoch,save_path))

                
            if epoch % display_step == 0:
                feed_dict = {x_image: val_input,y: val_output}
                c=sess.run(cost, feed_dict=feed_dict)
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")



def evaluate(test_input,test_output):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        
    #    for v in tf.trainable_variables():
    #        print v.name
    #        if v.name=='conv/weights:0' or v.name=='fully_connected/weights:0':
    #        print sess.run(v[0])    
        feed_dict = {x_image: test_input}
        n = sess.run(net, feed_dict=feed_dict)
        img =  np.reshape(n[0], (37,27))
    #    print img
        img = img*255    
    #    img =  np.reshape(n[1], (37,27))
        plt.subplot(4, 4, 3)
        plt.imshow(img)
    #    img = img*255
        print "second"
        print(img);
    #    plt.subplot(4, 4, 4)
    #    plt.imshow(img)
        
        print("Model restored from file")
    

start = timeit.default_timer()

train(30,train_input=train_input,train_output=train_output,batch_size=32,display_step=5)
stop = timeit.default_timer()
print "Traing Time : "
print stop - start 

#colored_imgs_path = 'train_colors/';
#colored_imgs = [f for f in listdir(colored_imgs_path) if isfile(join(colored_imgs_path, f))]
#
#
#depth_imgs_path = 'train_depths/';
#depth_imgs = [f for f in listdir(depth_imgs_path) if isfile(join(depth_imgs_path, f))]

#print "train colored : " , len(colored_imgs)
#print "train depth : " , len(depth_imgs)

