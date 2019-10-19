# Finetune the multi-view CNN (MVCNN)  
# Most of the codes were written by us except dedeicated.

# filter warnings
from __future__ import print_function

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from randomMiniBatchKS import random_mini_batches4CNNPool

# keras imports
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.models import model_from_json
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time
import tensorflow as tf
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
import uuid

# load the user configs
with open('conf.json') as f:    
  config = json.load(f)

# config variables
model_name    = config["model"]
weights     = config["weights"]
include_top   = config["include_top"]
features_train_path   = config["features_train_path"]
labels_train_path   = config["labels_train_path"]
features_dev_path   = config["features_dev_path"]
labels_dev_path   = config["labels_dev_path"]
features_test_path   = config["features_test_path"]
labels_test_path   = config["labels_test_path"]

cost_train_path   = config["cost_train_DO"]
cost_dev_path   = config["cost_dev_DO"]   
accuracy_train_path   = config["accuracy_train_DO"]   
accuracy_dev_path   = config["accuracy_dev_DO"]   
results_train_path   = config["results_train_DO"]   
results_dev_path   = config["results_dev_DO"]   
results_test_path   = config["results_test_DO"]  

seed      = config["seed"]

# start time
print ("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()

# import train features and labels
h5f_data  = h5py.File(features_train_path, 'r')
h5f_label = h5py.File(labels_train_path, 'r')

features_string = h5f_data['dataset_train']
labels_string   = h5f_label['dataset_train']

features_train = np.array(features_string)
trainDataRaw = np.reshape(features_train,(features_train.shape[0],8,14,14,512))
labels_train   = np.array(labels_string)
trainLabels = np.reshape(labels_train, (labels_train.shape[0],1))

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] trainRaw features shape: {}".format(trainDataRaw.shape))
print ("[INFO] train labels shape: {}".format(trainLabels.shape))


# import dev features and labels
h5f_data  = h5py.File(features_dev_path, 'r')
h5f_label = h5py.File(labels_dev_path, 'r')

features_string = h5f_data['dataset_dev']
labels_string   = h5f_label['dataset_dev']

features_dev = np.array(features_string)
testDataRaw = np.reshape(features_dev,(features_dev.shape[0],8,14,14,512))
labels_dev   = np.array(labels_string)
testLabels = np.reshape(labels_dev, (labels_dev.shape[0],1))

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] devRaw features shape: {}".format(testDataRaw.shape))
print ("[INFO] dev labels shape: {}".format(testLabels.shape))

# import test features and labels
h5f_data  = h5py.File(features_test_path, 'r')
h5f_label = h5py.File(labels_test_path, 'r')

features_string = h5f_data['dataset_test']
labels_string   = h5f_label['dataset_test']

features_test = np.array(features_string)
testFDataRaw = np.reshape(features_test,(features_test.shape[0],8,14,14,512))
labels_test   = np.array(labels_string)
testFLabels = np.reshape(labels_test, (labels_test.shape[0],1))

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] testRaw features shape: {}".format(testFDataRaw.shape))
print ("[INFO] test labels shape: {}".format(testLabels.shape))

# create the pretrained models
# check for pretrained weight usage or not
# check for top layers to be included or not

trainData = np.zeros((trainDataRaw.shape[0],14,14,512))

# multi-view pool (elementwise max operation)
for i in range(trainDataRaw.shape[0]):
  trainData[i,:,:,:] = trainDataRaw[i,1,:,:,:] 
  for j in range(trainDataRaw.shape[1]-1):
    trainData[i,:,:,:] = np.maximum(trainData[i,:,:,:],trainDataRaw[i,j+1,:,:,:])

testData = np.zeros((testDataRaw.shape[0],14,14,512))

for i in range(testDataRaw.shape[0]):
  testData[i,:,:,:] = testDataRaw[i,1,:,:,:] 
  for j in range(testDataRaw.shape[1]-1):
    testData[i,:,:,:] = np.maximum(testData[i,:,:,:],testDataRaw[i,j+1,:,:,:])

testFData = np.zeros((testFDataRaw.shape[0],14,14,512))

for i in range(testFDataRaw.shape[0]):
  testFData[i,:,:,:] = testFDataRaw[i,1,:,:,:] 
  for j in range(testDataRaw.shape[1]-1):
    testFData[i,:,:,:] = np.maximum(testFData[i,:,:,:],testFDataRaw[i,j+1,:,:,:])

# verify the shape of features and labels
print ("[INFO] train features shape: {}".format(trainData.shape))
print ("[INFO] test features shape: {}".format(testData.shape))

# extracting model weight from Keras documentations
base_model = VGG16(weights="imagenet")
model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
image_size = (224, 224)

names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()

paramInitWeight = {}

# assign the weights form pretrained VGG-16 on imagenet 
for name, weight in zip(names, weights):
  if name == 'block5_conv1/kernel:0':
      paramInitWeight['B5_C1_K'] = weight
  elif name == 'block5_conv1/bias:0':
      paramInitWeight['B5_C1_B'] = weight
  elif name == 'block5_conv2/kernel:0':
      paramInitWeight['B5_C2_K'] = weight
  elif name == 'block5_conv2/bias:0':
      paramInitWeight['B5_C2_B'] = weight
  elif name == 'block5_conv3/kernel:0':
      paramInitWeight['B5_C3_K'] = weight                
  elif name == 'block5_conv3/bias:0':
      paramInitWeight['B5_C3_B'] = weight
  elif name == 'fc1/kernel:0':
      paramInitWeight['fc1_K'] = weight   
  elif name == 'fc1/bias:0':
      paramInitWeight['fc1_B'] = weight        

# create the placeholders
with tf.name_scope('inputs'):
  x = tf.placeholder(tf.float32,shape=[None,14,14,512])
  y = tf.placeholder(tf.float32,shape=[None,1])
  keep_prob = tf.placeholder(tf.float32)

# build fine-tined model
conv_kernel_1_sub1 = tf.nn.conv2d(x, paramInitWeight['B5_C1_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_1_sub1 = tf.nn.bias_add(conv_kernel_1_sub1, paramInitWeight['B5_C1_B'])
layer_1_sub1 = tf.nn.relu(bias_layer_1_sub1)
conv_kernel_2_sub1 = tf.nn.conv2d(layer_1_sub1, paramInitWeight['B5_C2_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_2_sub1 = tf.nn.bias_add(conv_kernel_2_sub1, paramInitWeight['B5_C2_B'])
layer_2_sub1 = tf.nn.relu(bias_layer_2_sub1)
conv_kernel_3_sub1 = tf.nn.conv2d(layer_2_sub1, paramInitWeight['B5_C3_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_3_sub1 = tf.nn.bias_add(conv_kernel_3_sub1, paramInitWeight['B5_C3_B'])
layer_3_sub1 = tf.nn.relu(bias_layer_3_sub1)
last_pool_sub1 = tf.layers.max_pooling2d(inputs=layer_3_sub1, pool_size=[2, 2], strides=2)
last_flattening_sub1 = tf.reshape(last_pool_sub1, [-1, 7*7*512])
extractF_sub1 = tf.nn.relu_layer(last_flattening_sub1, paramInitWeight['fc1_K'], paramInitWeight['fc1_B'])

dense0 = tf.layers.dense(extractF_sub1, 4096, activation = tf.nn.relu)
dense0DO = tf.nn.dropout(dense0, keep_prob)
dense1 = tf.layers.dense(dense0DO, 1000, activation = tf.nn.relu)
dense2 = tf.layers.dense(dense1,500,activation = tf.nn.relu)
yHat = tf.layers.dense(dense1,1,activation = None)

# define cost and optimizers
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=yHat))
optimizer = tf.train.AdamOptimizer(learning_rate= 0.00001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost)

# get predictions and accuracy
y_pred = tf.to_float(tf.greater(tf.sigmoid(yHat),0.5))
accuracy = tf.reduce_mean(tf.to_float(tf.equal(y,y_pred)))

m = trainData.shape[0]
minibatch_size = 480
num_epochs = 50
display_step = 10
costs = []
trainCosts = []
devCosts = []
trainAccuracies = []
devAccuracies = []

with tf.Session() as sess:
  # initialize all variables
  sess.run(tf.global_variables_initializer())

  for epoch in range(num_epochs):
    cost_in_each_epoch = 0
    num_minibatches = int(m / minibatch_size)
    seed = seed + 1
    minibatches = random_mini_batches4CNNPool(trainData, trainLabels, minibatch_size, seed)

    for minibatch in minibatches:
      (minibatch_X, minibatch_Y) = minibatch

      # let's start training
      _, c = sess.run([optimizer, cost], feed_dict={x: minibatch_X, y: minibatch_Y, keep_prob: 0.8})
      cost_in_each_epoch += c
      
    # you can uncomment next two lines of code for printing cost when training
    if (epoch+1) % display_step == 0:
      costs.append(cost_in_each_epoch)

      # yHatDev = sess.run(yHat, feed_dict={x: testData})
      trainC = sess.run(cost, feed_dict={x: trainData, y: trainLabels, keep_prob: 1.0})
      trainCosts.append(trainC)

      devC = sess.run(cost, feed_dict={x: testData, y: testLabels, keep_prob: 1.0})
      devCosts.append(devC)
      
      trainAccuracy = accuracy.eval({x: trainData, y: trainLabels, keep_prob: 1.0})
      trainAccuracies.append(trainAccuracy)
      devAccuracy = accuracy.eval({x: testData, y: testLabels, keep_prob: 1.0})
      devAccuracies.append(devAccuracy)

    if (epoch+1) % (display_step*10) == 0:
      print("Epoch: {}".format(epoch + 1), "cost={}".format(cost_in_each_epoch), "Train cost={}".format(trainC), "Dev cost={}".format(devC), "Train Acc={}".format(trainAccuracy), "Dev Acc={}".format(devAccuracy))

  print("Optimization Finished!")

  # Test model
  print("Train Accuracy:", accuracy.eval({x: trainData, y: trainLabels, keep_prob: 1.0}))
  print("Dev Accuracy:", accuracy.eval({x: testData, y: testLabels, keep_prob: 1.0}))
  print("Test Accuracy:", accuracy.eval({x: testFData, y: testFLabels, keep_prob: 1.0}))

  y_train = sess.run(y_pred, feed_dict={x: trainData, keep_prob: 1.0})
  y_train = np.squeeze(y_train)

  y_dev = sess.run(y_pred, feed_dict={x: testData, keep_prob: 1.0})
  y_dev = np.squeeze(y_dev)

  y_test = sess.run(y_pred, feed_dict={x: testFData, keep_prob: 1.0})
  y_test = np.squeeze(y_test)

  print("Precision", precision_score(testLabels, y_dev))
  print("Recall", recall_score(testLabels, y_dev))
  print("f1_score", f1_score(testLabels, y_dev))

  print("Precision Test", precision_score(testFLabels, y_test))
  print("Recall Test", recall_score(testFLabels, y_test))
  print("f1_score Test", f1_score(testFLabels, y_test))

  print ("[STATUS] finish time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))

  # write output files
  h5f_y_train = h5py.File(results_train_path, 'w')
  h5f_y_train.create_dataset('y_train', data=y_train)

  h5f_y_dev = h5py.File(results_dev_path, 'w')
  h5f_y_dev.create_dataset('y_dev', data=y_dev)

  h5f_y_test = h5py.File(results_test_path, 'w')
  h5f_y_test.create_dataset('y_test', data=y_test)

  h5f_cost_train = h5py.File(cost_train_path, 'w')
  h5f_cost_train.create_dataset('cost_train', data=np.squeeze(trainCosts))

  h5f_cost_dev = h5py.File(cost_dev_path, 'w')
  h5f_cost_dev.create_dataset('cost_dev', data=np.squeeze(devCosts))

  h5f_accuracy_train = h5py.File(accuracy_train_path, 'w')
  h5f_accuracy_train.create_dataset('accuracy_train', data=np.squeeze(trainAccuracies))

  h5f_accuracy_dev = h5py.File(accuracy_dev_path, 'w')
  h5f_accuracy_dev.create_dataset('accuracy_dev', data=np.squeeze(devAccuracies))

  h5f_y_train.close()
  h5f_y_dev.close()
  h5f_y_test.close()
  h5f_cost_train.close()
  h5f_cost_dev.close()
  h5f_accuracy_train.close()
  h5f_accuracy_dev.close()
  print ("[STATUS] saved training and accuracy")


  #plot the cost
  plt.figure(0)
  plt.plot(np.squeeze(trainCosts),'b')
  plt.plot(np.squeeze(devCosts),'r')
  plt.ylabel('cost')
  plt.xlabel('iterations (per tens)')
  plt.show()

  plt.figure(1)
  plt.plot(np.squeeze(trainAccuracies),'b')
  plt.plot(np.squeeze(devAccuracies),'r')
  plt.ylabel('accuracy')
  plt.xlabel('iterations (per tens)')
  plt.show()


