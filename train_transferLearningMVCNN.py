# train the transfer learning after getting the outout from pretrained VGG-16 on imageNet
# Most of the codes were written by us except dedeicated.

# organize imports
from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from randomMiniBatchKS import random_mini_batches
import numpy as np
import h5py
import os
import json
import datetime
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
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

cost_train_path   = config["cost_train"]
cost_dev_path   = config["cost_dev"]   
accuracy_train_path   = config["accuracy_train"]   
accuracy_dev_path   = config["accuracy_dev"]  
results_train_path   = config["results_train"]   
results_dev_path   = config["results_dev"]   
results_test_path   = config["results_test"]   

seed      = config["seed"]

# import train features and labels
h5f_data  = h5py.File(features_train_path, 'r')
h5f_label = h5py.File(labels_train_path, 'r')

features_string = h5f_data['dataset_train']
labels_string   = h5f_label['dataset_train']

trainData = np.array(features_string)
labels   = np.array(labels_string)
trainLabels = np.reshape(labels, (labels.shape[0],1))

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] train features shape: {}".format(trainData.shape))
print ("[INFO] train labels shape: {}".format(trainLabels.shape))

# import dev features and labels
h5f_data  = h5py.File(features_dev_path, 'r')
h5f_label = h5py.File(labels_dev_path, 'r')

features_string = h5f_data['dataset_dev']
labels_string   = h5f_label['dataset_dev']

testData = np.array(features_string)
labels   = np.array(labels_string)
testLabels = np.reshape(labels, (labels.shape[0],1))

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] dev features shape: {}".format(testData.shape))
print ("[INFO] dev labels shape: {}".format(testLabels.shape))

# import test features and labels
h5f_data  = h5py.File(features_test_path, 'r')
h5f_label = h5py.File(labels_test_path, 'r')

features_string = h5f_data['dataset_test']
labels_string   = h5f_label['dataset_test']

testFData = np.array(features_string)
labels   = np.array(labels_string)
testFLabels = np.reshape(labels, (labels.shape[0],1))

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] test features shape: {}".format(testFData.shape))
print ("[INFO] test labels shape: {}".format(testFLabels.shape))



print ("[INFO] training started...")
# split the training and testing data

# crate place holder
x = tf.placeholder(shape=[None, 32768], dtype=tf.float32, name = "x")
y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name = "y")

# dense NN
dense1 = tf.layers.dense(x, 1000, activation = tf.nn.relu, name = 'd1')
with tf.variable_scope('d1', reuse=True):
  w1 = tf.get_variable('kernel')
  b1 = tf.get_variable('bias')

dense2 = tf.layers.dense(dense1,500,activation = tf.nn.relu, name = 'd2')
with tf.variable_scope('d2', reuse=True):
  w2 = tf.get_variable('kernel')
  b2 = tf.get_variable('bias')

yHat = tf.layers.dense(dense1,1,activation = None, name = 'dout')
with tf.variable_scope('dout', reuse=True):
  wout = tf.get_variable('kernel')
  bout = tf.get_variable('kernel')

# define cost and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=yHat))
optimizer = tf.train.AdamOptimizer(learning_rate= 0.00001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost)

# get prediction and accuracy
y_pred = tf.to_float(tf.greater(tf.sigmoid(yHat),0.5))
accuracy = tf.reduce_mean(tf.to_float(tf.equal(y,y_pred)))

# training parameters
m = trainData.shape[0]
minibatch_size = 450
num_epochs = 50
display_step = 10
costs = []
trainCosts = []
devCosts = []
trainAccuracies = []
devAccuracies = []

#'Saver' operation to save and restore all the variables 
saver = tf.train.Saver() 

with tf.Session() as sess:
  # initialize all variables
  sess.run(tf.global_variables_initializer())

  for epoch in range(num_epochs):
    cost_in_each_epoch = 0
    num_minibatches = int(m / minibatch_size)
    seed = seed + 1
    minibatches = random_mini_batches(trainData, trainLabels, minibatch_size, seed)

    for minibatch in minibatches:
      (minibatch_X, minibatch_Y) = minibatch

      # let's start training
      _, c = sess.run([optimizer, cost], feed_dict={x: minibatch_X, y: minibatch_Y})
      cost_in_each_epoch += c
      
    if (epoch+1) % display_step == 0:
      costs.append(cost_in_each_epoch)

      # getting teh costs to display
      trainC = sess.run(cost, feed_dict={x: trainData, y: trainLabels})
      trainCosts.append(trainC)

      devC = sess.run(cost, feed_dict={x: testData, y: testLabels})
      devCosts.append(devC)
      
      trainAccuracy = accuracy.eval({x: trainData, y: trainLabels})
      trainAccuracies.append(trainAccuracy)
      devAccuracy = accuracy.eval({x: testData, y: testLabels})
      devAccuracies.append(devAccuracy)

    if (epoch+1) % (display_step) == 0:
      print("Epoch: {}".format(epoch + 1), "cost={}".format(cost_in_each_epoch), "Train cost={}".format(trainC), "Dev cost={}".format(devC), "Train Acc={}".format(trainAccuracy), "Dev Acc={}".format(devAccuracy))

  print("Optimization Finished!")

  # Test model
  print("Train Accuracy:", accuracy.eval({x: trainData, y: trainLabels}))
  print("Dev Accuracy:", accuracy.eval({x: testData, y: testLabels}))
  print("Test Accuracy:", accuracy.eval({x: testFData, y: testFLabels}))

  y_train = sess.run(y_pred, feed_dict={x: trainData})
  y_train = np.squeeze(y_train)

  y_dev = sess.run(y_pred, feed_dict={x: testData})
  y_dev = np.squeeze(y_dev)

  y_test = sess.run(y_pred, feed_dict={x: testFData})
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

  # plot the cost
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

