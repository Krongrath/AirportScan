#This files is used for extracting features from the result of 
#forward propagation of pre-trained VGG-16. The output from this file will be used to train in train.py file.
#This code is modified from the blog post of "using pre-trained deep learning models for your own dataset"
#by Gogul09 (https://gogul09.github.io/software/flower-recognition-deep-learning)

# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from Preprocess import *

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

# load the user configs
with open('conf/conf.json') as f:    
  config = json.load(f)

# config variables
dataset_path = config["dataset_path"]
model_name    = config["model"]
weights     = config["weights"]
include_top   = config["include_top"]
train_path    = config["train_path"]
dev_path = config["dev_path"]
test_path = config["test_path"]
features_train_path   = config["features_train_path"]
labels_train_path   = config["labels_train_path"]
features_dev_path   = config["features_dev_path"]
labels_dev_path   = config["labels_dev_path"]
features_test_path   = config["features_test_path"]
labels_test_path   = config["labels_test_path"]
test_size     = config["test_size"]
results     = config["results"]
model_path    = config["model_path"]

# start time
print ("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()

# create the pretrained models
# check for pretrained weight usage or not
# check for top layers to be included or not
if model_name == "vgg16":
  base_model = VGG16(weights=weights)
  # model = Model(input=base_model.input, output=base_model.get_layer('fc1').output) #use this line to get fc1 layer (transfer learning)
  model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output) #use this line to get block4_pool layer (fine-tuning)
  image_size = (224, 224)
elif model_name == "vgg19":
  base_model = VGG19(weights=weights)
  model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
  image_size = (224, 224)
elif model_name == "resnet50":
  base_model = ResNet50(weights=weights)
  model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)
  image_size = (224, 224)
elif model_name == "inceptionv3":
  base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
  model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
  image_size = (299, 299)
elif model_name == "inceptionresnetv2":
  base_model = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
  model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
  image_size = (299, 299)
elif model_name == "mobilenet":
  base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
  model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
  image_size = (224, 224)
elif model_name == "xception":
  base_model = Xception(weights=weights)
  model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
  image_size = (299, 299)
else:
  base_model = None

print ("[INFO] successfully loaded base model and model...")

image_angle = np.array([1,3,5,7,9,11,13,15])

# path to training/dev/test datasets
train_labels = os.listdir(train_path)
dev_labels = os.listdir(dev_path)
test_labels = os.listdir(test_path)

#------------------------------TRAINING-----------------------------------------
# encode the labels
print ("[INFO] encoding labels...")
le = LabelEncoder()
le.fit([tl for tl in train_labels])

# variables to hold features and labels
features_train = []
labels_train   = []

# type of all images in training 
type_of_image_train_no_threat = ['original','translate_y_up','translate_y_down','sharp','lightness','translate_sharp']
type_of_image_train_thrat = ['original','translate_y_up_down']

# loop over all the labels in the train folder
count = 1
for i, label in enumerate(train_labels):
  print("i is", i)
  cur_path = train_path + "/" + label
  count = 1
  if i==0:
     IDlist = create_ID_list(GetID(dataset_path,'train_no_threat.txt'),type_of_image_train_no_threat)

  if i==1:
     IDlist = create_ID_list2(GetID(dataset_path,'train_threat.txt'),'original',GetID(dataset_path,'train_threat_augment.txt'),'translate_y_up_down')

  for ID in IDlist: 
    flatTotal_train = np.array([])
    for j in image_angle: 
  #for image_path in glob.glob(cur_path + "/*.jpg"):ss
      image_path = ID+'_angle_{}.jpg'.format(j)
      img = image.load_img(cur_path+"/"+image_path, target_size=image_size)
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      feature_train = model.predict(x)
      flat_train = feature_train.flatten()      
      flatTotal_train = np.concatenate([flatTotal_train,flat_train])
      # print("flat total lenght is ", flatTotal.shape)
    features_train.append(flatTotal_train)
    labels_train.append(label)
    # print("features lenght is ", len(features))
    print ("[INFO] processed - " + str(count))
    count += 1
  print ("[INFO] completed label - " + label)

# encode the labels using LabelEncoder
le = LabelEncoder()
le_labels = le.fit_transform(labels_train)

# get the shape of training labels
print ("[STATUS] training labels: {}".format(le_labels))
print ("[STATUS] training labels shape: {}".format(le_labels.shape))

# save features and labels
h5f_data = h5py.File(features_train_path, 'w')
h5f_data.create_dataset('dataset_train', data=np.array(features_train))

h5f_label = h5py.File(labels_train_path, 'w')
h5f_label.create_dataset('dataset_train', data=np.array(le_labels))

h5f_data.close()
h5f_label.close()

#------------------------------DEV-----------------------------------------
# encode the labels
print ("[INFO] encoding labels...")
le = LabelEncoder()
le.fit([tl for tl in dev_labels])

# variables to hold features and labels
features_dev = []
labels_dev   = []

# loop over all the labels in the train folder
count = 1
for i, label in enumerate(dev_labels):
  print("i is", i)
  cur_path = dev_path + "/" + label
  count = 1
  if i==0:
     IDlist = create_ID_list_no_aug(GetID(dataset_path,'dev_no_threat.txt'))
    
  if i==1:
     IDlist = create_ID_list_no_aug(GetID(dataset_path,'dev_threat.txt'))

  for ID in IDlist: 
    flatTotal_dev = np.array([])
    for j in image_angle: 
  #for image_path in glob.glob(cur_path + "/*.jpg"):ss
      image_path = ID+'_angle_{}.jpg'.format(j)
      img = image.load_img(cur_path+"/"+image_path, target_size=image_size)
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      feature_dev = model.predict(x)
      flat_dev = feature_dev.flatten()      
      flatTotal_dev = np.concatenate([flatTotal_dev,flat_dev])
      # print("flat total lenght is ", flatTotal.shape)
    features_dev.append(flatTotal_dev)
    labels_dev.append(label)
    # print("features lenght is ", len(features))
    print ("[INFO] processed - " + str(count))
    count += 1
  print ("[INFO] completed label - " + label)

# encode the labels using LabelEncoder
le = LabelEncoder()
le_labels = le.fit_transform(labels_dev)

# get the shape of training labels
print ("[STATUS] training labels: {}".format(le_labels))
print ("[STATUS] training labels shape: {}".format(le_labels.shape))

# save features and labels
h5f_data = h5py.File(features_dev_path, 'w')
h5f_data.create_dataset('dataset_dev', data=np.array(features_dev))

h5f_label = h5py.File(labels_dev_path, 'w')
h5f_label.create_dataset('dataset_dev', data=np.array(le_labels))

h5f_data.close()
h5f_label.close()

# #------------------------------TEST-----------------------------------------
# encode the labels
print ("[INFO] encoding labels...")
le = LabelEncoder()
le.fit([tl for tl in test_labels])

# variables to hold features and labels
features_test = []
labels_test   = []

# loop over all the labels in the train folder
count = 1
for i, label in enumerate(test_labels):
  print("i is", i)
  cur_path = test_path + "/" + label
  count = 1
  if i==0:
     IDlist = create_ID_list_no_aug(GetID(dataset_path,'test_no_threat.txt'))
  if i==1:
     IDlist = create_ID_list_no_aug(GetID(dataset_path,'test_threat.txt'))

  for ID in IDlist: 
    flatTotal_test = np.array([])
    for j in image_angle: 
  #for image_path in glob.glob(cur_path + "/*.jpg"):ss
      image_path = ID+'_angle_{}.jpg'.format(j)
      img = image.load_img(cur_path+"/"+image_path, target_size=image_size)
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      feature_test= model.predict(x)
      flat_test = feature_test.flatten()      
      flatTotal_test = np.concatenate([flatTotal_test,flat_test])
      # print("flat total lenght is ", flatTotal.shape)
    features_test.append(flatTotal_test)
    labels_test.append(label)
    # print("features lenght is ", len(features))
    print ("[INFO] processed - " + str(count))
    count += 1
  print ("[INFO] completed label - " + label)

# encode the labels using LabelEncoder
le = LabelEncoder()
le_labels = le.fit_transform(labels_test)

# get the shape of training labels
print ("[STATUS] training labels: {}".format(le_labels))
print ("[STATUS] training labels shape: {}".format(le_labels.shape))

# save features and labels
h5f_data = h5py.File(features_test_path, 'w')
h5f_data.create_dataset('dataset_test', data=np.array(features_test))

h5f_label = h5py.File(labels_test_path, 'w')
h5f_label.create_dataset('dataset_test', data=np.array(le_labels))

h5f_data.close()
h5f_label.close()

print ("[STATUS] features and labels saved..")

#-----------------------------------------------------------------------
# end time
end = time.time()
print ("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))