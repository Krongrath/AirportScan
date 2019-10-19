#This file is used for loading the data, including .aps images and labels, 
#split data into train/dev/test sets, and augment training data.

# import libraries
from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
import os
import re

import random
from timeit import default_timer as timer
from tsahelper import *

import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import scipy.misc
import cv2 as cv2

from imgaug import augmenters as iaa
import json
import glob 
from PIL import Image
import random

#--------------------------------------------------------------------------------------------
# Get_IDs_labels(csv_file,folder_path):         read the IDs and labels from .csv file and generate output array
#
# csv_file:                 .csv file provided on Kaggle
#
# folder_path:              a path to the folder containing all .aps files  
#
# return:                   output matrix, column = 17 different zones, row = different IDs
#
#--------------------------------------------------------------------------------------------

def Get_IDs_labels(csv_file,folder_path):
    
    #Read the labels from .csv file
    unsorted_df = pd.read_csv(folder_path+'/'+csv_file)

    # Get IDs for rows
    s = list(range(0,len(unsorted_df),17))
    obs = unsorted_df.loc[s,'Id'].str.split('_')

    #list of all ID 
    scanID = [x[0] for x in obs] 

    # Put zones in columns
    columns = sorted(['Zone'+str(i) for i in range(1,18)])

    #The matrix of ID and threats in different zones 
    df = pd.DataFrame(index=scanID, columns=columns)

    # Sort labels by zone
    for i in range(17):
        s = list(range(i,len(unsorted_df),17))
        df.iloc[:,i] = unsorted_df.iloc[s,1].values
    
    return df
#----------------------------------------------------------------------------
df = Get_IDs_labels('/Users/chayawanjaikla/Desktop/Project/sample','stage1_labels.csv')

# nobj_scan = number of threats used in the scans 
nobj_scan = df.sum(1).value_counts().sort_values()

#----------------------------------------------------------------------------
#separate scanID for threat 0,1,2,3
nobj_threat = df.sum(1) #label number of threat in each example from 1 to 1147

nobj_threat_0_index = np.array(np.where(nobj_threat==0)).flatten()
ID_0_threat = list(scanID[i] for i in nobj_threat_0_index)

nobj_threat_1_index = np.array(np.where(nobj_threat==1)).flatten()
ID_1_threat = list(scanID[i] for i in nobj_threat_1_index)

nobj_threat_2_index = np.array(np.where(nobj_threat==2)).flatten()
ID_2_threat = list(scanID[i] for i in nobj_threat_2_index)

nobj_threat_3_index = np.array(np.where(nobj_threat==3)).flatten()
ID_3_threat = list(scanID[i] for i in nobj_threat_3_index)

#----------------------------------------------------------------------------
# train_dev_test(train_ratio,dev_ratio,IDlist):     split data into three different set
#
# train_ratio:          the ratio of training examples
#
# dev_ratio:            the ratio of development set examples
#
# return:               IDs of data in train,dev,and test sets
#
#----------------------------------------------------------------------------
def train_dev_test(train_ratio,dev_ratio,IDlist):
     split_1 = int(train_ratio*len(IDlist))
     split_2 = int((train_ratio+dev_ratio)*len(IDlist))
     train = IDlist[:split_1]
     dev = IDlist[split_1:split_2]
     test = IDlist[split_2:]
     return train, dev, test 
#--------------------------------------------------------------------------------------------
# split data into 3 sets based on different numbers of threats 
train_0_threat, dev_0_threat, test_0_threat = train_dev_test(train_ratio=0.7,dev_ratio=0.2,IDlist=ID_0_threat)
train_1_threat, dev_1_threat, test_1_threat = train_dev_test(train_ratio=0.7,dev_ratio=0.2,IDlist=ID_1_threat)
train_2_threat, dev_2_threat, test_2_threat = train_dev_test(train_ratio=0.7,dev_ratio=0.2,IDlist=ID_2_threat)
train_3_threat, dev_3_threat, test_3_threat = train_dev_test(train_ratio=0.7,dev_ratio=0.2,IDlist=ID_3_threat)

# combine data from different labels to train/dev/test sets 
train_no_threat = train_0_threat
train_threat = train_1_threat+train_2_threat+train_3_threat 
dev_no_threat = dev_0_threat
dev_threat = dev_1_threat+dev_2_threat+dev_3_threat
test_no_threat = test_0_threat
test_threat = test_1_threat+test_2_threat+test_3_threat

#--------------------------------------------------------------------------------------------
# This function was written by Brian Farrar (https://www.kaggle.com/jbfarrar/exploratory-data-analysis-and-example-generation)
#
# preprocess_tsa_data(subject_list,folder_path):  takes .aps files and creates .jpg images
#
# subject_list:         a list of all body scan IDs (1147 body scans) 
#
# folder_path:          a path to the folder containing all .aps files    
#
#--------------------------------------------------------------------------------------------

def preprocess_tsa_data(subject_list,folder_path):
    
    #get a list of subjects for small bore test purposes
    SUBJECT_LIST = subject_list
    FOLDER = folder_path
    
    #initialize image_data 
    image_data = ([]) #[np.zeros((512, 660, 8)) for i in range(len(scanID))]
    image_angle = np.array([1, 3, 5, 7, 9, 11, 13, 15])

    for subject in SUBJECT_LIST:

        # read in the images
        print('--------------------------------------------------------------')
        print('t+> {:5.3f} |Reading images for subject #: {}'.format(timer()-start_time, 
                                                                     subject))
        print('--------------------------------------------------------------')
        images = read_data(FOLDER + '/' + subject + '.aps')
        
        images = images[:,:,[1, 3, 5, 7, 9, 11, 13, 15]] #images size = (512,660,8)
        
        #only output the images that have all selected angles 
        if np.shape(images)[2]!=8:
        	next(subject)
        else:
        	for i in range(len(image_angle)):
                 image = images[:,:,i]
                 color_img=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
                 scipy.misc.imsave(FOLDER +'ID_{}_angle_{}.jpg'.format(subject,image_angle[i]), color_img)
#----------------------------------------------------------------------------
#running the prepropess to generate .jpg files of all images and put them 
#into different folders 

TRAIN_NO_THREAT_FOLDER = '/Users/chayawanjaikla/Google Drive File Stream/Team Drives/DL2018/images/train_no_threat/'
TRAIN_THREAT_FOLDER = '/Users/chayawanjaikla/Google Drive File Stream/Team Drives/DL2018/images/train_threat/'
DEV_NO_THREAT_FOLDER = '/Users/chayawanjaikla/Google Drive File Stream/Team Drives/DL2018/images/dev_no_threat/'
DEV_THREAT_FOLDER = '/Users/chayawanjaikla/Google Drive File Stream/Team Drives/DL2018/images/dev_threat/'
TEST_NO_THREAT_FOLDER = '/Users/chayawanjaikla/Google Drive File Stream/Team Drives/DL2018/images/test_no_threat/'
TEST_THREAT_FOLDER = '/Users/chayawanjaikla/Google Drive File Stream/Team Drives/DL2018/images/test_threat/'

preprocess_tsa_data(train_no_threat,TRAIN_NO_THREAT_FOLDER)
preprocess_tsa_data(train_threat,TRAIN_THREAT_FOLDER)

preprocess_tsa_data(dev_no_threat,DEV_NO_THREAT_FOLDER)
preprocess_tsa_data(dev_threat,DEV_THREAT_FOLDER)

preprocess_tsa_data(test_no_threat,TEST_NO_THREAT_FOLDER)
preprocess_tsa_data(test_threat,TEST_THREAT_FOLDER)

#----------------------------------------------------------------------------
# GetID(D_file_path,filename):          read the text file containing IDs and output the list
#
# ID_file_path:             a path to a folder containing .txt file
#
# filename:                 a .txt file containing lists of IDs
#
# return:                   a list of all IDs in the .txt file
#
#----------------------------------------------------------------------------
def GetID(ID_file_path,filename):
    IDlist=[]
    with open(ID_file_path+filename, "r") as f:
      for line in f:
        IDlist.append(line.strip())
    # with open(ID_file_path+filename, 'r') as f:
    #      IDlist = json.loads(f.read())

    return IDlist
#----------------------------------------------------------------------------
FOLDER = r"I:\Krongrath\Transfer\dataset\\"

# train_no_threat_list = GetID(FOLDER,'train_no_threat.txt')
# train_threat_list = GetID(FOLDER,'train_threat.txt')
# print(ID_list)

#----------------------------------------------------------------------------
# GetName(Folder,ID_file_path,filename:          get the full path to access .jpg file of each images 
#
# ID_file_path:             a path to a folder containing .jpg files 
#
# filename:                 a .txt file containing lists of IDs
#
# return:                   a list of all paths of .jpg files 
#
#----------------------------------------------------------------------------
def GetName(Folder,ID_file_path,filename):
    list_of_ID = GetID(ID_file_path,filename)
    image_angle = np.array([1,3,5,7,9,11,13,15])
    image_list = []
    for ID in list_of_ID:
         for i in np.nditer(image_angle):
             image = Folder+'ID_{}_angle_{}.jpg'.format(ID,i) 
             image_list.append(image)
    return image_list

#----------------------------------------------------------------------------
# GetName2(Folder,ID_file_path,list_of_ID):         get the full path to access .jpg file of each images 
#
# Folder                    a folder contain .jpg files 
#
# ID_file_path:             a path to a folder containing .jpg files 
#
# list_of_ID:               a list of IDs
#
# return:                   a list of all IDs in the .txt file
#
#----------------------------------------------------------------------------
def GetName2(Folder,ID_file_path,list_of_ID):
    image_angle = np.array([1,3,5,7,9,11,13,15])
    image_list = []
    for ID in list_of_ID:
         for i in np.nditer(image_angle):
             image = Folder+'ID_{}_angle_{}.jpg'.format(ID,i) 
             image_list.append(image)
    return image_list
#----------------------------------------------------------------------------
# get the full paths to access .jpg files for "no threat" in training set
TRAIN_NO_THREAT_FOLDER = r"I:\Krongrath\Transfer\dataset\imageFile\train_no_threat\\"
ID_list = GetID(FOLDER,'train_no_threat.txt')
image_train_no_threat = GetName(TRAIN_NO_THREAT_FOLDER, FOLDER, 'train_no_threat.txt')

#Save the array of all images into .npy file
images_train_no_threat = np.array([np.array(Image.open(fname)) for fname in image_train_no_threat])
np.save('images_train_no_threat_array',images_train_no_threat)

#load 4D array of images (shape = (1576,660,512,3))
images = np.load('./images_train_no_threat_array.npy')

#----------------------------------------------------------------------------
# get the full paths to access .jpg files for "threat" in training set
# NOTE: only random 270 samples will be augmented
TRAIN_THREAT_FOLDER = r"I:\Krongrath\Transfer\dataset\imageFile\train_threat\\"
ID_list = GetID(FOLDER,'train_threat.txt')
image_train_threat = GetName2(TRAIN_THREAT_FOLDER, FOLDER, ID_list)

num_to_select = 270                         # set the number to select here.
ID_list_270 = random.sample(ID_list, num_to_select)
image_train_threat_270 = GetName2(TRAIN_THREAT_FOLDER, FOLDER, ID_list_270)

#write all ID of "threat" that will be augmneted in training data
with open(r'.\dataset\train_threat_augment.txt', 'w') as outfile:
   json.dump(ID_list_270, outfile)

#Save the array of all images into .npy file
images_train_threat = np.array([np.array(Image.open(fname)) for fname in image_train_threat])
np.save('images_train_threat_array',images_train_threat)

images_train_threat_270 = np.array([np.array(Image.open(fname)) for fname in image_train_threat_270])
np.save('images_train_threat_270_array',images_train_threat_270)

#load 4D array of images 
images = np.load('./images_train_threat_array.npy')
images_270 = np.load('./images_train_threat_270_array.npy')

#----------------------------------------------------------------------------
# All of the augmentation used in this project
# We use "imgaug" library to help augmenting images (https://github.com/aleju/imgaug)

# Shifting images in y upward direction at randomly between 20 to 100 pixels
seq_translate_y_up = iaa.Sequential([ #USE THIS
     iaa.Affine(
      translate_px={"y": (-100,-20)})
  ])

# Shifting images in y downward direction at randomly between 20 to 100 pixels
seq_translate_y_down = iaa.Sequential([ #USE THIS
     iaa.Affine(
        translate_px={"y": (20,100)})
    ])

# Shifting images in y upward/downward direction at randomly at the range of 100 pixels
seq_translate_up_down = iaa.Sequential([
     iaa.Affine(
      translate_px={"y": (-100,100)})
  ])

# Sharpening images 
seq_sharpen = iaa.Sequential([ #USE THIS
   iaa.Sharpen(
        alpha=(0.3,0.5))
   ])

# Changing the lightness of images 
seq_lightness = iaa.Sequential([ #USE THIS
     iaa.Sharpen(
            alpha=(0.3,0.5), lightness=(0.3,0.8))
     ])

# Combiation of shifting images in y directions, sharpening, and changing the lightness of images 
seq_translate_sharpen = iaa.Sequential([ #USE THIS 
     iaa.Affine(
      translate_px={"y": (-100,100)}),

     iaa.Sharpen(
        alpha=(0.3,0.5), lightness=(0.3,0.8))
  ])

#For train "no threat" (augment = 765 files)
images_translate_y_up = seq_translate_y_up.augment_images(images)
images_translate_y_down = seq_translate_y_down.augment_images(images)
images_sharp = seq_sharpen.augment_images(images)
images_lightness = seq_lightness.augment_images(images)
images_translate_sharp = seq_translate_sharpen.augment_images(images)
print('Finish "no threat" augmentation')

#For train "threat"(augment=290 files)
images_translate_y_up_down = seq_translate_up_down.augment_images(images_270)
print('image_translate_y_up_down.shape=',images_translate_y_up_down.shape)
print('Finish "threat" augmentation')

#----------------------------------------------------------------------------
#
# save_image_aug(FOLDER,type_of_image,ID_list,images): save numpy array to .jpg
#
# FOLDER:          a path of output file
#
# type_of_image    a string type of augmentation
#
# ID_list          a list of all IDs
#
# images           an array of images 
#
#----------------------------------------------------------------------------
def save_image_aug(FOLDER,type_of_image,ID_list,images):
    count = 0
    image_angle = np.array([1,3,5,7,9,11,13,15])
    for ID in ID_list:
        for j in image_angle:
             scipy.misc.imsave(FOLDER + '{}_ID_{}_angle_{}.jpg'.format(type_of_image,ID,j), images[count])
             count += 1
#----------------------------------------------------------------------------
#For train "no threat" (augment = 765 files)
AUGMENT_FOLDER = r'I:\Krongrath\Transfer\dataset\imageFile\train_no_threat_augment\\'

#save all images from different types of augmentations together
save_image_aug(AUGMENT_FOLDER,'original', ID_list,images)
save_image_aug(AUGMENT_FOLDER,'translate_y_up',ID_list,images_translate_y_up)
save_image_aug(AUGMENT_FOLDER,'translate_y_down',ID_list,images_translate_y_down)
save_image_aug(AUGMENT_FOLDER,'sharp',ID_list,images_sharp)
save_image_aug(AUGMENT_FOLDER,'lightness',ID_list,images_lightness)
save_image_aug(AUGMENT_FOLDER,'translate_sharp',ID_list,images_translate_sharp)

#For train "threat" (augment = 765 files)
AUGMENT_FOLDER = r'I:\Krongrath\Transfer\dataset\imageFile\train\train_threat_augment\\'
#save all images from different types of augmentations together
save_image_aug(FOLDER=AUGMENT_FOLDER,type_of_image='original',ID_list=ID_list,images=images)
save_image_aug(AUGMENT_FOLDER,'translate_y_up_down',ID_list_270,images_translate_y_up_down)

#----------------------------------------------------------------------------
# create_ID_list(type_of_image,ID_list):    create the ID list to use in extract_feature.py
# 
# type_of_image     a string type of augmentation
#
# ID_list           a list of all IDs
#
#----------------------------------------------------------------------------
def create_ID_list(type_of_image,ID_list):
    all_ID = []
    for type in type_of_image:
        for ID in ID_list:
             single_ID = '{}_ID_{}'.format(type,ID)
             all_ID.append(single_ID)

    return all_ID

