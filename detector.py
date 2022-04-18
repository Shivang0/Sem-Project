import cv2
import os
import numpy as np
import keras
import matplotlib.pyplot as plt
import wget
from random import shuffle
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense, Activation
import sys
import h5py


class Detective:
    def __init__(self) -> None:

        self.runfirst()

    def runfirst(self):
        self.initstuff()
        self.detect()

    def initstuff(self):
        self.model = keras.models.load_model('wholemodel')
        
        # Frame size  
        self.img_size = 224

        self.img_size_touple = (self.img_size, self.img_size)

        # Number of channels (RGB)
        self.num_channels = 3

        # Flat frame size
        self.img_size_flat = self.img_size * self.img_size * self.num_channels

        # Number of classes for classification (Violence-No Violence)
        self.num_classes = 2

        # Number of files to train
        self._num_files_train = 1

        # Number of frames per video
        self._images_per_file = 20

        # Number of frames per training set
        self._num_images_train = self._num_files_train * self._images_per_file

        # Video extension
        self.video_exts = ".avi"


    def get_frames(self, current_dir, file_name):
                
        in_file = os.path.join(current_dir, file_name)
        
        images = []
        
        vidcap = cv2.VideoCapture(in_file)
        
        success,image = vidcap.read()
            
        count = 0

        while count< self._images_per_file:
                    
            RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
            res = cv2.resize(RGB_img, dsize=(self.img_size, self.img_size),
                                    interpolation=cv2.INTER_CUBIC)
        
            images.append(res)
        
            success,image = vidcap.read()
        
            count += 1
            
        resul = np.array(images)
        
        resul = (resul / 255.).astype(np.float16)
            
        return resul

    def proces_transfer(self, vid_names, in_dir, labels):
    
        count = 0
        
        tam = len(vid_names)
        
        # Pre-allocate input-batch-array for images.
        shape = (self._images_per_file,) + self.img_size_touple + (3,)
        
        while count<tam:
            
            video_name = vid_names[count]
            
            image_batch = np.zeros(shape=shape, dtype=np.float16)
        
            image_batch = self.get_frames(in_dir, video_name)
            
            # Note that we use 16-bit floating-points to save memory.
            shape = (self._images_per_file, self.transfer_values_size)
            transfer_values = np.zeros(shape=shape, dtype=np.float16)
            
            transfer_values = \
                self.image_model_transfer.predict(image_batch)
            
            labels1 = labels[count]
            
            aux = np.ones([20,2])
            
            labelss = labels1*aux
            
            yield transfer_values, labelss
            
            count+=1





    def process_alldata_test(self):
    
        joint_transfer=[]
        frames_num=20
        count = 0
        
        with h5py.File('pruebavalidation.h5', 'r') as f:
                
            X_batch = f['data'][:]
            y_batch = f['labels'][:]

        for i in range(int(len(X_batch)/frames_num)):
            inc = count+frames_num
            joint_transfer.append([X_batch[count:inc],y_batch[count]])
            count =inc
            
        data =[]
        target=[]
        
        for i in joint_transfer:
            data.append(i[0])
            target.append(np.array(i[1]))
            
        return data, target
        
    def detect(self):

        self.model.summary()
        # data_test, target_test = self.process_alldata_test()
        # self.model.evaluate(np.array(data_test), np.array(target_test))



