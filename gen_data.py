import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Multiply, Embedding
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU
#from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.utils import load_img, img_to_array
from keras.optimizers import Adam
import matplotlib
import matplotlib.pyplot as plt
import sys, os, glob
import numpy as np
import pandas as pd
import pydicom
from PIL import Image

#savefileのメモリエラー回避用
matplotlib.use('Agg')

class gen_data():  
    def __init__(self) -> None:
        # パラメータ
        #======================================
        # 256*256 1channels(Gray scale)を入力
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1 #RGBは3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)  

        #元データフォルダ
        self.dir = "./DICOMdata"
        #self.trainlist = pd.read_csv("./mass_train.txt", header=0)
        self.trainlist = glob.glob(self.dir + "/*.dcm")
        
        # numpyデータファイル
        self.gen_data_f1 = './gen_data1.npy'
        self.gen_data_f2 = './gen_data2.npy'

        self.train_data = np.zeros((len(self.trainlist), self.img_rows, self.img_cols, self.channels), dtype="float32")

        self.files = glob.glob(self.dir + "/*.dcm") 
        #======================================
    
    def gen1(self):
        # dicom import and generate numpy
        #======================================
        for i in range(0,len(self.trainlist)):
            dicomdata = pydicom.dcmread(self.trainlist[i])
            tmp = np.zeros((dicomdata.Rows, dicomdata.Columns), dtype="float32")
            tmp = dicomdata.pixel_array/32767.5 - 1.
            img = Image.fromarray(tmp)
            img_resize = img.resize((self.img_rows, self.img_cols),Image.LANCZOS)
            tmp2 = img_to_array(img_resize)
            self.train_data[i] = tmp2.reshape(self.img_shape)
        self.train_data = np.array(self.train_data)
        np.save(self.gen_data_f1, self.train_data)
        print("convert complete!")
        return
        #======================================

    def gen2(self):
        # dicom import and generate numpy
        #======================================
        for i in range(0,len(self.trainlist)):
            dicomdata = pydicom.dcmread(self.trainlist[i])
            tmp = np.zeros((dicomdata.Rows, dicomdata.Columns), dtype="float32")
            tmp = dicomdata.pixel_array/32767.5 - 0.5
            img = Image.fromarray(tmp)
            img_resize = img.resize((self.img_rows, self.img_cols),Image.LANCZOS)
            tmp2 = img_to_array(img_resize)
            self.train_data[i] = tmp2.reshape(self.img_shape)
        self.train_data = np.array(self.train_data)
        np.save(self.gen_data_f2, self.train_data)
        print("convert complete!")
        return
        #======================================


if __name__ == '__main__':
    gen = gen_data()
    gen.gen1()