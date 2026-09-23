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
import sys, os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import csv

#savefileのメモリエラー回避用
matplotlib.use('Agg')

class DCGAN():

    def __init__(self) -> None:
        # パラメータ
        #======================================
        # 256*256 1channels(Gray scale)を入力
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1 #RGBは3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)  

        # 潜在変数の次元数
        self.z_dim = 100
        # (lr=0.0002, beta_1=0.5)
        optimizer = Adam(0.0002, 0.5)
        #======================================

        # file import
        #======================================
        #self.train_data.shape => (n, 256, 256, 1)
        self.train_data = np.load("./gen_data1.npy", allow_pickle=True) 
        #======================================

        # discriminator model
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Generator model
        self.generator = self.build_generator()

        # combined
        self.combined = self.build_combined()
        self.combined.compile(loss='binary_crossentropy', 
                                    optimizer=optimizer)
        
    
    def build_generator(self):

        noise_shape = (self.z_dim,)
        model = Sequential()

        model.add(Dense(8 * 8 * 8, activation='relu', input_shape=noise_shape))
        model.add(Reshape((8, 8, 8)))
        model.add(UpSampling2D())#16x16
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(BatchNormalization(momentum=0.8)) 
        model.add(Activation("relu"))
        model.add(UpSampling2D())#32x32
        model.add(Conv2D(256,(3,3), padding="same")) 
        model.add(BatchNormalization(momentum=0.8)) 
        model.add(Activation("relu"))
        model.add(UpSampling2D())#64x64
        model.add(Conv2D(128,(3,3), padding="same")) 
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())#128x128
        model.add(Conv2D(64,(3,3), padding="same")) 
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())#256x256
        model.add(Conv2D(self.channels,(3,3), padding="same")) 
        model.add(Activation("tanh"))
        
        model.summary()

        return model
    
    def build_discriminator(self):

        img_shape = self.img_shape
        model = Sequential()

        model.add(Conv2D(64,(3,3), strides=2, input_shape=img_shape, padding="same"))#128x128
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64,(3,3), strides=2, padding="same"))#64x64
        model.add(LeakyReLU(alpha=0.2)) 
        model.add(Dropout(0.25))
        model.add(Conv2D(128,(3,3), strides=2, padding="same"))#32x32
        model.add(LeakyReLU(alpha=0.2)) 
        model.add(Dropout(0.25))
        model.add(Conv2D(128,(3,3), strides=2, padding="same"))#16x16
        model.add(LeakyReLU(alpha=0.2)) 
        model.add(Dropout(0.25))
        model.add(Conv2D(256,(3,3), strides=2, padding="same"))#8x8
        model.add(LeakyReLU(alpha=0.2)) 
        model.add(Dropout(0.25))
        model.add(Conv2D(256,(3,3), strides=2, padding="same"))#4x4
        model.add(LeakyReLU(alpha=0.2)) 
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        model.summary()
        
        return model
    
    def build_combined(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        return model
    
    def train(self, epochs, batch_size=128, save_interval=50):

        num_batches = int(self.train_data.shape[0] / batch_size)

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            for batch in range(num_batches):

                # ---------------------
                #  Discriminator learning
                # ---------------------

                input_noise = np.random.normal(0,1,(batch_size, self.z_dim))
                gen_imgs = self.generator.predict(input_noise)
                real_imgs= self.train_data[batch*batch_size:(batch+1)*batch_size]
                
                d_loss_real = self.discriminator.train_on_batch(real_imgs, real)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Generator learning
                # ---------------------

                g_loss = self.combined.train_on_batch(input_noise, real)
                
                # progress
                #print("epoch: %d \td_loss: %f \tg_loss: %f" %(epoch, np.mean(d_loss), np.mean(g_loss)))
                print ("epoch:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, np.mean(d_loss), 100*d_loss[1], np.mean(g_loss)))

            # save images 
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                with open('./loss.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, np.mean(d_loss), 100*d_loss[1], np.mean(g_loss)])
                self.generator.save('./weights/generator_ep%02d.h5' % epoch) 
                self.discriminator.save('./weights/discriminator_ep%02d.h5' % epoch)

    def save_imgs(self, epoch):
        # row,col
        r, c = 2, 5

        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = gen_imgs * 127.5 + 127.5
        gen_imgs = gen_imgs.astype('uint8')

        fig, axs = plt.subplots(r, c, figsize=(15,6))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt].reshape(self.img_rows, self.img_cols), cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig('./generated_images/generatedimg_%02d.png' % epoch) 
        plt.close()
        self.generator.save('./weights/generator_ep%02d.h5' % epoch) 
        self.discriminator.save('./weights/discriminator_ep%02d.h5' % epoch)


if __name__ == '__main__':
    gan = DCGAN()
    gan.train(epochs=100000, batch_size=32, save_interval=50)  #epochは30000くらい必要?