#import os
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

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
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import csv

from keras import backend as K
from keras.layers import Lambda
from keras.utils import to_categorical

#savefileのメモリエラー回避用
matplotlib.use('Agg')

class SGAN():

    def __init__(self, img_rows: int = 256, img_cols: int = 256, channels: int = 1, z_dim: int = 100, num_classes: int = 2, num_labeled: int = 100, learning_rate: float = 0.0002, beta_1: float = 0.5) -> None:
        # パラメータ
        #======================================
        # 256*256 1channels(Gray scale)を入力
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels #RGBは3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)  

        # 潜在変数の次元数
        self.z_dim = z_dim
        self.num_classes = num_classes #nodules or non-nodules
        self.num_labeled = num_labeled #ラベル付きデータ数
        # (lr=0.0002, beta_1=0.5)
        optimizer = Adam()  #以下から変更
        #optimizer = Adam(learning_rate, beta_1)
        #======================================

        # file import
        #======================================
        self.train_data = np.load("./gen_data_sgan1.npy", allow_pickle=True) 
        # self.train_data.shape => (4,)
        self.x_train = self.train_data[0]
        self.x_test = self.train_data[1]
        self.y_train = self.train_data[2]
        self.y_test = self.train_data[3]
        # self.x_train.shape => (197, 256, 256, 1)
        # self.x_test.shape => (50, 256, 256, 1)
        # self.y_train.shape => (197, 2)
        # self.y_test.shape => (50, 2)
        #======================================

        self.discriminator_net = self.build_discriminator_net()
        
        # 教師ありdiscriminator model
        self.discriminator_supervised = self.build_discriminator_supervised()
        self.discriminator_supervised.compile(loss='categorical_crossentropy',
                                        metrics=['accuracy'],
                                        optimizer=optimizer)

        # 教師なしdiscriminator model
        self.discriminator_unsupervised = self.build_discriminator_unsupervised()
        self.discriminator_unsupervised.compile(loss='binary_crossentropy',
                                        optimizer=optimizer)

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
    
    def build_discriminator_net(self):

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
    
    def build_discriminator_supervised(self):

        model = Sequential()
        model.add(self.build_discriminator_net())
        #model.add(Activation("softmax")) # 以下に変更
        model.add(Dense(2, activation='softmax')) # 2つのニューロンとsoftmax活性化関数

        model.summary()

        return model
    
    def build_discriminator_unsupervised(self):

        model = Sequential()
        model.add(self.build_discriminator_net())
        def predict(x):
            '''
            10ニューロンの出力大きい（つまりニューロンが入力画像を本物だと知覚した)時、predictionは1.0に近づく。
            逆に、10ニューロンの出力が全て小さい（つまりニューロンが入力画像を偽物だと知覚した）時、predictionは0に近づく。
            '''
            prediction = 1.0 - (1.0 /(K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0))
            return prediction
        model.add(Lambda(predict))

        return model  
    
    def build_combined(self):
        self.discriminator_unsupervised.trainable = False
        model = Sequential([self.generator, self.discriminator_unsupervised])
        model.compile(loss='binary_crossentropy', optimizer=Adam())

        return model
    
    def batch_labeled(self, batch_size):
        idx = np.random.randint(0, self.num_labeled, batch_size)
        imgs = self.x_train[idx].astype('float32')  # float32からfloat16に変更
        labels = self.y_train[idx]
        return imgs, labels

    def batch_unlabeled(self, batch_size):
        idx = np.random.randint(self.num_labeled, self.x_train.shape[0], batch_size)
        imgs = self.x_train[idx].astype('float32')  # float32からfloat16に変更

        return imgs

    def training_set(self):
        x_train = self.x_train[range(self.num_labeled)]
        y_train = self.y_train[range(self.num_labeled)]
        return x_train, y_train

    def train(self, epochs, batch_size=128, save_interval=50):

        num_batches = int(self.x_train.shape[0] / batch_size)

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            for batch in range(num_batches):

                #ラベル付きサンプルの取得
                imgs, labels = self.batch_labeled(batch_size)

                #ラベルなしサンプルの取得
                imgs_unlabeled = self.batch_unlabeled(batch_size)

                #偽画像バッチの作成
                z = np.random.normal(0, 1, ((batch_size, self.z_dim)))
                gen_imgs = self.generator.predict(z)

                d_loss_supervised, accuracy = self.discriminator_supervised.train_on_batch(imgs, labels)

                d_loss_real = self.discriminator_unsupervised.train_on_batch(imgs_unlabeled, real)

                d_loss_fake = self.discriminator_unsupervised.train_on_batch(gen_imgs, fake)

                d_loss_unsupervised = 0.5 * np.add(d_loss_real, d_loss_fake)

                z = np.random.normal(0, 1, (batch_size, self.z_dim))
                gen_imgs = self.generator.predict(z)

                g_loss = self.combined.train_on_batch(z, real)

                print("epoch:%d [D loss supervised: %.4f, acc.: %.2f%%] [D loss unsupervised: %.4f] [G loss: %f]" % (epoch + 1, d_loss_supervised, 100 * accuracy, d_loss_unsupervised, g_loss))


            # save images 
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

                with open('./loss_sgan.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, np.mean(d_loss_supervised), d_loss_unsupervised, np.mean(g_loss), 100 * accuracy])
                #self.generator.save('./weights_sgan/generator_ep%02d.h5' % epoch) 
                #self.discriminator_supervised.save('./weights_sgan/discriminator_supervised_ep%02d.h5' % epoch)
                #self.discriminator_unsupervised.save('./weights_sgan/discriminator_unsupervised_ep%02d.h5' % epoch)

    def save_imgs(self, epoch):
        # row,col
        r, c = 2, 5

        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = gen_imgs * 127.5 + 127.5
        #gen_imgs = gen_imgs.astype('uint8') # 以下に変更
        gen_imgs = gen_imgs.astype('float32')

        fig, axs = plt.subplots(r, c, figsize=(15,6))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt].reshape(self.img_rows, self.img_cols), cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig('./generated_images_sgan/generatedimg_%02d.png' % epoch) 
        plt.close()
        #self.generator.save('./weights_sgan/generator_ep%02d.h5' % epoch) 
        #self.discriminator_unsupervised.save('./weights_sgan/discriminator_ep%02d.h5' % epoch)


if __name__ == '__main__':
    gan = SGAN()
    gan.train(epochs=50000, save_interval=100)