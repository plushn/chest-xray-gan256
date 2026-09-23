import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Multiply, Embedding
from keras.layers import BatchNormalization, Activation, Concatenate
from keras.layers import LeakyReLU
#from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import matplotlib
import matplotlib.pyplot as plt
import sys, os
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

class CGAN():

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
        self.num_classes = 2 #nodules or non-nodules
        self.num_labeled = 100 #100から変更
        # (lr=0.0002, beta_1=0.5)
        optimizer = Adam(0.0002, 0.5)
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

        # Generator model
        self.generator = self.build_generator()

        # combined
        self.combined = self.build_combined()
        self.combined.compile(loss='binary_crossentropy', 
                                    optimizer=optimizer)

    # Generatorモデル
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

    def build_cgan_generator(self):

        z_dim = self.z_dim

        z = Input(shape=(z_dim, ))
        label = Input(shape=(1, ), dtype="int32")

        #Embedding層を用いて、ラベルの埋め込みを行う
        #ラベルをz_dim次元の密ベクトルに変換する
        label_embedding = Embedding(self.num_classes, z_dim, input_length=1)(label)
        label_embedding = Flatten()(label_embedding)

        #ベクトルzと、ラベルが埋め込まれたベクトルの、要素ごとの掛け算を行う
        joined_embedding = Multiply()([z, label_embedding])

        generator = self.build_generator()
        conditioned_img = generator(joined_embedding)

        return Model([z, label], conditioned_img)
    
    # Discriminatorモデル
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
    
    
    def build_cgan_discriminator(self):

        img_shape = self.img_shape    

        img = Input(shape=img_shape)
        label = Input(shape=(1, ), dtype="int32")

        label_embedding = Embedding(self.num_classes, np.prod(img_shape), input_length=1)(label)
        label_embedding = Flatten()(label_embedding)
        label_embedding = Reshape(img_shape)(label_embedding)

        #画像と、ラベルが埋め込まれたテンソルを結合する
        concatenated = Concatenate(axis=-1)([img, label_embedding])

        discriminator = self.build_discriminator(img_shape)
        classification = discriminator(concatenated)

        return Model([img, label], classification)
    
    # このコンパイルではできない(171行目はmodelを返さないので)
    def build_combined(self):
        self.build_cgan_discriminator.trainable = False
        model = Sequential([self.build_cgan_generator, self.build_cgan_discriminator])
        model.compile(loss='binary_crossentropy', optimizer=Adam())

        return model

    def train(self, epochs, batch_size=128, save_interval=50):

        num_batches = int(self.x_train.shape[0] / batch_size)

        generator = self.build_cgan_generator()
        discriminator = self.build_cgan_discriminator()

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        x_train = self.x_train
        y_train = self.y_train
        z_dim = self.z_dim
        num_classes = self.num_classes

        for epoch in range(epochs):
            for batch in range(num_batches):

                #識別器の訓練
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                imgs, labels = x_train[idx], x_train[idx]

                z = np.random.normal(0, 1, (batch_size, z_dim))
                gen_imgs = generator.predict([z, labels])

                d_loss_real = discriminator.train_on_batch([imgs, labels], real)
                d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                #生成器の訓練
                z = np.random.normal(0, 1, (batch_size, z_dim))
                labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)

                g_loss = self.build_combined.train_on_batch([z, labels], real)

                print("epoch:%d [D loss supervised: %.4f, acc.: %.2f%%] [G loss: %f]" % (epoch + 1, d_loss[0], 100 * d_loss[1], g_loss))

            # save images 
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

                with open('./loss_sgan.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(epoch + 1, np.mean(d_loss[0]), 100 * 100*d_loss[1], g_loss)
                self.generator.save('./weights_sgan/generator_ep%02d.h5' % epoch) 
                self.discriminator_supervised.save('./weights_sgan/discriminator_supervised_ep%02d.h5' % epoch)
                self.discriminator_unsupervised.save('./weights_sgan/discriminator_unsupervised_ep%02d.h5' % epoch)

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
        fig.savefig('./generated_images_sgan/generatedimg_%02d.png' % epoch) 
        plt.close()
        #self.generator.save('./weights_cgan/generator_ep%02d.h5' % epoch) 
        #self.discriminator.save('./weights_cgan/discriminator_ep%02d.h5' % epoch)


if __name__ == '__main__':
    gan = CGAN()
    gan.train(epochs=300000, batch_size=32, save_interval=100)  #epochは30000くらい必要?