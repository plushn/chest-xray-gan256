import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
import torch.optim as optim
from functools import Lambda
from keras.optimizers import Adam
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import csv

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
        # `use_pytorch=True` を指定する必要があります
        optimizer = Adam(learning_rate=0.0002, beta_1=0.5, use_pytorch=True)

        #======================================
        # file import
        #======================================
        data = np.load("./gen_data_sgan1.npy", allow_pickle=True) 
        device = torch.device('mps')
        # Macの場合、inputデータの設定を、GPUのメモリに移動  
        input = torch.tensor(data, dtype=torch.float)
        self.train_data = input.to(device)
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
        z_dim = self.z_dim

        model = nn.Sequential(
            # Initial dense layer
            nn.Linear(z_dim, 8 * 8 * 8),
            nn.ReLU(),

            # Reshape into 3D 'image' 
            nn.Reshape((8, 8, 8)),

            # Transposed convolutions (for upsampling)
            nn.ConvTranspose2d(8, 128, (3, 3), stride=2, padding=1, output_padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 256, (3, 3), stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, (3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, (3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, self.channels, (3, 3), stride=2, padding=1, output_padding=1), 
            nn.Tanh()
            )

        return model
    
    def build_discriminator_net(self):
        img_shape = self.img_shape

        model = nn.Sequential(
            nn.Conv2d(img_shape[0], 64, (3, 3), stride=2, padding=1),  # 128x128
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 64, (3, 3), stride=2, padding=1),  # 64x64
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, (3, 3), stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 128, (3, 3), stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, (3, 3), stride=2, padding=1),  # 8x8
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 256, (3, 3), stride=2, padding=1),  # 4x4
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),  # Adjust if input shape requires
            nn.Sigmoid()
        )
        
        return model
    
    def build_discriminator_supervised(self):
        model = nn.Sequential(
            self.build_discriminator_net(),
            nn.Linear(256 * 4 * 4, 2),
            nn.Softmax(dim=1)
        )

        return model
    
    def build_discriminator_unsupervised(self):

        model = nn.Sequential()
        model.add_module('discriminator_net', self.build_discriminator_net())

        def predict(x):
            """
            10ニューロンの出力大きい（つまりニューロンが入力画像を本物だと知覚した)時、predictionは1.0に近づく。
            逆に、10ニューロンの出力が全て小さい（つまりニューロンが入力画像を偽物だと知覚した）時、predictionは0に近づく。
            """
            prediction = 1.0 - (1.0 / (torch.sum(torch.exp(x), dim=-1, keepdim=True) + 1.0))
            return prediction

        model.add_module('prediction', Lambda(predict))

        return model
    
    def build_combined(self):
        # Freeze parameters of the unsupervised discriminator
        for param in self.discriminator_unsupervised.parameters():
            param.requires_grad = False

        # Construct the combined model
        self.combined_model = nn.Sequential(self.generator, self.discriminator_unsupervised)

        # Optimizer and loss function
        self.combined_optimizer = torch.optim.Adam(self.combined_model.parameters())
        self.loss_fn = nn.BCELoss()

        return self.combined_model
    
    def batch_labeled(self, batch_size):
        idx = torch.randint(0, self.num_labeled, (batch_size,))  # Generate random indices
        imgs = self.x_train[idx].astype(torch.float32)  # Select images and change dtype 
        labels = self.y_train[idx]

        # If self.x_train and self.y_train are NumPy arrays:
        return torch.from_numpy(imgs), torch.from_numpy(labels)
        # If self.x_train and self.y_train are NumPy arrays:
        #return imgs, labels 

    def batch_unlabeled(self, batch_size):
        idx = torch.randint(self.num_labeled, self.x_train.shape[0], (batch_size,))  
        imgs = self.x_train[idx].astype(torch.float32)  # Adjust dtype to float32


        # If self.x_train is a NumPy array:
        return torch.from_numpy(imgs)
        # Assuming self.x_train is already a PyTorch tensor:
        #return imgs

    def training_set(self):
        x_train = torch.from_numpy(self.x_train[range(self.num_labeled)])
        y_train = torch.from_numpy(self.y_train[range(self.num_labeled)]) 
        return x_train, y_train
    '''
    def training_set(self):
        x_train = self.x_train[:self.num_labeled]  # Assuming x_train is a tensor
        y_train = self.y_train[:self.num_labeled]  # Assuming y_train is a tensor
        return x_train, y_train
    '''
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

    def train(self, epochs, batch_size=128, save_interval=50):

        num_batches = len(self.x_train) // batch_size  # Assumes data is loaded as PyTorch tensors

        real_label = torch.ones((batch_size, 1), device=self.device)  
        fake_label = torch.zeros((batch_size, 1), device=self.device) 

        for epoch in range(epochs):
            for batch_idx in range(num_batches):

                # Labeled samples
                imgs, labels = self.batch_labeled(batch_size)
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                # Unlabeled samples          
                imgs_unlabeled = self.batch_unlabeled(batch_size).to(self.device) 

                # Generate fake images
                z = np.random.normal(0, 1, ((batch_size, self.z_dim)))
                gen_imgs = self.generator(z)

                # Discriminator updates
                self.discriminator_supervised.zero_grad()
                d_loss_supervised, accuracy = self.discriminator_supervised(imgs, labels)
                d_loss_supervised.backward()
                self.optimizer_d_supervised.step()

                self.discriminator_unsupervised.zero_grad()
                d_loss_real = self.discriminator_unsupervised(imgs_unlabeled).mean()
                d_loss_real.backward()

                d_loss_fake = self.discriminator_unsupervised(gen_imgs).mean()
                d_loss_fake.backward()

                self.optimizer_d_unsupervised.step()

                d_loss_unsupervised = (d_loss_real + d_loss_fake) / 2

                # Generator Update
                self.generator.zero_grad()
                g_output = self.discriminator_unsupervised(gen_imgs)
                g_loss = nn.functional.binary_cross_entropy(g_output, real_label)  
                g_loss.backward()
                self.optimizer_g.step()

                # Logging
                print("epoch:%d [D loss supervised: %.4f, acc.: %.2f%%] [D loss unsupervised: %.4f] [G loss: %f]" % 
                    (epoch + 1, d_loss_supervised.item(), 100 * accuracy, d_loss_unsupervised.item(), g_loss.item()))

                # ... (Saving images and models as needed) 
            
    # Assuming you have:
    #   - self.device (specify 'cuda' or 'cpu')
    #   - self.batch_labeled, self.batch_unlabeled functions providing image batches
    #   - self.optimizer_d_supervised, self.optimizer_d_unsupervised, self.optimizer_g optimizers already defined.  



if __name__ == '__main__':
    gan = SGAN()
    # バッチサイズを小さくする
    smaller_batch_size = 80  # 128から64に変更
    gan.train(epochs=50000, batch_size=smaller_batch_size, save_interval=100)