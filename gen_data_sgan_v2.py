import matplotlib
import matplotlib.pyplot as plt
import glob
import numpy as np
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import img_to_array
from keras.utils import to_categorical
from inspect import currentframe


#訓練とテストの為のデータの整形

class gen_data():
    def __init__(self) -> None:
        # パラメータ
        #======================================
        # 256*256 1channels(Gray scale)を入力
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1 #RGBは3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_labeled = 100 #ラベル付きデータ数

        #元データフォルダ
        self.dir_1 = "./DICOMdata_Nodule"     #154枚
        self.dir_0 = "./DICOMdata_NonNodule"  #93枚
        self.trainlist_1 = glob.glob(self.dir_1 + "/*.dcm")
        self.trainlist_0 = glob.glob(self.dir_0 + "/*.dcm")
        
        # numpyデータファイル
        self.gen_data = './gen_data_sgan1.npy'

        self.files_1 = glob.glob(self.dir_1 + "/*.dcm") # image
        self.files_0 = glob.glob(self.dir_0 + "/*.dcm") # label

        self.train_data_1 = np.zeros((len(self.trainlist_1), self.img_rows, self.img_cols, self.channels), dtype="float32")
        self.train_data_0 = np.zeros((len(self.trainlist_0), self.img_rows, self.img_cols, self.channels), dtype="float32")
        #======================================

        def preprocess_imgs(x):
            #x = (x.astype(np.float32) - 32767.5) / 32767.5
            x = (x.astype(np.float32) /32767.5) - 1.
            return x
        
        def read_dicom(path):
            try:
                d = pydicom.dcmread(path)
                np_d = d.pixel_array
                return np_d
            except Exception as e:
                print(f"Error reading {path}: {e}")
                return np.zeros((self.img_rows, self.img_cols), dtype="float32")  # 空の画像配列を返す
        
        def resize_img(img):
            img = Image.fromarray(img)
            img_resize = img.resize((self.img_rows, self.img_cols), Image.LANCZOS)
            img_array = img_to_array(img_resize)

            # グレースケール画像の場合、チャンネル次元を1にする
            if self.channels == 1 and img_array.shape[2] == 3:
                # RGBの3チャンネルの平均を取るか、単に1チャンネル目を選択する
                img_array = img_array[:, :, 0:1]

            return img_array
        
        def show_property(d):
            print('======================================')
            #print('file: ', currentframe().f_back.f_code.co_name)
            print('type: ', type(d))
            print('shape: ', d.shape)
            print('======================================')
            #plt.imshow(d)
            #plt.show()
            return
        
        def img_show(d):
            '''
            d : [x, y]
            '''
            plt.imshow(d, cmap='gray')
            plt.show()
            return

        # dicom import and generate numpy
        #======================================
        for i in range(0,len(self.trainlist_1)):
            dicomdata_1 = read_dicom(self.trainlist_1[i])
            dicomdata_1 = preprocess_imgs(dicomdata_1)
            dicomdata_1 = resize_img(dicomdata_1)
            self.train_data_1[i] = dicomdata_1.reshape(self.img_shape)
        #show_property(self.train_data_1)
        self.labels_1 = to_categorical(np.full(len(self.trainlist_1), 1), num_classes=2)

        for i in range(0,len(self.trainlist_0)):
            dicomdata_0 = read_dicom(self.trainlist_0[i])
            dicomdata_0 = preprocess_imgs(dicomdata_0)
            dicomdata_0 = resize_img(dicomdata_0)
            self.train_data_0[i] = dicomdata_0.reshape(self.img_shape)
        #show_property(self.train_data_0)
        self.labels_0 = to_categorical(np.full(len(self.trainlist_0), 0), num_classes=2)

        self.train_data_x = np.concatenate([self.train_data_1, self.train_data_0])
        self.train_data_y = np.concatenate([self.labels_1, self.labels_0])
        #show_property(self.train_data_x)
        #show_property(self.train_data_y)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.train_data_x, self.train_data_y, test_size=0.2, random_state=42)
        show_property(self.x_train)
        show_property(self.x_test)
        show_property(self.y_train)
        show_property(self.y_test)

        print("@@@@@@@")

        dataset = np.array([self.x_train, self.x_test, self.y_train, self.y_test])
        np.save(self.gen_data, dataset)
        show_property(dataset)
        show_property(dataset[0])
        show_property(dataset[1])
        show_property(dataset[2])
        show_property(dataset[3])
        print("convert complete!")


if __name__ == '__main__':
    dataset = gen_data()

        #======================================

    '''
    def batch_labeled(self, batch_size):
        idx = np.random.randint(0, self.num_labeled, batch_size)
        imgs = self.x_train[idx]
        labels = self.y_train[idx]
        return imgs, labels

    def batch_unlabeled(self, batch_size):
        idx = np.random.randint(self.num_labeled, self.x_train.shape[0], batch_size)
        imgs = self.x_train[idx]

        return imgs

    def training_set(self):
        x_train = self.x_train[range(self.num_labeled)]
        y_train = self.y_train[range(self.num_labeled)]
        return x_train, y_train

    def test_set(self):
        return self.x_test, self.y_test
    
    def batch_labeled(self, batch_size):
        idx = np.random.randint(0, self.num_labeled, batch_size)
        imgs = self.x_train[idx]
        labels = self.y_train[idx]
        return imgs, labels

    def batch_unlabeled(self, batch_size):
        idx = np.random.randint(self.num_labeled, self.x_train.shape[0], batch_size)
        imgs = self.x_train[idx]

        return imgs

    def training_set(self):
        x_train = self.x_train[range(self.num_labeled)]
        y_train = self.y_train[range(self.num_labeled)]
        return x_train, y_train

    def test_set(self):
        return self.x_test, self.y_test
    '''
