"""
# coding=utf-8
# Author: Junjie Wang
# School: Xiamen University
# Time: 2021-7-28
"""
from numpy import ndarray
import numpy as np
import matplotlib.image as c_image
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import os
from PIL import Image
import h5py
from skimage.transform import resize
import torch

component_classes = ['Nonbridge', 'Slab', 'Beam', 'Column', 'Nonstructural components', 'Rail', 'Sleeper',
                     'Other components']

component_colormap = [[153, 51, 250], [106, 90, 205], [51, 161, 201], [127, 255, 212],
                      [255, 255, 0], [0, 255, 127], [124, 252, 0], [128, 128, 128]]

def read_image(ftrain, idx_row, path_ds):
    N = len(idx_row)
    data_list, label_list = [], []
    for i in range(N):
        idx = idx_row[i]
        imageName = os.path.join(path_ds, ftrain.iloc[idx][0])
        labelName = os.path.join(path_ds, ftrain.iloc[idx][1])
        input_array = c_image.imread(imageName)
        data_list.append(imageName)
        label_list.append(labelName)
    return data_list, label_list


def h5_save_data(file, x, y):
    print("Saving.....Please be patient.")
    h5f = h5py.File(file, 'w')
    h5f['image'] = x
    h5f['label'] = y
    print("It's over")

class LabelProcessor:
    """对标签图像进行编码"""
    def __init__(self, classes, colormap):
        self.classes = classes
        self.colormap = colormap
        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def encode_label_pix(colormap):  # 哈希算法
        cm2lbl = np.zeros(256**3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')

class MyDataset(Dataset):
    def __init__(self, images_list, labels_list, crop_size=None):
        self.images_list = images_list
        self.labels_list = labels_list
        self.crop_size = crop_size

    def __getitem__(self, index):
        img = self.images_list[index]
        label = self.images_list[index]
        img = Image.open(img)
        label = Image.open(label).convert("RGB")
        img, label = self.center_crop(img, label, self.crop_size)
        img, label = self.transform(img, label)
        sample = {'img': img, 'label': label}
        return sample

    def center_crop(self, img, label, crop_size):
        """进行中心裁剪"""
        img = F.center_crop(img, crop_size)
        label = F.center_crop(label, crop_size)
        return img, label

    def transform(self, img, label):
        """对图片进行预处理"""
        label = np.array(label)
        label = Image.fromarray(label.astype('uint8'))
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406],
            #                      [0.229, 0.224, 0.225])
        ])
        img = img_transform(img)
        label = label_process.encode_label_img(label)
        label = torch.from_numpy(label)
        return img, label

    def __len__(self):
        return len(self.images_list)

label_process = LabelProcessor(component_classes, component_colormap)

if __name__ == '__main__':
    crop_size = (352, 640)

    label_process = LabelProcessor(component_classes, component_colormap)
    # 读取图像
    path_ds = os.path.join('..', '数据', 'Tokaido_dataset')
    ftrain = pd.read_csv(os.path.join(path_ds, 'files_train.csv'), header=None, index_col=None, delimiter=',')

    # 读取csv文件中为True的图片
    col_valid = ftrain[5]
    idx_valid = [i for i in range(len(col_valid)) if col_valid[i]]
    X_train_path, Y_train_path = read_image(ftrain, idx_valid[:20], path_ds)
    X_val_path, Y_val_path = read_image(ftrain, idx_valid[50:70], path_ds)
    X_test_path, Y_test_path = read_image(ftrain, idx_valid[70:100], path_ds)

    Train_Data = MyDataset(X_train_path, Y_train_path, crop_size)
    Val_Data = MyDataset(X_val_path, Y_val_path, crop_size)
    Test_Data = MyDataset(X_test_path, Y_test_path, crop_size)

    train_loader = DataLoader(Train_Data, batch_size=4, shuffle=True, num_workers=2)

    for i, sample in enumerate(train_loader):
        plt.figure(i, figsize=(12, 8))
        img = sample['img']
        label = sample['label']
        for j in range(4):
            plt.subplot(2, 4, 2 * j + 1)
            plt.imshow(img[j].permute(1, 2, 0))
            plt.subplot(2, 4, 2 * j + 2)
            plt.imshow(label[j])
        plt.show()
        print('batch[%d]:' %i)
        print(sample['img'].shape)
        print(sample['label'].shape)


    # X_val, Y_val = read_image(ftrain, idx_valid[20:25], path_ds)
    # plt.figure(figsize=(12, 8))
    # plt.subplot(2, 2, 1)
    # plt.imshow(X_train[6, :, :, :])
    # plt.subplot(2, 2, 2)
    # plt.imshow(Y_train[6])
    # plt.subplot(2, 2, 3)
    # plt.imshow(X_train[4, :, :, :])
    # plt.subplot(2, 2, 4)
    # plt.imshow(Y_train[4])
    # plt.show()

    # file = "Train_data.h5"
    # h5_save_data(file, X_train, Y_train)
    #
    # file1 = "Val_data.h5"
    # h5_save_data(file1, X_val, X_val)

