"""
# coding=utf-8
# Author: Junjie Wang
# School: Xiamen University
# Time: 2021-7-28
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import os
from PIL import Image
import torch
import pickle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image_path(ftrain, idx_row, path_ds, event='component'):
    N = len(idx_row)
    data_path, label_path = [], []
    labelName = ''
    for i in range(N):
        idx = idx_row[i]
        imageName = os.path.join(path_ds, ftrain.iloc[idx][0])
        if event == 'component':
            labelName = os.path.join(path_ds, ftrain.iloc[idx][1])
        elif event == 'damage':
            labelName = os.path.join(path_ds, ftrain.iloc[idx][2])
        data_path.append(imageName)
        label_path.append(labelName)
    return data_path, label_path

def read_image_test_path(ftest, idx_row, path_ds):
    N = len(idx_row)
    data_path = []
    for i in range(N):
        idx = idx_row[i]
        imageName = os.path.join(path_ds, ftest.iloc[idx][0])
        data_path.append(imageName)
    return data_path

class MyDataset(Dataset):
    def __init__(self, images_path, labels_path, crop_size=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.crop_size = crop_size

    def __getitem__(self, index):
        img = self.images_path[index]
        label = self.labels_path[index]
        img = Image.open(img).convert('RGB')
        label = Image.open(label)
        img, label = self.img_transform(img, label)
        label = label - 1
        label[label >= 3] = 0  # 如果是结构构件识别就是8， 损伤识别是3
        sample = {'img': img, 'label': label}
        return sample

    def img_transform(self, img, label):
        transform_img = transforms.Compose(
            [
                transforms.Resize((352, 640), interpolation=2),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        transform_label = transforms.Compose(
            [
                transforms.Resize((352, 640), interpolation=2),
            ]
        )
        img = transform_img(img)
        label = transform_label(label)
        label = np.array(label)
        label = torch.from_numpy(label)
        return img, label

    def center_crop(self, data, label, crop_size):
        """裁剪输入的图片和标签大小"""
        data = F.center_crop(data, crop_size)
        label = F.center_crop(label, crop_size)
        return data, label

    def __len__(self):
        return len(self.images_path)

class MytestDataset(Dataset):
    """因为缺少测试标签，故仅将测试样本整合为一个测试数据集，图像处理方式与训练数据集一致"""
    def __init__(self, images_path, crop_size=None):
        self.images_path = images_path
        self.crop_size = crop_size

    def __getitem__(self, index):
        img = self.images_path[index]
        img = Image.open(img)
        img = self.img_transform(img)
        sample = {'img': img}
        return sample

    def img_transform(self, img):
        transform_img = transforms.Compose(
            [
                transforms.Resize((352, 640), interpolation=2),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform_img(img)
        return img

    def center_crop(self, data, crop_size):
        """裁剪输入的图片和标签大小"""
        data = F.center_crop(data, crop_size)
        return data

    def __len__(self):
        return len(self.images_path)

def pickle_save_data(file, Data):
    print("Saving.....Please be patient.")
    dict_ = {'Dataset': Data}
    File = open(file, 'wb')
    pickle.dump(dict_, File)
    File.close()
    print("It's over")

if __name__ == '__main__':
    # 读取图像
    path_ds = os.path.join('..', '数据', 'Tokaido_dataset')
    model = 'damage'
    if model == 'component':
        ftrain = pd.read_csv(os.path.join(path_ds, 'files_train.csv'), header=None, index_col=None, delimiter=',')  # 训练集的文件路径
        ftest = pd.read_csv(os.path.join(path_ds, 'files_test.csv'), header=None, index_col=None, delimiter=',')  # 测试集的文件路径

        # 训练数据部分，读取csv文件中为True的图片
        col_valid = ftrain[5]
        idx_valid = [i for i in range(len(col_valid)) if col_valid[i]]  # 构件识别7574个训练样本， 损伤识别4380个训练样本

        # 训练集和验证集的划分
        X_train, Y_train = read_image_path(ftrain, idx_valid[:7275], path_ds, event=model)  # 构件识别的训练集7275个图像
        X_val, Y_val = read_image_path(ftrain, idx_valid[7275:], path_ds, event=model)  # 构件识别的验证集300个图像

        # 测试数据部分，读取csv文件中为True的图片
        test_valid = ftest[5]
        idx_test = [i for i in range(len(test_valid)) if test_valid[i]]  # 1073个测试样本
        X_test = read_image_test_path(ftest, idx_test[:], path_ds)

        """构建划分训练、验证和测试数据集"""
        Train_Data = MyDataset(X_train, Y_train)
        Val_Data = MyDataset(X_val, Y_val)
        Test_Data = MytestDataset(X_test)

        # train_loader = DataLoader(Train_Data, batch_size=4, shuffle=True, num_workers=1)
        #
        # for i, sample in enumerate(train_loader):  # 可以看批次的图像和标签情况
        #     plt.figure(i, figsize=(12, 8))
        #     img, label = sample['img'], sample['label']
        #     plt.imshow(img[0].permute(1, 2, 0))
        #     plt.show()
        #     plt.imshow(label[0])
        #     plt.show()

        file = "Train_data.pickle"
        pickle_save_data(file, Train_Data)

        file1 = "Val_data.pickle"
        pickle_save_data(file1, Val_Data)

        file2 = "Test_data.pickle"
        pickle_save_data(file2, Test_Data)

    elif model == 'damage':
        ftrain1 = pd.read_csv(os.path.join(path_ds, 'files_train.csv'), header=None, index_col=None, delimiter=',')
        ftrain2 = pd.read_csv(os.path.join(path_ds, 'files_puretex_train.csv'), header=None, index_col=None, delimiter=',')
        ftest1 = pd.read_csv(os.path.join(path_ds, 'files_test.csv'), header=None, index_col=None, delimiter=',')
        ftest2 = pd.read_csv(os.path.join(path_ds, 'files_puretex_test.csv'), header=None, index_col=None, delimiter=',')

        col_valid = ftrain1[6]
        idx_valid = [i for i in range(len(col_valid)) if col_valid[i]]

        X, Y = read_image_path(ftrain1, idx_valid[:4380], path_ds, event=model)

        test_valid = ftest1[6]
        idx_test = [i for i in range(len(test_valid)) if test_valid[i]]
        X_test = read_image_test_path(ftest1, idx_test[:], path_ds)

        for i in range(len(ftrain2)):
            X.append(os.path.join(path_ds, ftrain2.iloc[i][0]))
            Y.append(os.path.join(path_ds, ftrain2.iloc[i][1]))

        # for i in range(len(ftest2)):
        #     X_test.append(ftest2.iloc[i][0])

        X_train, X_val = X[:6780], X[6780:]
        Y_train, Y_val = Y[:6780], Y[6780:]

        Train_Data = MyDataset(X_train, Y_train)
        Val_Data = MyDataset(X_val, Y_val)
        Test_Data = MytestDataset(X_test)

        # train_loader = DataLoader(Train_Data, batch_size=4, shuffle=True, num_workers=1)
        #
        # for i, sample in enumerate(train_loader):  # 可以看批次的图像和标签情况
        #     plt.figure(i, figsize=(12, 8))
        #     img, label = sample['img'], sample['label']
        #     plt.imshow(img[0].permute(1, 2, 0))
        #     plt.show()
        #     plt.imshow(label[0])
        #     plt.show()

        file = "Train_data.pickle"
        pickle_save_data(file, Train_Data)

        file1 = "Val_data.pickle"
        pickle_save_data(file1, Val_Data)

        file2 = "Test_data.pickle"
        pickle_save_data(file2, Test_Data)