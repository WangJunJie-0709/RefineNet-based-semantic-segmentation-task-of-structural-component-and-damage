from train import plot_confusion
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
from model import FCN_VGG16, ResNet50, Block, ResNet_FCN, rf50
import h5py
import pandas as pd
from load_data import MyDataset
from evalution import *
import pickle
import os
from train import bag_data
from evalution import eval_semantic_segmentation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Val_loader = bag_data("Val_data.pickle", batch_size=1)
model = rf50(3).to(device)  # 构件识别类别为8，损伤识别为3
model.load_state_dict(torch.load('xxx.pth'))  # 加载训练好的模型权重

model.eval()
val_confusion = [[0] * 3 for _ in range(3)]  # 结构构件识别是8，损伤识别是3
for i, sample in enumerate(Val_loader):
    X_Val = Variable(sample['img'].type(torch.FloatTensor).to(device))
    Y_Val = Variable(sample['label'].to(device))

    output = model(X_Val)
    output = F.log_softmax(output, dim=1)

    pre_label = output.max(dim=1)[1].data.cpu().numpy()
    pre_label = [i for i in pre_label]

    true_label = Y_Val.data.cpu().numpy()
    true_label = [i for i in true_label]

    val_metric = eval_semantic_segmentation(pre_label, true_label)
    val_confusion += val_metric['confusion'][:3, :3]

plot_confusion(val_confusion, type='damage', train=False)