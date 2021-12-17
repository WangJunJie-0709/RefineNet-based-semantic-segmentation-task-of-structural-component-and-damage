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

# 加载并封存训练数据
def bag_data(file, batch_size):
    with open(file, 'rb') as f:
        dict_get = pickle.load(f)
    Data = dict_get['Dataset']
    loader = DataLoader(Data, batch_size=batch_size, shuffle=False)  # 训练数据的迭代器, 批数量为batch_size， 随机打乱
    return loader

def plot_figure(figure_num, title=None,
                x_label=None, y_label=None,
                ylegend1=None, ylegend2=None,
                x=None, y1=None, y2=None):  # 可视化训练指标变化
    plt.figure(figure_num, figsize=(10, 6))
    plt.plot(x, y1, 'r-')
    plt.plot(x, y2, 'b-')
    plt.legend(labels=[ylegend1, ylegend2])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

def plot_confusion(confusion, title='Confusion Matrix', type='component', train=True):
    firstConfusion = pd.DataFrame(confusion)
    firstConfusion.to_csv('./first_confusion.csv')
    labels = []
    if type == 'component':
        labels = ['Nonbridge', 'Slab', 'Beam', 'Column', 'Nonstructural', 'Rail', 'Sleeper']
    elif type == 'damage':
        labels = ['No Damage', 'Concrete Damage', 'Exposed Rebar']
    confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(16, 12))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.get_cmap("Blues"))
    plt.title(title)
    plt.colorbar()
    num_local = np.array(range(len(labels)))
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(second_index, first_index, round(confusion[first_index][second_index], 3), va='center', ha='center', fontsize=15)
    plt.xticks(num_local, labels, rotation=0, fontsize=12)
    plt.yticks(num_local, labels, fontsize=12)
    plt.ylabel('True label', rotation=90)
    plt.xlabel('Predict label')
    if train:
        confusion = pd.DataFrame(confusion)
        confusion.to_csv('./train_confusion.csv')
        plt.savefig('./train_confusion.jpg')
    else:
        confusion = pd.DataFrame(confusion)
        confusion.to_csv('./eval_confusion.csv')
        plt.savefig('./eval_confusion.jpg')
    plt.show()


# 训练模型
def train_model(model, Train_loader, Val_loader, loss_fn, optimizer, num_epochs=2):

    best = [0]  # 保存最好指标
    train_loss_all, eval_loss_all = [], []
    train_miou_all, eval_miou_all = [], []
    train_iou_all, eval_iou_all = [], []
    train_acc_all, eval_acc_all = [], []
    train_class_acc_all, eval_class_acc_all = [], []

    model.train()  # 训练模式开始
    for epoch in range(num_epochs):
        prec_time = datetime.now()  # 记录起始时间
        print("第{}轮训练开始, 共{}轮".format(epoch+1, num_epochs))
        if epoch % 40 == 0 and epoch != 0:  # 衰减学习率
            for group in optimizer.param_groups:
                group['lr'] *= 0.5

        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0
        train_iou = 0

        # 开始训练
        for i, sample in enumerate(Train_loader):
            X_train = Variable(sample['img'].type(torch.FloatTensor).to(device))
            Y_train = Variable(sample['label'].to(device))
            output = model(X_train)
            a = F.log_softmax(output, dim=1)
            loss = loss_fn(a, Y_train.long())

            # 优化器优化模型
            optimizer.zero_grad()  # 先梯度清零
            loss.backward()
            optimizer.step()

            # 指标评估
            train_loss += loss.item()

            pre_label = output.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = Y_train.data.cpu().numpy()
            true_label = [i for i in true_label]

            # 混淆矩阵计算
            train_metric = eval_semantic_segmentation(pre_label, true_label)
            train_confusion = train_metric['confusion']
            train_acc += train_metric['mean_class_accuracy']
            train_miou += train_metric['miou']
            train_iou += train_metric['iou']
            train_class_acc += train_metric['class_accuracy']

        train_loss = train_loss / len(Train_loader)
        train_loss_all.append(train_loss)  # 储存训练损失

        train_acc = train_acc / len(Train_loader)
        train_miou = train_miou / len(Train_loader)
        train_iou = train_iou / len(Train_loader)
        train_class_acc = train_class_acc / len(Train_loader)

        metric_description = 'Train Acc: {:.5f}\n Train Miou: {:.5f}\n Train IoU:{:}\n Train_class_acc: {:}'.format(
                            train_acc,
                            train_miou,
                            train_iou,
                            train_class_acc
        )
        # 储存训练指标
        train_miou_all.append(train_miou)
        train_iou_all.append(train_iou)
        train_acc_all.append(train_acc)
        train_class_acc_all.append(train_class_acc)

        # 开始验证模式
        model.eval()
        eval_loss = 0
        eval_acc = 0
        eval_miou = 0
        eval_iou = 0
        eval_class_acc = 0

        with torch.no_grad():
            for i, sample in enumerate(Val_loader):
                X_val = Variable(sample['img'].type(torch.FloatTensor).to(device))
                Y_val = Variable(sample['label'].to(device))

                output = model(X_val)
                output = F.log_softmax(output, dim=1)

                loss = loss_fn(output, Y_val.long())
                eval_loss += loss.item()

                pre_val_label = output.max(dim=1)[1].data.cpu().numpy()
                pre_val_label = [i for i in pre_val_label]

                true_val_label = Y_val.data.cpu().numpy()
                true_val_label = [i for i in true_val_label]

                eval_metric = eval_semantic_segmentation(pre_val_label, true_val_label)
                eval_confusion = eval_metric['confusion']
                eval_acc += eval_metric['mean_class_accuracy']
                eval_miou += eval_metric['miou']
                eval_iou += eval_metric['iou']
                eval_class_acc += eval_metric['class_accuracy']


        eval_loss = eval_loss / len(Val_loader)
        eval_loss_all.append(eval_loss)  # 储存验证损失

        cur_time = datetime.now()  # 记录现在时间

        # 计算一次训练一次验证花费了多少时间
        h, remainder = divmod((cur_time - prec_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time = 3600 * h + 60 * m + s
        time_str = 'Time: {}s'.format(time)

        # 计算验证指标
        eval_miou = eval_miou / len(Val_loader)
        eval_iou = eval_iou / len(Val_loader)
        eval_class_acc = eval_class_acc / len(Val_loader)
        eval_acc = eval_acc / len(Val_loader)

        # 储存验证指标
        eval_miou_all.append(eval_miou)
        eval_iou_all.append(eval_iou)
        eval_class_acc_all.append(eval_class_acc)
        eval_acc_all.append(eval_acc)

        print(metric_description)
        if max(best) <= eval_miou:
            best.append(eval_miou)
            torch.save(model.state_dict(), 'xxx.pth')  # 保存验证效果最好的模型

        val_metric_description = 'Val Acc: {:.5f}\n Val Miou: {:.5f}\n Val IoU: {:}\n Val class acc: {:}'.format(
            eval_acc,
            eval_miou,
            eval_iou,
            eval_class_acc
        )
        print(val_metric_description)
        print(time_str)
    train_process = pd.DataFrame(data={"epoch": range(1, num_epochs + 1),
                                       "train_loss_all": train_loss_all, "eval_loss_all": eval_loss_all,
                                       'train_acc_all': train_acc_all, 'eval_acc_all': eval_acc_all,
                                       'train_miou_all': train_miou_all, 'eval_miou_all': eval_miou_all,
                                       'train_class_acc_all': train_class_acc_all, 'eval_class_acc_all': eval_class_acc_all})

    return model, train_process, train_confusion, eval_confusion


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 创建训练和验证数据迭代器
    train_file = "Train_data.pickle"
    val_file = "Val_data.pickle"
    Train_loader = bag_data(train_file, batch_size=1)  # 这里自己的电脑batch_size只能设置为4，因为显存不够，在云GPU上设置大一些
    Val_loader = bag_data(val_file, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用gpu运行程序，若无gpu使用cpu

    # 创建网络模型, 将模型放在gpu上
    model = rf50(num_classes=3)
    model = model.to(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

    # 创建损失函数
    loss_fn = nn.NLLLoss()
    loss_fn = loss_fn.to(device)

    # 优化器
    learning_rate = 2e-5
    optimal = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练并可视化
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    fcn_vgg16, train_process, train_confusion, eval_confusion = train_model(model, Train_loader, Val_loader, loss_fn, optimal)

    # 保存混淆矩阵并可视化
    train_process.to_csv('./train_process.csv')

    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    plot_confusion(train_confusion, train=True, type='damage')
    plot_confusion(eval_confusion, train=False, type='damage')

    plot_figure(1, title='Loss', x_label='epoch', y_label='loss', x=train_process.epoch,
                y1=train_process.train_loss_all, y2=train_process.eval_loss_all, ylegend1='Train Loss', ylegend2='Val Loss')
    plt.savefig('train_curve/Loss_Curve.png')
    plot_figure(2, title='Acc', x_label='epoch', y_label='acc', x=train_process.epoch,
                y1=train_process.train_acc_all, y2=train_process.eval_acc_all, ylegend1='Train Acc', ylegend2='Val Acc')
    plt.savefig('train_curve/Acc_Curve.png')
    plot_figure(3, title='Miou', x_label='epoch', y_label='Miou', x=train_process.epoch,
                y1=train_process.train_miou_all, y2=train_process.eval_miou_all, ylegend1='Train Miou', ylegend2='Val Miou')
    plt.savefig('train_curve/Miou_Curve.png')
    plt.show()




