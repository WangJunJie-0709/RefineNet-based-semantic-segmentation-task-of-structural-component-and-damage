import torch
from torch.utils.data import Dataset
from evalution import eval_semantic_segmentation
from train import bag_data
from load_data import MytestDataset
from load_data import read_image_path
from model import FCN_VGG16, ResNet50, rf50
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from matplotlib import image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dir = 'pre_picture'  # 保存预测图像的文件路径

Test_loader = bag_data("Test_data.pickle", batch_size=1)

model = rf50(3).to(device)  # 构件识别类别为8，损伤识别为3
# model = nn.DataParallel(model)
model.load_state_dict(torch.load('xxx.pth'))  # 加载训练好的模型权重

model.eval()

for i, sample in enumerate(Test_loader):
    X_test = Variable(sample['img'].type(torch.FloatTensor).to(device))
    output = model(X_test)
    output = F.log_softmax(output, dim=1)

    pre_label = output.max(dim=1)[1].squeeze().data.cpu().numpy()
    if not os.path.exists(r'test_picture\测试样本{}'.format(i)):  # 判断是否有该文件夹，没有则创建一个新文件夹
        os.makedirs(r'test_picture\测试样本{}'.format(i))

    toPIL = transforms.ToPILImage()
    X = X_test.squeeze().cpu()
    pic = toPIL(X)  # 将Tensor类型转换为PIL格式
    pic.save(r'test_picture\测试样本{}\true.jpg'.format(i))
    image.imsave(r'test_picture\测试样本{}\pre.png'.format(i), pre_label)





