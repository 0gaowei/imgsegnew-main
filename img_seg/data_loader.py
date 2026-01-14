# 自定义数据加载器
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import Common
from config import Train
import os
from PIL import Image
import torch.utils.data as Data
import numpy

# 定义数据处理transform
transform = transforms.Compose([
    transforms.Resize(Common.imageSize),
    transforms.ToTensor()
])


def loadDataFromDir(dataDir):
    '''
    从文件夹中获取数据
    '''
    path = dataDir
    images = []
    labels = []
    labels_text = []
    #读取数据集所有类别，写入列表
    for d in os.listdir(path):
        type_path = os.path.join(path, d)
        filename = os.path.basename(type_path)
        labels_text.append(filename)
    # 1. 获取根文件夹下所有分类文件夹
    for d in os.listdir(path):
        # 2. 获取某一类型下所有的图片名称
        type_path = os.path.join(path, d)
        for imagePath in os.listdir(type_path):
            # 3. 读取文件
            try:
                image = Image.open(type_path + "/" + imagePath).convert('RGB')
            except Exception as e:
                print(type_path+"/"+imagePath+"读取失败")
                continue
            # print("加载数据" + str(len(images)) + "条")

            # 4. 添加到图片列表中
            images.append(transform(image))
            # 5. 构造label
            categoryIndex = labels_text.index(d)  # 获取分类下标
            length = len(labels_text)
            label = [0] * length  # 初始化label
            label[categoryIndex] = 1  # 根据下标确定目标值
            label = torch.tensor(label, dtype=torch.float)  # 转为tensor张量
            # 6. 添加到目标值列表
            labels.append(label)
            # 7. 关闭资源
            image.close()
    # 返回图片列表和目标值列表
    print(f"共加载数据{len(images)}条")
    return images, labels


class WeatherDataSet(torch.utils.data.Dataset):
    '''
    自定义DataSet
    '''

    def __init__(self, dataDir):
        super().__init__()
        '''
        初始化DataSet
        :param transform: 自定义转换器
        '''
        images, labels = loadDataFromDir(dataDir)  # 在文件夹中加载图片
        self.images = images
        self.labels = labels

    def __len__(self):
        '''
        返回数据总长度
        :return:
        '''
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


def splitData(dataset):
    '''
    分割数据集
    :param dataset:
    :return:
    '''
    # # 求解一下数据的总量
    # total_length = len(dataset)
    #
    # # 确认一下将80%的数据作为训练集, 剩下的20%的数据作为测试集
    # train_length = int(total_length * 0.8)
    # validation_length = total_length - train_length
    #
    # # 利用Data.random_split()直接切分数据集, 按照80%, 20%的比例进行切分
    # train_dataset, validation_dataset = Data.random_split(dataset=dataset, lengths=[train_length, validation_length])
    # return train_dataset, validation_dataset
    # 直接返回整个数据集作为训练集和测试集
    return dataset, dataset

# datasetid = 'resnet_dataset_1'
#
# # 1. 分割数据集
# train_dataset, validation_dataset = splitData(WeatherDataSet(datasetid))
# # 2. 训练数据集加载器
# trainLoader = DataLoader(train_dataset, batch_size=Train.batch_size, shuffle=True, num_workers=Train.num_workers)
# # 3. 验证集数据加载器
# valLoader = DataLoader(validation_dataset, batch_size=Train.batch_size, shuffle=False,
#                        num_workers=Train.num_workers)