# 训练部分
import os
import time
import uuid

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from config import Common, Train
from model import model as Model
from torch import optim
# from data_loader import trainLoader, valLoader, splitData, WeatherDataSet
from data_loader import splitData, WeatherDataSet
# 1. 获取模型
#model = Model
#model.to(Common.device)
# 2. 定义损失函数
#criterion = nn.CrossEntropyLoss()
# 3. 定义优化器
#optimizer = optim.Adam(model.parameters(), lr=0.001)
# 4. 创建writer
# writer = SummaryWriter(log_dir=Train.logDir, flush_secs=500)


def train(epoch, trainLoader, model, criterion, optimizer):
    '''
    训练函数
    '''
    # 1. 获取dataLoader

    loader = trainLoader
    # 2. 调整为训练状态
    model.train()
    print()
    print('========== Train Epoch:{} Start =========='.format(epoch))
    epochLoss = 0  # 每个epoch的损失
    epochAcc = 0  # 每个epoch的准确率
    correctNum = 0  # 正确预测的数量
    for data, label in loader:
        data, label = data.to(Common.device), label.to(Common.device)  # 加载到对应设备
        batchAcc = 0  # 单批次正确率
        batchCorrectNum = 0  # 单批次正确个数
        optimizer.zero_grad()  # 清空梯度
        output = model(data)  # 获取模型输出
        loss = criterion(output, label)  # 计算损失
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 更新参数
        epochLoss += loss.item() * data.size(0)  # 计算损失之和
        # 计算正确预测的个数
        labels = torch.argmax(label, dim=1)
        outputs = torch.argmax(output, dim=1)
        for i in range(0, len(labels)):
            if labels[i] == outputs[i]:
                correctNum += 1
                batchCorrectNum += 1
        batchAcc = batchCorrectNum / data.size(0)
        # print("Epoch:{}\t TrainBatchAcc:{}".format(epoch, batchAcc))

    epochLoss = epochLoss / len(trainLoader.dataset)  # 平均损失
    epochAcc = correctNum / len(trainLoader.dataset)  # 正确率
    print("Epoch:{}\t Loss:{} \t Acc:{}".format(epoch, epochLoss, epochAcc))
    return epochAcc


def val(epoch, valLoader,model ,criterion, Labels):
    '''
    验证函数
    :param epoch: 轮次
    :return:
    '''
    # 1. 获取dataLoader
    loader = valLoader
    # 2. 初始化损失、准确率列表
    valLoss = []
    valAcc = []
    # 3. 调整为验证状态
    model.eval()
    print()
    print('========== Val Epoch:{} Start =========='.format(epoch))
    epochLoss = 0  # 每个epoch的损失
    epochAcc = 0  # 每个epoch的准确率
    correctNum = 0  # 正确预测的数量
    classNum = len(Labels)
    single_correct_num = list(0. for i in range(classNum))  # 各个组件类型正确个数列表
    single_total = list(0. for i in range(classNum))    # 各个组件类型的总个数列表
    single_acc = list(0. for i in range(classNum))      # 各个组件类型的准确率
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(Common.device), label.to(Common.device)  # 加载到对应设备
            batchAcc = 0  # 单批次正确率
            batchCorrectNum = 0  # 单批次正确个数
            output = model(data)  # 获取模型输出
            loss = criterion(output, label)  # 计算损失
            epochLoss += loss.item() * data.size(0)  # 计算损失之和
            # 计算正确预测的个数
            labels = torch.argmax(label, dim=1)
            outputs = torch.argmax(output, dim=1)
            for i in range(0, len(labels)):
                if labels[i] == outputs[i]:
                    correctNum += 1
                    single_correct_num[labels[i]] += 1      # label[i] 这一组件预测正确个数加1
                    batchCorrectNum += 1
                single_total[labels[i]] += 1               # labels[i] 这一组件总个数加1
            batchAcc = batchCorrectNum / data.size(0)
            # print("Epoch:{}\t ValBatchAcc:{}".format(epoch, batchAcc))
        epochLoss = epochLoss / len(valLoader.dataset)  # 平均损失
        epochAcc = correctNum / len(valLoader.dataset)  # 正确率
        print("Epoch:{}\t Loss:{} \t Acc:{}".format(epoch, epochLoss, epochAcc))
        # print(f"single_correct_num:{single_correct_num},single_total:{single_total}")
        for i in range(len(Labels)):
            try:
                single_acc[i] = single_correct_num[i]/single_total[i]
            except Exception:
                pass
            # print(f"{Labels[i]}的准确率：{single_acc[i]}")
    return epochAcc, single_acc, epochLoss

if __name__ == '__main__':
    maxAcc = 0.75
    Loss = 1
    data_path = Train.dataDir + '/'
    labels = os.listdir(data_path)
    classNum = len(labels)
    print(classNum)
    train_task_id = "Train-imgSeg-" + str(uuid.uuid4())
    # 1. 获取模型
    model = Model(classNum)
    model.to(Common.device)
    # 2. 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 3. 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 4. 创建writer
    # writer = SummaryWriter(log_dir=Train.logDir+str(uuid.uuid4()), flush_secs=500)
    # 获取数据
    # 1. 分割数据集
    train_dataset, validation_dataset = splitData(WeatherDataSet(Train.dataDir))
    print(f"数据集总量：{WeatherDataSet(Train.dataDir).__len__()}")
    # 2. 训练数据集加载器
    trainLoader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=Train.num_workers)
    # 3. 验证集数据加载器
    valLoader = DataLoader(validation_dataset, batch_size=32, shuffle=False,
                           num_workers=Train.num_workers)
    for epoch in range(1, 61):  # 训练60轮
        trainAcc = train(epoch, trainLoader, model, criterion, optimizer)
        valAcc, valSingleAcc, valLoss = val(epoch, valLoader, model, criterion, labels)
        if valAcc >= maxAcc and valLoss <= Loss:
            maxAcc = valAcc
            Loss = valLoss
            # 保存最大模型
            torch.save({
                'model': model,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': valAcc,
                'loss': valLoss,
                'labels': labels,
                'singleAcc': valSingleAcc
            }
            , Train.modelDir +train_task_id + ".pth")
