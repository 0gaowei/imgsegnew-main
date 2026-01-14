import time
import torch

class Common:
    '''
    通用配置
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 设备配置
    imageSize = (224, 224)  # 模型修正的图片大小


class Train:
    '''
    训练相关配置
    '''
    num_workers = 0  # 对于Windows用户，这里应设置为0，否则会出现多线程错误
    baseDir = "./"  # 基本路径，日志、模型存放在该文件夹下
    modelDir = "{}".format(baseDir) + "model/"  # 模型存放位置
    dataDir = "{}".format(baseDir) + "data/"  # 训练数据存放位置
    testDataDir = "{}".format(baseDir) + "test_data/"   # 测试数据存放位置

    def __init__(self, baseDir):
        self.baseDir = './{}'.format(baseDir)
        self.modelDir = "{}".format(self.baseDir) + "model/"
        self.dataDir = "{}".format(self.baseDir) + "data/"
