from torch import nn
import torchvision.models as models



def model(class_num):
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = net.fc.in_features
    for param in net.parameters():
        param.requires_grad = False  # False：冻结模型的参数，
        # 也就是采用该模型已经训练好的原始参数。
        # 只需要训练我们自己定义的Linear层
    # net.fc = nn.Sequential(nn.Linear(num_ftrs, class_num),
    #                             nn.LogSoftmax(dim=1))
    net.fc = nn.Linear(num_ftrs, class_num)
    return net


#model = net
# class WeatherModel(nn.Module):
#     def __init__(self, net):
#         super(WeatherModel, self).__init__()
#         # resnet50
#         self.net = net
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.1)
#         self.fc = nn.Linear(1000, 2)
#         self.output = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.net(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc(x)
#         x = self.output(x)
#         return x


#model = WeatherModel(net)
