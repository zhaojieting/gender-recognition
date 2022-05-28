import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def loadtraindata():
    # 路径
    path = "data/train"
    trainset = torchvision.datasets.ImageFolder(path,transform=transforms.Compose([
                                                    # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.Resize((32, 32)),
                                                    transforms.CenterCrop(32),
                                                    transforms.ToTensor()])
                                                )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=True, num_workers=2)
    return trainloader

# 定义网络，继承torch.nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)     
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # 2个输出
        self.fc3 = nn.Linear(84, 2)

    # 前向传播
    def forward(self, x):        
        # F就是torch.nn.functional
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #在CNN中卷积或者池化之后需要连接全连接层
        #所以需要把多维度的tensor展平成一维
        x = x.view(x.size(0), -1)
        # 从卷基层到全连接层的维度转换
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def trainandsave():
    trainloader = loadtraindata()
    # 神经网络结构
    net = Net()
    # 优化器，学习率为0.001
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 训练部分，训练的数据量为5个epoch，每个epoch为一个循环
    for epoch in range(5):
        # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        # 定义一个变量方便我们对loss进行输出
        running_loss = 0.0
        # 这里我们遇到了第一步中出现的trailoader，代码传入数据
        for i, data in enumerate(trainloader, 0):
            # enumerate是python的内置函数，既获得索引也获得数据
            # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
            import pdb;pdb.set_trace()
            inputs, labels = data

            # 转换数据格式用Variable
            inputs, labels = Variable(inputs), Variable(labels)
            # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
            optimizer.zero_grad()

            # forward + backward + optimize，把数据输进CNN网络net
            outputs = net(inputs)
            # 计算损失值
            loss = criterion(outputs, labels)
            # loss反向传播
            loss.backward()
            # 反向传播后参数更新
            optimizer.step()
            # loss累加
            running_loss += loss.item()
            if i % 200 == 199:
                # 然后再除以200，就得到这两百次的平均损失值
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                 # 这一个200次结束后，就把running_loss归零，下一个200次继续使用
                running_loss = 0.0

    print('Finished Training')
    # 保存神经网络
    # netScript = torch.jit.script(net)
    # 保存整个神经网络的结构和模型参数
    torch.save(net,'net_params.pth')
    # 只保存神经网络的模型参数
    #torch.jit.save(net.state_dict(), 'net_params.pt')
if __name__=="__main__":
    trainandsave()
