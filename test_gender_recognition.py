from tkinter import NE
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchviz import make_dot
from torchsummary import summary
from train_gender_recognition import Net

classes = ('男','女')
mbatch_size = 1

def loadtestdata():
    path = "data/test"
    testset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((32, 32)),
                                                    transforms.ToTensor()])
                                                )
    testloader = torch.utils.data.DataLoader(testset, batch_size=mbatch_size,
                                             shuffle=True, num_workers=2)
    return testloader

def reload_net():
    trainednet = torch.load("net_params.pth",map_location="cuda:0")
    return trainednet

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test():
    testloader = loadtestdata()
    net = reload_net()
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # import pdb;pdb.set_trace()
    # nrow是每行显示的图片数量，缺省值为8
    imshow(torchvision.utils.make_grid(images,nrow=1))
    # 打印前25个GT（test集里图片的标签）
    # print('真实值: ', " ".join('%5s' % classes[labels[j]] for j in range(mbatch_size)))
    outputs = net(Variable(images).cuda())
    MyConvNetVis = make_dot(outputs, params=dict(list(net.named_parameters()) + [('x',Variable(images))]) )
    MyConvNetVis.format = "png"
    # 指定文件生成的文件夹
    MyConvNetVis.directory = "data"
    # 生成文件
    MyConvNetVis.view() 
    _, predicted = torch.max(outputs.data, 1)
    # 预测值
    
    print('预测值: ', " ".join('%5s' % classes[predicted[j]] for j in range(mbatch_size)))

if __name__=="__main__":
    test()
