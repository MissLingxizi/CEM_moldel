# -*- coding: utf-8 -*-
"""
"""
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torchvision.transforms import transforms
import glob
import torch
import sys
import os
import torchvision.models as models
from sklearn.metrics import classification_report as CR
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import accuracy_score as ACC

import seaborn as sns
from efficientnet_pytorch import EfficientNet
from torchsummary import summary
import torchvision.models as models
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import datetime #
# print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


plt.rc('font', size=16)
plt.rcParams['font.family'] = ['SimSun', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'


#chose model
singleModelNames =['ResNet101', 'ResNet152', 'DenseNet121', 'DenseNet201', 'MobileNetV2','EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4','EfficientNetV2M']

for singleModelName in singleModelNames:

    #Path
    trainPath = r'D:\\train'
    testPath = r'D:\\val'
    #
    PICSize = (224,224)
    trainTransformer = transforms.Compose([transforms.Resize(PICSize),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])#R,G,B
    
    
    testTransformer = transforms.Compose([transforms.Resize(PICSize),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    
    trainData = ImageFolder(trainPath, transform=trainTransformer)#
    testData = ImageFolder(testPath, transform=testTransformer)#
    
    batch_size=128  #batchsize
    trainDataLoader  = DataLoader(trainData, batch_size, shuffle=True, num_workers=0, pin_memory=True)#load
    testDataLoader  = DataLoader(testData, batch_size, num_workers=0, pin_memory=True)#
    trainNum = len(trainData)#
    testNum = len(testData)#
     #
    def selectModel(singleModelName):
        if 'efficientnet' in singleModelName.lower():
            singleModelName = 'efficientnet-b' + singleModelName[-1]
            model = EfficientNet.from_pretrained(singleModelName)
            num_features = model._fc.in_features
            model._fc = torch.nn.Linear(num_features, len(trainData.classes))
            model.cuda()
            # print(summary(model, input_size=(3, 224, 224)))
            
        elif singleModelName == 'VGG16':
            model = models.vgg16(pretrained=True)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, len(trainData.classes))
            model.cuda()
            # print(summary(model, input_size=(3, 224, 224)))
            
            
        elif singleModelName == 'VGG19':
            model = models.vgg19(pretrained=True)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, len(trainData.classes))
            model.cuda()
            # print(summary(model, input_size=(3, 224, 224)))
            
        elif singleModelName == 'ResNet50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=len(trainData.classes)),
            nn.Softmax(dim=1)
            )
            model.cuda()
            # print(summary(model, input_size=(3, 224, 224)))
            
        elif singleModelName == 'ResNet101':
            model = models.resnet101(pretrained=True)
            model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=len(trainData.classes)),
            nn.Softmax(dim=1)
            )
            model.cuda()
            # print(summary(model, input_size=(3, 224, 224)))
            
        elif singleModelName == 'ResNet152':
            model = models.resnet152(pretrained=True)
            model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=len(trainData.classes)),
            nn.Softmax(dim=1)
            )
            model.cuda()
            # print(summary(model, input_size=(3, 224, 224)))
            
        elif singleModelName == 'DenseNet121':
            model = models.densenet121(pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, len(trainData.classes))
            model.cuda()
            # print(model)
        elif singleModelName == 'DenseNet169':
            model = models.densenet169(pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, len(trainData.classes))
            model.cuda()
            # print(model)
        elif singleModelName == 'DenseNet201':
            model = models.densenet201(pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, len(trainData.classes))
            model.cuda()
            # print(model)
            
        elif singleModelName == 'MobileNetV2':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(in_features=1280, out_features=len(trainData.classes)) #
            model.cuda()
            # print(summary(model, input_size=(3, 224, 224)))
            
        
        return model
        
    model = selectModel(singleModelName)
    
    #training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#
    epoch = 10 #
    
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)#
    criterion = nn.CrossEntropyLoss()
    
    train_steps = len(trainDataLoader)#设置迭代次数
    valid_steps = len(testDataLoader)
    
    trainLossList = []
    testLossList = []
    
    trainAccList = []
    testAccList = []

    Best_Acc = 0.0
    for epochs in range(epoch):
        running_loss = 0
        val_acc = 0
        train_acc = 0

        # train
        train_bar = tqdm(trainDataLoader, file=sys.stdout, colour='red')
        for step, data in enumerate(train_bar):
            model.train()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)#
            train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = f'train epoch[{epoch} / {epochs+1}], loss{loss.data:.3f}'

        trainLossList.append(running_loss)

        oValLoss = 0
        with torch.no_grad():
            valid_bar = tqdm(testDataLoader, file=sys.stdout, colour='red')
            for data in valid_bar:
                model.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)#CPU
                outputs = model(images)#
                val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
                
                ValLoss = criterion(outputs, labels)#
                oValLoss += ValLoss.item()


        testLossList.append(oValLoss)
        val_accuracy = val_acc / testNum
        train_accuracy = train_acc / trainNum
        
        trainAccList.append(train_accuracy)
        testAccList.append(val_accuracy)

        # save bestmodel
        if val_accuracy > Best_Acc:
            Best_Acc = val_accuracy
            state_dict = model.state_dict()
            os.makedirs('./log/%s/'%singleModelName, exist_ok=True)
            torch.save(state_dict ,'./log/%s/model.pth'%singleModelName)
            print('Best Acc:', Best_Acc)
