
import glob
import numpy as np
from sklearn.metrics import classification_report as CR
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from pretrainedmodels import xception
from efficientnet_pytorch import EfficientNet
from EM1_utils import majority_voting, unweighted_averaging, weighted_averaging

def selectModel(singleModelName):
    trainPath = r'/ColhisCode/40_zg_balance-split3/train'
    trainData = ImageFolder(trainPath)
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
        model.classifier[1] = nn.Linear(in_features=1280, out_features=len(trainData.classes))
        model.cuda()
        # print(summary(model, input_size=(3, 224, 224)))
    elif singleModelName == 'xception':
        model = xception(pretrained='imagenet')
        model.last_linear = nn.Linear(in_features=2048, out_features=len(trainData.classes), bias=True)

    return model


singleModelNames = ['DenseNet201', 'MobileNetV2', 'EfficientNetB1']
# singleModelNames = ['EfficientNetB0', 'DenseNet201', 'EfficientNetB1']
ensemble_type = ['majority voting', 'unweighted averaging']


if __name__ == "__main__":


    trainPath = r'/train'
    testPath = r'/val'
    # ==============================================================================

    PICSize = (224, 224)
    trainTransformer = transforms.Compose([transforms.Resize(PICSize),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                (0.2023, 0.1994, 0.2010))])

    testTransformer = transforms.Compose([transforms.Resize(PICSize),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                               (0.2023, 0.1994, 0.2010))])  #

    trainData = ImageFolder(trainPath, transform=trainTransformer)
    testData = ImageFolder(testPath, transform=testTransformer)

    batch_size = 64 #
    trainDataLoader = DataLoader(trainData, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    testDataLoader = DataLoader(testData, batch_size, num_workers=0, pin_memory=True)
    trainNum = len(trainData)
    testNum = len(testData)

    model_list = []
    for modelname in singleModelNames:
        model = selectModel(modelname)
        # load weight
        state_dict = torch.load('./log/%s/model.pkl'%modelname)
        model.load_state_dict(state_dict)
        print('Model %s Successly Load Weight!'%modelname)
        model_list.append(model)

    picFilePath = glob.glob(r'{}/*/*'.format(testPath))
    labelMap = trainData.classes
    print(labelMap)
    for ensem_type in ensemble_type:
        if ensem_type == 'majority voting':
            # vote
            imgs, trueLabels, preLabels, preScore = majority_voting(model_list, picFilePath, labelMap)
        elif ensem_type == 'unweighted averaging':
            imgs, trueLabels, preLabels, preScore = unweighted_averaging(model_list, picFilePath, labelMap)



        print('==={}_{}_Result'.format(singleModelNames,ensem_type))
         # preScore = np.array(preScore)
        print('\n{}_Result'.format(singleModelNames))
        print('ACC:{}'.format(ACC(trueLabels, preLabels)))

        trueLabels_ = [trainData.classes[I] for I in trueLabels]
        preLabels_ = [trainData.classes[I] for I in preLabels]
        print('\n', CR(trueLabels_, preLabels_))

        CMValues = CM(trueLabels_, preLabels_)


        TP = CMValues.diagonal()
        FP = np.sum(CMValues, axis=0) - TP
        FN = np.sum(CMValues, axis=1) - TP
        TN = np.sum(CMValues) - TP - FP - FN

        for I in range(len(trainData.classes)):
            print('The{}:TP {}; FP {}; FN {}; TN {}.'.format(trainData.classes[I], TP[I], FP[I], FN[I], TN[I]))

        AUC = roc_auc_score(preLabels, trueLabels)
        print(AUC)
        print('===========================================')

        # MCC
        MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
        print('MCC:', MCC)
        print('===========================================')

        # DOR
        DOR = (TP * TN) / (FP * FN)
        print('DOR:', DOR)
        print('===========================================')