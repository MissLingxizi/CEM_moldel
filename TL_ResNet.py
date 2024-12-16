import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


data_dir = 'xxxx' 

model = models.resnet152(pretrained=True)
modelname='Resnet152-'

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)

model = model.to(device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(data_dir, transform=transform)


train_size = int(0.6 * len(dataset))
test_size = int(0.2 * len(dataset))
val_size= len(dataset) - train_size-test_size

train_dataset, test_dataset,val_dataset= random_split(dataset, [train_size, test_size,val_size])


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


print(modelname,data_dir)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 50
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(num_epochs):
    print('-' * 10)
    print(f'Epoch {epoch + 1}/{num_epochs}')


    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  
            loader = train_loader
        else:
            model.eval()  
            loader = val_loader

        running_loss = 0.0
        running_corrects = 0


        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()


            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_corrects.double() / len(loader.dataset)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())


model.load_state_dict(best_model_wts)


model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())


accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')



print(f' Accuracy: {accuracy:.4f},'
      f' Precision: {precision:.4f}, '
      f' Recall: {recall:.4f}, '
          f' F1 Score: {f1:.4f}'
      )
torch.save(model.state_dict(), r'pthSave\{}-Resnet.pth'.format(modelname))

