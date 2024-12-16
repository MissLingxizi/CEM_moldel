import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import random_split, DataLoader
from torchvision.models import efficientnet_b0,efficientnet_b1,efficientnet_b2
from torchvision.datasets import ImageFolder
from cm_tools import get_remarks# cm_show,
import timm  #  EfficientNetV2M
from datetime import datetime 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# data_dir = 'D:\\' #
data_dir_train= 'train' # 
data_dir_test='test'


Dataclass='XXX'
model_name = 'b1'#  B1-240 B0-224  B2-260


if model_name == 'b0':
     model = efficientnet_b0(pretrained=True)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # B1-240 B0-224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
    if model_name == 'b1':
        model = efficientnet_b1(pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(240),  # B1-240 B0-224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        model = efficientnet_b2(pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(260),  # B1-240 B0-224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# dataset = ImageFolder(data_dir, transform=transform)
#

# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
#

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


train_dataset= ImageFolder(data_dir_train, transform=transform)
test_dataset = ImageFolder(data_dir_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('EfficientNet'+model_name)
# print(data_dir)

num_classes = 5

for layer in model.classifier.modules():
    if isinstance(layer, nn.Linear):
        in_features = layer.in_features
        break


model.classifier[-1] = nn.Linear(in_features, num_classes)


for param in model.parameters():
    param.requires_grad = True
model.classifier.requires_grad = True


model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50  # 
for epoch in range(num_epochs):
    y_true_all = []
    y_pred_all = []
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)


        outputs = model(images)
        loss = criterion(outputs, labels)

 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = 100 * correct / total

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true_all.extend(labels.cpu().numpy())
            y_pred_all.extend(predicted.cpu().numpy())

        test_loss = test_loss / len(test_loader.dataset)
        test_accuracy = 100 * correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}%, '
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}%')
    cm = confusion_matrix(y_true_all, y_pred_all)
    # cm_show(cm, y_true_all)
    get_remarks(cm)
    # if epoch==num_epochs-1 :
    #     cm_show(cm, y_true_all)

    torch.save(model.state_dict(), r'pthSave\EfficientNet{}.pth'.format(model_name))

# torch.save(model.state_dict(), r'pthSave\EfficientNet{}.pth'.format(model_name))

