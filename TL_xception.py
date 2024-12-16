import torch
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from pretrainedmodels import xception
import torch.nn as nn
import torch.optim as optim
from cm_tools import get_remarks#cm_show,

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((299, 299)),  # 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])


data_dir = 'XXXX'  
dataset = ImageFolder(root=data_dir, transform=transform)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


model = xception(pretrained='imagenet')
modelname='Xception'
print(modelname)
print(data_dir)

num_classes = len(dataset.classes)  
model.last_linear = nn.Linear(in_features=2048, out_features=num_classes, bias=True)


model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


num_epochs = 50
for epoch in range(num_epochs):
    y_true_all = []
    y_pred_all = []
    model.train()  
    train_loss = 0.0
    correct_t = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)


        outputs = model(images)
        loss = criterion(outputs, labels)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += labels.size(0)
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)

        correct_t += (predicted == labels).sum().item()
        # train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = 100 * correct_t / total

   
    model.eval()  
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            # correct += predicted.eq(labels).sum
            correct += predicted.eq(labels).sum().item()
            y_true_all.extend(labels.cpu().numpy())
            y_pred_all.extend(predicted.cpu().numpy())

    val_loss = val_loss / len(test_loader.dataset)
    val_accuracy = 100 * correct / total

    print( f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%,Val Accuracy: {val_accuracy:.2f}%')
    cm = confusion_matrix(y_true_all, y_pred_all)
    #cm_show(cm, y_true_all)
    get_remarks(cm)


    # if epoch==num_epochs-1 :
    #     cm_show(cm, y_true_all)


torch.save(model.state_dict(), r'pthSave\{}.pth'.format(modelname))