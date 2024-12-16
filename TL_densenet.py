import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torchvision.models import densenet121,densenet169,densenet201
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from cm_tools import get_remarks#cm_show,



dataset_root = 'xx'



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = ImageFolder(root=dataset_root, transform=transform)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

model_name = '201' #

if model_name =='121':
    model = densenet121(pretrained=True)
else :
    if model_name =='169':
        model = densenet169(pretrained=True)
    else:
        model = densenet201(pretrained=True)

print('DenseNet'+model_name)
print(dataset_root)
num_classes = len(dataset.classes)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier.in_features, 4096),
    torch.nn.ReLU(True),
    torch.nn.Dropout(),
    torch.nn.Linear(4096, num_classes),
)

#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 50
for epoch in range(num_epochs):
    y_true_all = []
    y_pred_all = []
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    model.eval()
    with torch.no_grad():
        correct_val = 0
        total_val = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            y_true_all.extend(labels.cpu().numpy())
            y_pred_all.extend(predicted.cpu().numpy())

        val_accuracy = 100 * correct_val / total_val

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader.dataset):.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%')

    cm = confusion_matrix(y_true_all, y_pred_all)
    get_remarks(cm)
    # if epoch==num_epochs-1 :
    #     cm_show(cm, y_true_all)

    # torch.save(model.state_dict(), r'pthSave\Move90%augmented_DenseNet{}.pth'.format(model_name))