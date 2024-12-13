
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torchmetrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from pretrainedmodels import xception


data_path=''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selected_models = ['efficientnet_b1','efficientnet_b0','densenet201','xception','efficientnet_v2_m']

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.images.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
        
def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, device, num_epochs=30):
    best_val_acc = 0.0
    best_model_path = f'best_{model_name}.pth'
    metrics = {}
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        # 计算指标
        # tn, fp, fn, tp = confusion_matrix(val_labels, val_preds).ravel()
        cm = confusion_matrix(val_labels, val_preds)
        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # f1 = 2 * (precision * recall) / (precision + recall)
        # cls_report = classification_report(val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels,
            val_preds,
            average='macro'
        )
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        # # print(cls_report)
        # # print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')
        # print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        # print(cm)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            metrics = {
                'accuracy': val_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
                # 'confusion_matrix': (tn, fp, fn, tp)
            }
    print(best_val_acc)
    return best_model_path, metrics

# stacking
class StackingEnsemble(nn.Module):
    def __init__(self, base_models, num_classes):
        super(StackingEnsemble, self).__init__()
        self.base_models = nn.ModuleList(base_models)
        input_size = len(base_models) * num_classes
        self.meta_classifier = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        base_outputs = []
        for model in self.base_models:
            with torch.no_grad():
                output = model(x)
                base_outputs.append(output)
        meta_features = torch.cat(base_outputs, dim=1)
        return self.meta_classifier(meta_features)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageDataset(data_path, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
num_classes = len(dataset.classes)

def initialize_model(model_name, num_classes):
    if model_name == 'densenet201':
        model = models.densenet201(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.aux_logits = False
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == 'efficientnet_v2_m':
        model = models.efficientnet_v2_m(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == 'xception':
        model = xception(pretrained='imagenet')
        model.last_linear = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    return model

models_config = {
    'efficientnet_b1': initialize_model('efficientnet_b1', num_classes),
    'efficientnet_v2_m': initialize_model('efficientnet_v2_m', num_classes),
    'xception': initialize_model('xception', num_classes),
    'densenet201': initialize_model('densenet201', num_classes),
    'efficientnet_b0': initialize_model('efficientnet_b0', num_classes),
}

trained_models = {
    'densenet201': 'best_densenet201.pth',
    'xception': 'best_xception.pth',
    'efficientnet_b0': 'best_efficientnet_b0.pth',
    'efficientnet_v2_m': 'best_efficientnet_v2_m.pth',
    'efficientnet_b1': 'best_efficientnet_b1.pth'
}

# stacking
base_models = []
for model_name in selected_models:
    model = models_config[model_name]
    model.load_state_dict(torch.load(trained_models[model_name]))
    model.eval()
    base_models.append(model)

ensemble = StackingEnsemble(base_models, num_classes=len(dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ensemble.meta_classifier.parameters(), lr=0.001)

print("\nTraining Stacking Ensemble...")
ensemble_path, ensemble_metrics = train_model(
    ensemble, 'stacking_ensemble', train_loader, val_loader,
    criterion, optimizer, device
)

# ROC curve
ensemble.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = torch.softmax(ensemble(inputs), dim=1)
        all_probs.extend(outputs.cpu().numpy())
        all_labels.extend(labels.numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

plt.figure(figsize=(10, 8))
for i in range(len(dataset.classes)):
    fpr, tpr, _ = roc_curve(
        (all_labels == i).astype(int),
        all_probs[:, i]
    )
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Stacking Ensemble')
plt.legend()
plt.savefig('roc_curves.png')

print("\nFinal Ensemble Results:")
print(f"Accuracy: {ensemble_metrics['accuracy']:.2f}%")
print(f"Precision: {ensemble_metrics['precision']:.4f}")
print(f"Recall: {ensemble_metrics['recall']:.4f}")
print(f"F1 Score: {ensemble_metrics['f1']:.4f}")
cm = ensemble_metrics['confusion_matrix']
print(f"Confusion Matrix:")
print(cm)