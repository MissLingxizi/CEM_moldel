# CEM_model
This is a model for classifying colorectal cancer histopathological images using transfer learning and ensemble learning techniques.
The dataset used is the Endoscopy Biopsy Histopathological Hematoxylin and Eosin Image dataset (EBHI).https://doi.org/10.6084/m9.figshare.16999363.v1
Data pre-processing are Patches cropping (380×380 pixel,remove 80% empty),Stain normalization,Data augmentation and Balancing.
split: 6:2:2
seed=1337
epochs = 50
learning rate =0.0001
optimizer: Adam 
GPU: NVIDIA 3090
CNN: ResNet101, ResNet152, InceptionV3, Xception, DenseNet169, DenseNet201, EfficientNetB0, EfficientNetB1  and  EfficientNetV2M
