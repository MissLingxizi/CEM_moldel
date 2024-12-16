# CEM_model
This is a model for classifying colorectal cancer histopathological images using transfer learning and ensemble learning techniques.
The dataset used is the Endoscopy Biopsy Histopathological Hematoxylin and Eosin Image dataset.https://doi.org/10.6084/m9.figshare.16999363.v1
Data pre-processing are Patches cropping (380×380 pixel,remove 80% empty).
split: 6:2:2
seed=1337
epochs = 50
learning rate =0.0001
optimizer: Adam 
GPU
CNNs: ResNet101, ResNet152, In
