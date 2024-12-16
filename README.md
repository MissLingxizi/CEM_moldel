# CEM_model
This is a model for classifying colorectal cancer histopathological images using transfer learning and ensemble learning techniques.
The dataset used is the Endoscopy Biopsy Histopathological Hematoxylin and Eosin Image dataset

The steps are as follows:
1. Download the data sets: https://figshare.com/articles/dataset/EBH-HE-IDS/16999363/1
2. To process the images, crop the images using getpatch.py (380*380).
3. EdgeDetection.py is used for image edge detection.
4. remove blank images using remove.py (threshold =0.2).
5. Use stain.py to stain the images with not good color.
6. Use predata.py to augment and balance the data set. According to the number of images with different magnification, set the amplification factor to make the dataset balanced.
7. Use TL_xx.py to perform transfer learning on the data set and test all CNNs. (splitting 6:2:2)
8. Use the PreTraining.py to pretrain the models. Put the specified CNNs into Models_config.
9. Use stacking.py to ensemble learning of stack . Selected_models,Models_config,trained_models put the same CNNs.
10. Using voteweightstack.py  for Majority voting and unweighted averaging for ensemble learning. Singlemodelnames load CNNs. Ensemle_type load ensemble strategies.
11. Other parameter settings:
seed=1337
epochs = 50
learning rate =0.0001
optimizer: Adam
average='macro' 
GPU

Tip: The final classification performance is better if you choose to train separately than if you train together.
I have provided some sample data sets and trained models in Baidu web disk, which can be used directly.
