# Assessing Water Quality Environmental Grades Using Hyperspectral Image and Deep Learning Model: A Case Study in Jiangsu
## Table of Contents
- [1. About The Project](#Project) 
-  [2. Usage](#directory-structure) 
- [3. Directory Structure](#usage) 
- [4. Structure Declaration](#Structure-declaration)
- [5. Getting Started](#Getting-Started)
- [6. Contact](#contact)
## 1. About The Project
This project introduces HybridNet, a deep learning model specifically designed for classifying water quality grades using hyperspectral image(HSI) data. HybridNet combines the capabilities of Capsule Networks with the Multi-Headed Dynamic Interactive Attention (MDIA) mechanism to effectively capture both spatial and spectral dependencies within HSI data. This hybrid approach enables the model to identify complex, hierarchical relationships in the data, while dynamically directing attention to the most relevant spectral bands and spatial regions, resulting in improved classification accuracy.

## 2. Usage
this repo provides several features:
- You can preprocess your data and perform segmentation of HSI with sliding window techniques in the `data_process` folder.
- Some modules used by the models are also in the `layers` folder, including existing modules (capsule_layer) and improved modules (MDIA).
- You can view some models used, including public models like CASNet, ECA-Net, GLCSA-Net, etc., as well as the HybridNet model proposed in this project. All of these are placed in the `models` folder.
- Once the environment is set up, you can proceed with model training, testing, and validation, with some visual information provided.

If you have any questions about the models, feel free to engage in technical discussions. Please note that there are still many possible improvements between models and modules.

## 3. Directory Structure
The repository is structured as follows:

watequality-assessing-main/

├── data_process/

│  &nbsp; &nbsp;├── apply-pca.py

│  &nbsp; &nbsp;├── slide cubes.py

│  &nbsp; &nbsp;└── preprocess.py

├── datasets/

│&nbsp; &nbsp; ├── Crab pond

│ &nbsp; &nbsp;├── Fish pond

│ &nbsp; &nbsp;├── Jingsi lake

│ &nbsp; &nbsp;├── Xinghai lake

│ &nbsp; &nbsp;├── Xishuang lake

│ &nbsp; &nbsp;└── finally_process.npy

├── exp/

│&nbsp; &nbsp; ├── train.py

│ &nbsp; &nbsp;├── test.py

│ &nbsp; &nbsp;└── validate.py

├── layers/

│ &nbsp; &nbsp;├── capsule_conv_layer.py

│ &nbsp; &nbsp;├── capsule_layer.py

│&nbsp; &nbsp; └── capsule_network.py


├── models/

│ &nbsp; &nbsp;├── CapsNet.py

│ &nbsp; &nbsp;├── CASNet.py

│ &nbsp; &nbsp;├── ECA-Net.py

│ &nbsp; &nbsp;├── GLCSA-Net.py

│&nbsp; &nbsp; ├── HybridNet.py

│&nbsp; &nbsp; ├── ResU-Net.py

│ &nbsp; &nbsp;├── RSCNet.py

│ &nbsp; &nbsp;├── SSAU-Net.py

│ &nbsp; &nbsp;└── SWUNet.py

├── scripts/

├── utils/

│ &nbsp; &nbsp;├── metrics.py

│ &nbsp; &nbsp;├── metrcs.py

│ &nbsp; &nbsp;└── visualization.py

└── README.md

## 4. Structure Declaration
**data_process/**
The folders contain files for reading data:
- **apply-pca.py**: Applying PCA to reduce the dimensionality of HSI data.
- **slide cubes.py**: Using sliding window technology to segment HSI data.
- **preprocess.py**: Other data preprocessing operations.

**datasets/**
This folder contains complete HSI for five water quality regions, which have undergone reflectance calibration, atmospheric correction, and geometric correction (raw files).Additionally, we have provided the data segmented using the sliding window method for training and testing (npy file). 

**exp/**
This folder is used for executing model training, testing, and validation.
- **train.py**: Manages model device selection (CPU or GPU) and initiates the training process.
- **test.py**: Handles model testing and validation, including the calculation of relevant performance metrics.
- **validate.py**: Performs model validation.

**layers/**
This folder contains the network modules required by various models, stored modularly.

**models/**
This folder contains integrated models of various modules, including comparison models and HybridNet.

**utils/**
This folder contains functions for calculating evaluation metrics and generating visualizations.

## 5. Getting Started

1. Install Python 3.10+, PyTorch 1.9.0, and Shell environment.
2. Prepare the HSI data. Public HIS datasets can be selected.
3. Train the model. Adjust hyperparameters like learning rate, batch size, and the number of epochs as needed.

## 6. Contact

If you have any questions, please contact [zhaohui990401@163.com](mailto:zhaohui990401@163.com).
