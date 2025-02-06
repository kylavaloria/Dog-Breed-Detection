# Dog Breed Detection

## Overview
This project aims to classify dog breeds using deep learning and computer vision techniques. We trained the model using a custom dataset of dog images, leveraging the power of transfer learning with the InceptionV3 model. The model is capable of classifying dog breeds with an accuracy of around 80%, which was achieved through data preprocessing, training, and fine-tuning.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Fine-tuning](#fine-tuning)
- [Real-time Detection](#real-time-detection)
- [Contributing](#contributing)
- [License](#license)

## Installation
To get started, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/kylavaloria/Dog-Breed-Detection.git
cd Dog-Breed-Detection
```

Set up a virtual environment (optional but recommended):
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage
You can run the model for dog breed classification with the Streamlit application:
```bash
streamlit run app.py
```

## Dataset
The dataset used for training the model is the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html). This dataset contains images of 120 dog breeds, with a total of 20,580 images. The data is organized into three folders:

- `train`: Contains training images with breed labels.
- `validation`: Used for validating the model during training.
- `test`: For testing the model's performance after training.

## Image Processing
The images are resized to 224x224 pixels for compatibility with the InceptionV3 model, and augmentation techniques such as rotation, shifting, zooming, and flipping were applied to prevent overfitting.

## Model
The model is built using the InceptionV3 architecture, pre-trained on the ImageNet dataset. It is fine-tuned to perform dog breed classification. The following layers were added on top of the pre-trained model:
- Global Average Pooling
- Dropout for regularization
- Dense layers with ReLU activation
- Output layer with Softmax activation for multi-class classification

## Model Summary:
- Pre-trained InceptionV3 base
- Fine-tuned for dog breed classification
- Achieved an accuracy of ~80%
