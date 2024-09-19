# Diabetic Retinopathy Prediction Web Application

This project focuses on building a web application for predicting diabetic retinopathy using advanced deep learning models. The application is designed to provide probability predictions, model activation maps, and advice based on the user's input and diabetic retinopathy predictions.

## Overview
### Dataset
The project utilizes the [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) dataset from a Kaggle competition to train and evaluate the models.

### Model Training
The models are built using the PyTorch framework, with custom dataloaders and pretrained architectures such as VGG19, ResNet50, and Swin Transformer. The training process is controlled through a YAML configuration file.

### Web Application
The web application is developed using Streamlit. Users can upload retina images to receive probability predictions and view model activation maps. Additionally, a large language model is integrated to provide advice based on the user's questions and the model's predictions.

### Features
* Custom Dataloaders: For efficient data handling.
* Pretrained Models: Utilizes VGG19, ResNet50, and Swin Transformer.
Configurable Training: Training process can be easily controlled via YAML files.
* Web Interface: Provides a user-friendly interface for predictions and visualizations.
* Large Language Model: Offers advice based on predictions and user queries.
 ![screencapture-localhost-8501-2024-08-17-12_23_04](https://github.com/user-attachments/assets/64428006-c738-4a80-b91f-04a348c5fc8a)

## Getting Started

### Model Training
To train the model, follow these steps:
```bash
# Setup
git clone https://github.com/thehienliu/BlindnessDetection.git
cd BlindnessDetection
pip install -r requirements.txt

# Train
python main.py --config="configs/config.yaml"
```
### Web Application
To run the web application:
```bash
# Prepare web materials
mkdir web_materials
cp "configs/config.yaml" "web_materials"  # Configuration for model information
cp "outputs/model.pt" "web_materials"  # Pretrained model weights

# Run the web app
streamlit run web.py
```
## Future Work
* Information Retrieval: Enhance the large language model to retrieve information from research articles related to diabetic retinopathy diagnosis and treatment, reducing the likelihood of hallucinations.
* Chatbot Integration: Develop a chatbot for more convenient communication and support.
