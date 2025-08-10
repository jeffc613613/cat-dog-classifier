# Cat vs Dog Classifier

A **Convolutional Neural Network (CNN)** image classifier built with **TensorFlow** and **TensorFlow Datasets (TFDS)**.  
Includes a **Streamlit** web interface where you can upload an image and instantly find out whether it’s a **cat** or a **dog**.

## Live Demo

<img width="1918" height="950" alt="Streamlit Interface Demo" src="https://github.com/user-attachments/assets/f6c8b970-a6e0-4c6a-bde0-e9e6d97dc009" />

*Web interface showing cat classification with 89.61% confidence*
---

## Features

- Uses the official **Cats vs Dogs** dataset from **TensorFlow Datasets**
- CNN architecture for binary image classification
- Automatic image resizing and normalization
- Visualizes training accuracy & loss curves
- Interactive **Streamlit** web app for real-time predictions
- Automatic model saving and loading

---

## 安裝與環境需求

Make sure you have Python 3.8+ installed. Then install the dependencies:

```bash
pip install tensorflow tensorflow-datasets streamlit pillow matplotlib
# or
pip install -r requirements.txt
```

## Project Structure

```bash
cat-dog-classifier/
│
├── model_training.py          # Script for training and saving the model
├── UI.py                      # Streamlit web application
├── cat_dog_classifier_tfds.h5 # Pre-trained model weights
└── README.md
```

## Training the Model

Run the following command to start training:

```bash
python model_training.py
```

## Launching the Web Interface

Start the Streamlit app:

```bash
streamlit run UI.py
```
