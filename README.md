# Image Classification Using Fashion MNIST Dataset

This project implements an image classification model using Convolutional Neural Networks (CNNs) to classify images from the Fashion MNIST dataset into various categories, such as T-shirts, trousers, sneakers, and more. It demonstrates fundamental machine learning techniques and achieves ~89% accuracy.

---

## Features
- Built with TensorFlow and Keras.
- Data preprocessing using Pandas and NumPy.
- Implements techniques like early stopping to improve model robustness.
- Confusion matrix and classification report for performance evaluation.
- Results visualized using Matplotlib.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/image-classification-fmnist.git
2. Navigate to the project directory:
   ```bash
   cd image-classification-fmnist
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the project:
   ```bash
   python3 image_classification.py

---

## Dataset
The project uses the Fashion MNIST dataset. Download the dataset files (fashion-mnist_train.csv and fashion-mnist_test.csv) and place them in the root directory of the project.

--- 

## Usage
1. Run the script: python3 image_classification.py
2. The model will train on the Fashion MNIST dataset, display the training and validation accuracy, and output a confusion matrix and classification report.
3. A plot of the training and validation accuracy will also be shown for performance visualization.

---

## Results
The CNN achieved the following performance metrics:
1. Validation accuracy : 89%
2. Precision, Recall, and F1-score for each class:

| Class         | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| T-shirt/top   | 0.87      | 0.81   | 0.84     |
| Trouser       | 0.96      | 0.99   | 0.97     |
| Pullover      | 0.83      | 0.83   | 0.83     |
| Dress         | 0.89      | 0.91   | 0.90     |
| Coat          | 0.83      | 0.84   | 0.84     |
| Sandal        | 0.98      | 0.97   | 0.97     |
| Shirt         | 0.72      | 0.73   | 0.72     |
| Sneaker       | 0.94      | 0.96   | 0.95     |
| Bag           | 0.97      | 0.97   | 0.97     |
| Ankle boot    | 0.96      | 0.95   | 0.95     |


---

## Technologies Used
Python
TensorFlow and Keras
Pandas and NumPy
Matplotlib
Scikit-learn

---
