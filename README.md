📩 Neural Network SMS Text Classifier

A Deep Learning project that automatically classifies SMS messages as Spam or Ham (Not Spam) using a neural network.

This project demonstrates how Natural Language Processing (NLP) and Deep Learning can be applied to detect unwanted messages and protect users from spam.

The system is trained using the SMS Spam Collection Dataset and implemented using TensorFlow/Keras.

---

📚 Project Overview

Spam messages are a common problem in mobile communication.
This project builds an AI-powered spam detection model capable of understanding text patterns in SMS messages and classifying them automatically.

The system performs the following steps:

1. Load SMS message dataset
2. Clean and preprocess text data
3. Convert text into numerical vectors
4. Train a neural network model
5. Evaluate model performance
6. Predict whether new messages are spam or ham

The final result is a trained AI model capable of filtering spam messages automatically.

---

🧠 Machine Learning Pipeline

The project follows a standard NLP machine learning workflow.

SMS Message

↓

Text Preprocessing

↓

Tokenization

↓

Vectorization

↓

Neural Network Model

↓

Prediction (Spam / Ham)

---

📊 Dataset

The model is trained using the SMS Spam Collection Dataset.

Dataset characteristics:

- Total messages: 5574
- Spam messages: 747
- Ham messages: 4827

Each dataset entry contains:

Label| Message
ham| I'm going to the meeting today
spam| Congratulations! You won a free ticket

Dataset format:

label,message
ham,I'm going to the meeting
spam,You won a prize

---

🧹 Data Preprocessing

Before training the neural network, the text data must be cleaned and converted into a machine-readable format.

Steps performed:

1. Convert text to lowercase
2. Remove punctuation
3. Remove special characters
4. Remove extra spaces
5. Tokenize text
6. Convert tokens to sequences
7. Pad sequences to fixed length

Example:

Original message:

Congratulations!!! You won $1000

Processed text:

congratulations you won

---

🔢 Text Vectorization

Neural networks cannot understand raw text, so the text must be converted into numbers.

This project uses Tokenization + Sequence Padding.

Example vocabulary:

Word| Index
free| 1
win| 2
prize| 3
meeting| 4

Example message:

win free prize

Converted sequence:

[2, 1, 3]

---

🤖 Neural Network Architecture

The model is built using TensorFlow / Keras.

Example architecture:

Input Layer

↓

Embedding Layer

↓

Dense Layer

↓

Dropout Layer

↓

Output Layer (Sigmoid)

Example model structure:

Embedding(vocab_size, 64)
Dense(32, activation='relu')
Dropout(0.5)
Dense(1, activation='sigmoid')

---

📈 Model Training

The neural network is trained using:

- Binary Cross Entropy Loss
- Adam Optimizer
- Accuracy Metric

Example training configuration:

epochs = 10
batch_size = 32
validation_split = 0.2

During training, the model learns patterns that distinguish spam messages from normal messages.

---

📊 Model Evaluation

After training, the model is evaluated using standard classification metrics.

Metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Example confusion matrix:

| Predicted Ham| Predicted Spam
Actual Ham| 950| 10
Actual Spam| 15| 200

---

📉 Visualization

Training performance can be visualized using graphs.

Possible plots:

- Training Accuracy vs Epochs
- Validation Accuracy vs Epochs
- Loss vs Epochs

Example tools:

- Matplotlib
- Seaborn

---

🔮 Making Predictions

Once trained, the model can classify new messages.

Example:

Input message:

Congratulations! You have won a free vacation

Prediction:

Spam

Another example:

Hey, are we meeting today?

Prediction:

Ham

---

🗂 Project Structure

sms-spam-classifier/

│
├── data/
│   └── spam.csv
│
├── models/
│   └── spam_classifier_model.h5
│
├── notebooks/
│   └── training.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── requirements.txt
└── README.md

---

⚙ Installation

Clone the repository:

git clone https://github.com/yourusername/sms-spam-classifier.git

Move into project directory:

cd sms-spam-classifier

Install dependencies:

pip install -r requirements.txt

---

📦 Requirements

Main libraries used:

tensorflow
pandas
numpy
scikit-learn
matplotlib
nltk

---

▶ Running the Project

Train the model:

python train.py

Evaluate the model:

python evaluate.py

Predict new messages:

python predict.py

---

🚀 Future Improvements

Possible improvements for this project:

Advanced NLP Models

Replace simple neural network with:

- LSTM
- GRU
- Transformer models
- BERT

---

Real-Time Spam Detection API

Build a REST API using:

- Flask
- FastAPI

---

Web Interface

Create a simple UI where users can paste SMS messages and see predictions.

---

Mobile Integration

Integrate the model into mobile apps to automatically filter spam messages.

---

Continuous Model Training

Allow the system to improve over time by retraining on new SMS data.

---

🎯 Applications

This project can be used in:

- SMS spam filtering
- Email spam detection
- Chat moderation systems
- Fraud detection
- Messaging platforms

---

👨‍💻 Author

A Deep Learning project demonstrating NLP-based spam classification using neural networks.

---

📜 License

MIT License

---

🤝 Contributions

Contributions are welcome.

Possible contributions include:

- improving model accuracy
- adding deep learning architectures
- improving preprocessing pipeline
- building deployment systems
