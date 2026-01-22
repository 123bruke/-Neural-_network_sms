import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset():
    url = "https://raw.githubusercontent.com/freeCodeCamp/boilerplate-neural-network-sms-text-classifier/master/sms_spam.csv"
    data = pd.read_csv(url)

    messages = data['message'].values
    labels = data['label'].values
    labels = np.array([1 if label == "spam" else 0 for label in labels])

    return messages, labels


def split_dataset(messages, labels):
    """
    Split dataset into train and test sets (80/20)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        messages, labels, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def create_vectorizer(train_text):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=10000,
        output_sequence_length=200,
        standardize="lower_and_strip_punctuation"
    )

    vectorizer.adapt(train_text)
    return vectorizer


def vectorize_text(vectorizer, train_text, test_text):
    train_vectors = vectorizer(train_text)
    test_vectors = vectorizer(test_text)
    return train_vectors, test_vectors
