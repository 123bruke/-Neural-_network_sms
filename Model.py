import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(vocab_size=10000):
    model = models.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=32),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
