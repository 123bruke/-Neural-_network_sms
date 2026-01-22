import tensorflow as tf
from data_preprocessing import (
    load_dataset,
    split_dataset,
    create_vectorizer,
    vectorize_text
)
from model import build_model
messages, labels = load_dataset()

X_train, X_test, y_train, y_test = split_dataset(messages, labels)
vectorizer = create_vectorizer(X_train)
X_train_vec, X_test_vec = vectorize_text(vectorizer, X_train, X_test)
model = build_model()
history = model.fit(
    X_train_vec,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
loss, accuracy = model.evaluate(X_test_vec, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

#
def predict_message(message):
    vec = vectorizer([message])
    prob = model.predict(vec)[0][0]
    label = "spam" if prob >= 0.5 else "ham"
    return [float(prob), label]
print(predict_message("Congratulations! You won a free prize"))
print(predict_message("Hey bro, are you coming today?"))
