import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from preprocess_data import load_data, IMG_SIZE

# Paths
TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"
MODEL_PATH = "models/rnn_invoice_model.h5"
ENCODER_PATH = "models/label_encoder.pkl"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# Load data
train_gen, class_indices = load_data(TRAIN_DIR)
val_gen, _ = load_data(VAL_DIR)

# Save label encoder
le = LabelEncoder()
le.fit(list(class_indices.keys()))
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(le, f)

# Build RNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(128, input_shape=(IMG_SIZE, IMG_SIZE), activation="relu"),
    tf.keras.layers.Dense(len(class_indices), activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save model
model.save(MODEL_PATH)

# Accuracy graph
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Training Accuracy')
plt.legend()
plt.savefig(os.path.join(REPORT_DIR, "accuracy.png"))
plt.show()

# Loss graph
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Training Loss')
plt.legend()
plt.savefig(os.path.join(REPORT_DIR, "loss.png"))
plt.show()

# Confusion Matrix
val_gen.reset()
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_gen.classes
labels = le.inverse_transform(range(len(class_indices)))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"))
plt.show()

print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=labels))
