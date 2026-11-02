
import numpy as np
import tensorflow as tf
from pathlib import Path

# Locate the project files
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "test"
MODEL_PATH = BASE_DIR / "emotion_model_v2.keras"
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64

# Check that the trained model exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

# Load the trained CNN
model = tf.keras.models.load_model(MODEL_PATH)

# Load the separate test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    str(DATASET_PATH),
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = test_data.class_names
print("\nEmotion classes:", class_names)

# Collect true labels and model predictions
all_true = []
all_predicted = []

for images, labels in test_data:

    # The model performs normalization internally
    probabilities = model.predict(images, verbose=0)

    predictions = np.argmax(probabilities, axis=1)

    all_true.extend(labels.numpy().tolist())
    all_predicted.extend(predictions.tolist())

y_true = np.array(all_true)
y_pred = np.array(all_predicted)

# Overall accuracy
accuracy = np.mean(y_true == y_pred)

print(f"\nOverall test accuracy: {accuracy:.2%}")

# Count how often each class is predicted
print("\n--- Prediction distribution ---")

for index, emotion in enumerate(class_names):
    count = int(np.sum(y_pred == index))
    percentage = count / len(y_pred) * 100

    print(
        f"{emotion:10s}: "
        f"{count:5d} predictions ({percentage:5.2f}%)"
    )

# Build a confusion matrix
num_classes = len(class_names)

confusion = np.zeros(
    (num_classes, num_classes),
    dtype=int
)

np.add.at(confusion, (y_true, y_pred), 1)

print("\n--- Confusion matrix ---")
print("Rows = actual class")
print("Columns = predicted class\n")

print(f"{'Actual / Pred.':>14}", end="")

for emotion in class_names:
    print(f"{emotion[:8]:>10}", end="")

print()

for index, emotion in enumerate(class_names):
    print(f"{emotion:>14}", end="")

    for count in confusion[index]:
        print(f"{count:>10}", end="")

    print()

# Precision and recall for each class
print("\n--- Per-class performance ---")

for index, emotion in enumerate(class_names):
    true_positive = confusion[index, index]
    actual_count = int(np.sum(y_true == index))
    predicted_count = int(np.sum(y_pred == index))

    recall = (
        true_positive / actual_count
        if actual_count else 0.0
    )

    precision = (
        true_positive / predicted_count
        if predicted_count else 0.0
    )

    print(
        f"{emotion:10s} | "
        f"Precision: {precision:.2%} | "
        f"Recall: {recall:.2%} | "
        f"Actual images: {actual_count} | "
        f"Predictions: {predicted_count}"
    )
