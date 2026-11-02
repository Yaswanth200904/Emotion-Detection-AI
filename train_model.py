from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 1. Configuration
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 5
DATASET_PATH = "dataset"

# 2. Load training data (80%)
train_data = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/train",
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=123
)

# 3. Load validation data (20%)
validation_data = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/train",
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=123
)

# 4. Load separate test data
test_data = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/test",
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# 5. Get emotion class names
class_names = train_data.class_names
num_classes = len(class_names)

# Count training images in each emotion folder
train_folder = Path(DATASET_PATH) / "train"

class_counts = {}

for emotion in class_names:
    emotion_folder = train_folder / emotion

    class_counts[emotion] = sum(
        1
        for file in emotion_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp"
        }
    )

if any(count == 0 for count in class_counts.values()):
    raise ValueError("At least one emotion class has no images.")

total_images = sum(class_counts.values())

# Calculate a larger weight for less frequent classes
class_weights = {
    index: total_images / (num_classes * class_counts[emotion])
    for index, emotion in enumerate(class_names)
}

print("\nTraining image counts:", class_counts)
print("Class weights:", class_weights)

if test_data.class_names != class_names:
    raise ValueError("Training and test class names do not match.")

print("\nEmotion classes:", class_names)

# Improve data-loading efficiency
train_data = train_data.prefetch(tf.data.AUTOTUNE)
validation_data = validation_data.prefetch(tf.data.AUTOTUNE)
test_data = test_data.prefetch(tf.data.AUTOTUNE)

# 6. Build the CNN
model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),

    # Normalize pixels from 0-255 to 0-1
    layers.Rescaling(1.0 / 255),

    # First convolution block
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Third convolution block
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Classification layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

# 7. Configure the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 8. Stop early if validation performance stops improving
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# 9. Train
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=15,
    callbacks=[early_stopping],
    class_weight=class_weights
)
# 10. Evaluate on the separate test dataset
test_loss, test_accuracy = model.evaluate(test_data)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.2%}")

# 11. Save the trained model
model.save("emotion_model_v2.keras")

print("\nModel saved as emotion_model.keras")