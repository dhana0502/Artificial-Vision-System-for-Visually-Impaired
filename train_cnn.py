import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_and_train_cnn(
    train_dir="dataset/training",
    test_dir="dataset/test",
    target_size=(128, 128),
    batch_size=32,
    epochs=25,
    steps_per_epoch=250,
    validation_steps=100,
):
    """
    Build, compile, and train a simple binary CNN classifier.

    Parameters
    ----------
    train_dir : path to training image directory
    test_dir : path to test/validation image directory
    target_size : image resize dimensions (H, W)
    batch_size : mini-batch size
    epochs : number of training epochs
    steps_per_epoch : steps per epoch
    validation_steps : validation steps per epoch

    Returns
    -------
    cnn : trained Keras model
    history : training history object
    """
    # Data augmentation for training
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    test_gen = ImageDataGenerator(rescale=1.0 / 255)

    train = train_gen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
    )

    test = test_gen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
    )

    # Model architecture
    cnn = Sequential([
        # Block 1
        Conv2D(32, (3, 3), input_shape=(target_size[0], target_size[1], 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        # Block 2
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        # Classifier head
        Flatten(),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    cnn.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    cnn.summary()

    # Training
    history = cnn.fit(
        train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=test,
        validation_steps=validation_steps,
    )

    # Plot accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()

    return cnn, history


if __name__ == "__main__":
    model, hist = build_and_train_cnn()
    model.save("cnn_face_model.h5")
    print("Model saved to cnn_face_model.h5")
