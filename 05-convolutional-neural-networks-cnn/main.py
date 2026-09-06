"""
CIFAR-10 Convolutional Neural Network
Project 5 - Keras CNN Training
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import keras
from keras import layers
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report


np.random.seed(42)
tf.random.set_seed(42)


CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

print("=" * 60)
print("Loading CIFAR-10 dataset...")
print("=" * 60)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0


VAL_SIZE = 5000
x_val,   y_val   = x_train[-VAL_SIZE:], y_train[-VAL_SIZE:]
x_train, y_train = x_train[:-VAL_SIZE], y_train[:-VAL_SIZE]

print(f"  Train : {x_train.shape}")
print(f"  Val   : {x_val.shape}")
print(f"  Test  : {x_test.shape}")


data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
], name="data_augmentation")


def build_model(input_shape=(32, 32, 3), num_classes=10):
    inputs = keras.Input(shape=input_shape, name="input")


    x = data_augmentation(inputs)


    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_1")(x)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)


    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)


    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(0.30, name="drop3")(x)


    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense1")(x)
    x = layers.Activation("relu", name="relu_dense1")(x)
    x = layers.Dropout(0.50, name="drop_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return keras.Model(inputs, outputs, name="CIFAR10_CNN")


model = build_model()
model.summary()

try:
    plot_model(
        model,
        to_file="model_architecture.png",
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        dpi=120
    )
    print("\n[OK] Model architecture saved as 'model_architecture.png'")
except Exception as e:
    print(f"[!] plot_model failed: {e}")
    print("    Install with: pip install pydot && sudo apt install graphviz")


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


EPOCHS = 80
BATCH  = 64

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=7, min_lr=1e-6, verbose=1),
    ModelCheckpoint("best_model.keras", monitor="val_accuracy",
                    save_best_only=True, verbose=0)
]

print("\n" + "=" * 60)
print("TRAINING STARTED")
print("=" * 60)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=1
)


print("\n" + "=" * 60)
print("TEST EVALUATION")
print("=" * 60)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"  Test Accuracy : {test_acc * 100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")

y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
y_true = y_test.flatten()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training History", fontsize=15, fontweight="bold")
epochs_ran = range(1, len(history.history["loss"]) + 1)

axes[0].plot(epochs_ran, history.history["loss"],     label="Train Loss", lw=2)
axes[0].plot(epochs_ran, history.history["val_loss"], label="Val Loss",   lw=2, linestyle="--")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_ran, history.history["accuracy"],     label="Train Accuracy", lw=2)
axes[1].plot(epochs_ran, history.history["val_accuracy"], label="Val Accuracy",   lw=2, linestyle="--")
axes[1].axhline(y=test_acc, color="red", linestyle=":", lw=1.5,
                label=f"Test Acc: {test_acc * 100:.1f}%")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
print("\n[OK] Saved 'training_history.png'")
plt.show()


cm      = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Confusion Matrix", fontsize=15, fontweight="bold")

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
axes[0].set_title("Raw Counts")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")
axes[0].tick_params(axis="x", rotation=45)

sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            vmin=0, vmax=1, ax=axes[1])
axes[1].set_title("Normalized (Recall per class)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("[OK] Saved 'confusion_matrix.png'")
plt.show()


per_class_acc = cm_norm.diagonal()

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#e74c3c" if a < 0.75 else "#f39c12" if a < 0.85 else "#2ecc71"
          for a in per_class_acc]
bars = ax.bar(CLASS_NAMES, per_class_acc * 100, color=colors, edgecolor="black")
ax.axhline(y=85, color="green", linestyle="--", lw=1.5, label="85% target")
ax.axhline(y=test_acc * 100, color="navy", linestyle=":", lw=1.5,
           label=f"Overall: {test_acc * 100:.1f}%")
ax.set_ylim(0, 108)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Per-Class Accuracy")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
for bar, val in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val * 100:.1f}%", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("per_class_accuracy.png", dpi=150, bbox_inches="tight")
print("[OK] Saved 'per_class_accuracy.png'")
plt.show()


fig, axes = plt.subplots(4, 8, figsize=(16, 9))
fig.suptitle("Sample Test Predictions  (Green = Correct, Red = Wrong)", fontsize=13)
indices = np.random.choice(len(x_test), 32, replace=False)

for i, idx in enumerate(indices):
    ax    = axes[i // 8, i % 8]
    pred  = CLASS_NAMES[y_pred[idx]]
    true  = CLASS_NAMES[y_true[idx]]
    color = "green" if y_pred[idx] == y_true[idx] else "red"
    ax.imshow(x_test[idx])
    ax.set_title(f"P: {pred}\nT: {true}", fontsize=7, color=color)
    ax.axis("off")

plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150, bbox_inches="tight")
print("[OK] Saved 'sample_predictions.png'")
plt.show()


model.save("cifar10_model.keras")
print("\n[OK] Model exported as 'cifar10_model.keras'")

print("\n" + "=" * 60)
print("ALL DONE")
print(f"  Final Test Accuracy : {test_acc * 100:.2f}%")
print(f"  Final Test Loss     : {test_loss:.4f}")
print("=" * 60)