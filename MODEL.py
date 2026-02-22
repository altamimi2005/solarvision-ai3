
import sklearn
import tensorflow as tf
import os
from pathlib import Path
import numpy 
from sklearn.model_selection import train_test_split
from LOAD import get_datasets, IMG_SIZE

print("Numpy:", numpy.__version__)
print("Sklearn:", sklearn.__version__)
print("TensorFlow:", tf.__version__)







# ===== ENV TEST (as you want) =====
print("Numpy:", numpy.__version__)
print("Sklearn:", sklearn.__version__)
print("TensorFlow:", tf.__version__)

# ===== LOAD DATA (from LOAD.py) =====
train_ds, val_ds, test_ds, class_names, info = get_datasets()
print("Dataset info:", info)
print("Label mapping:", {class_names[i]: i for i in range(len(class_names))})

# ===== PREPROCESS (inside model) =====
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10),
], name="augment")

# ===== MODEL (CNN via transfer learning) =====
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = augment(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

# ===== TRAIN SETTINGS =====
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="best_model.keras",
        monitor="val_loss",
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
]

print(model.summary())

# ===== TRAIN STAGE 1 =====
model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

# ===== FINE-TUNE STAGE 2 =====
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

# ===== FINAL TEST =====
test_results = model.evaluate(test_ds, verbose=0)
print("Test (loss, acc, precision, recall):", test_results)

# ===== SAVE FINAL MODEL =====
os.makedirs("outputs", exist_ok=True)
save_path = os.path.join("outputs", "solar_crack_model_v1")
model.save(save_path)
print("Saved model to:", save_path)
print("Classes order (0 then 1):", class_names)




