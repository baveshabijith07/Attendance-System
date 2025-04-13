import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import os

# Dataset Path
DATASET_DIR = "liveness_dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")

# Check dataset
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"❌ Train directory not found: {TRAIN_DIR}")

# Image settings
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30  

# ✅ Stronger Data Augmentation for Robustness
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=45,  # ✅ Handle different head angles
    width_shift_range=0.4,
    height_shift_range=0.4,
    brightness_range=[0.3, 1.8],  # ✅ Improve lighting robustness
    zoom_range=0.4,
    horizontal_flip=True,
    shear_range=0.3,
    channel_shift_range=70.0  # ✅ Improve contrast robustness
)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)

# ✅ Use Xception for better accuracy
base_model = Xception(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

# Freeze initial layers
for layer in base_model.layers[:80]: 
    layer.trainable = False

# Custom Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)  
x = Dropout(0.3)(x)  
output = Dense(1, activation="sigmoid")(x)  

# Define Model
model = Model(inputs=base_model.input, outputs=output)

# ✅ Learning Rate Scheduler
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Train Model
model.fit(train_generator, epochs=EPOCHS, validation_data=train_generator, callbacks=[lr_schedule])

# Save Model
model.save("anti_spoofing_model_xception.h5")
print("✅ Anti-Spoofing Model Successfully Saved!")
