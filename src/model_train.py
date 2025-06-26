import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

print("📢 model_train.py script started...")
print("🚀 Starting model training pipeline...")

# Step 1: Load dataset
print("🔄 Loading dataset from folder...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"✅ Dataset loaded. Classes found: {class_names}")
print(f"📦 Total batches: Train = {len(train_ds)}, Val = {len(val_ds)}")

# Step 2: Prefetching for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Step 3: Load EfficientNetB0 base
print("📡 Loading pretrained EfficientNetB0...")
base_model = EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                            include_top=False,
                            weights='imagenet')
base_model.trainable = False  # Freeze base model

# Step 4: Add classification head
print("🧱 Building classification layers...")
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Step 5: Compile the model
print("🔧 Compiling model...")
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Callbacks
checkpoint_cb = ModelCheckpoint('model_checkpoint.h5', save_best_only=True)
earlystop_cb = EarlyStopping(patience=5, restore_best_weights=True)
callbacks = [checkpoint_cb, earlystop_cb]
print("💾 Checkpointing enabled: model_checkpoint.h5")

# Step 7: Train the model
print(f"🏋️ Starting training for {EPOCHS} epochs...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Step 8: Save the final model
print("📦 Saving final model to waste_classifier_model.h5...")
model.save('waste_classifier_model.h5')
print("✅ Training complete. Model saved successfully!")
