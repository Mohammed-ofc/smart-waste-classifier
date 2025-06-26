# src/augment_hazardous.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

# Folder where hazardous images are stored
input_folder = 'dataset/hazardous'

# Setup image augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# List all image files in the hazardous folder
images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Create augmented images
for img_name in images:
    img_path = os.path.join(input_folder, img_name)
    img = load_img(img_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    # Save 10 augmented versions per image
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=input_folder,
                              save_prefix='aug',
                              save_format='jpeg'):
        i += 1
        if i >= 10:
            break

print("✅ Augmentation complete. Check the 'hazardous' folder for new images.")
