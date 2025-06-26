import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.preprocessing import image

# ✅ Load the trained model
model = tf.keras.models.load_model("waste_classifier_model.h5")

# ✅ Define your class names (same as training)
class_names = ['hazardous', 'organic', 'recyclable']

# ✅ Get the image path from command line
img_path = sys.argv[1]  # run like: python src/predict_image.py image.jpg

# ✅ Load and preprocess the image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ✅ Make prediction
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print(f"\n🧠 Predicted Class: {predicted_class}")
