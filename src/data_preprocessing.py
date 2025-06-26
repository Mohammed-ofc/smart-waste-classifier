import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE = 128
CATEGORIES = ['organic', 'recyclable', 'hazardous']  # or remove 'hazardous' if not using yet

def load_data(data_dir):
    X, y = [], []
    for idx, category in enumerate(CATEGORIES):
        path = os.path.join(data_dir, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(idx)
            except:
                pass
    return np.array(X)/255.0, to_categorical(y)

if __name__ == '__main__':
    X, y = load_data('dataset')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    np.savez('dataset/data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
