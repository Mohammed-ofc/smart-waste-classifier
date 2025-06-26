from PIL import Image
import os

dataset_path = "dataset"  # Your dataset folder path

def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception:
        return False

bad_images = []

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if not os.path.isdir(folder_path):  # ✅ Skip non-directories like .DS_Store
        continue
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not is_valid_image(file_path):
            print(f"❌ Removing corrupted: {file_path}")
            os.remove(file_path)
            bad_images.append(file_path)

print(f"\n✅ Finished cleaning. Removed {len(bad_images)} bad image(s).")
