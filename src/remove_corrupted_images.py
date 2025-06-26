import os
from PIL import Image

def remove_corrupted_images(folder):
    removed = 0
    total = 0
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                total += 1

    count = 0
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(root, file)
                count += 1
                print(f"🔍 Checking ({count}/{total}): {file}", end="\r")
                try:
                    img = Image.open(filepath)
                    img.verify()  # Raises an exception if corrupted
                except Exception:
                    print(f"\n❌ Removing corrupted image: {filepath}")
                    os.remove(filepath)
                    removed += 1

    print(f"\n✅ Done! Removed {removed} corrupted images out of {total} checked.")

remove_corrupted_images("dataset")
