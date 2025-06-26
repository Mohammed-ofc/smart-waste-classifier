import os

dataset_path = "dataset"
for folder in ['hazardous', 'organic', 'recyclable']:
    folder_path = os.path.join(dataset_path, folder)
    count = len(os.listdir(folder_path))
    print(f"{folder.capitalize():<12}: {count} images")
