import os
import shutil
import random

original_cat_dir = "cats_n_dogs/Cat"
original_dog_dir = "cats_n_dogs/Dog"
base_dir = "dataset"

train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

for directory in [train_dir, validation_dir, test_dir]:
    os.makedirs(os.path.join(directory, "cats"), exist_ok=True)
    os.makedirs(os.path.join(directory, "dogs"), exist_ok=True)

all_cats = os.listdir(original_cat_dir)
all_dogs = os.listdir(original_dog_dir)

random.seed(42)
random.shuffle(all_cats)
random.shuffle(all_dogs)

train_cats = all_cats[:10000]
val_cats = all_cats[10000:11250]
test_cats = all_cats[11250:12500]

train_dogs = all_dogs[:10000]
val_dogs = all_dogs[10000:11250]
test_dogs = all_dogs[11250:12500]

def copy_files(file_list, src_dir, target_dir):
    for fname in file_list:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(target_dir, fname)
        try:
            shutil.copyfile(src, dst)
        except Exception as e:
            print(f"Copy error {src} -> {dst}: {e}")

copy_files(train_cats, original_cat_dir, os.path.join(train_dir, "cats"))
copy_files(val_cats, original_cat_dir, os.path.join(validation_dir, "cats"))
copy_files(test_cats, original_cat_dir, os.path.join(test_dir, "cats"))

copy_files(train_dogs, original_dog_dir, os.path.join(train_dir, "dogs"))
copy_files(val_dogs, original_dog_dir, os.path.join(validation_dir, "dogs"))
copy_files(test_dogs, original_dog_dir, os.path.join(test_dir, "dogs"))

print("Dataset organization completed (dataset/)")
