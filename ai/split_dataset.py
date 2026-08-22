import os
import glob
import random
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'ai', 'dataset', 'raw')
TRAIN_DIR = os.path.join(BASE_DIR, 'ai', 'dataset', 'images', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'ai', 'dataset', 'images', 'test')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Find all XML files recursively inside raw/
xml_files = glob.glob(os.path.join(RAW_DIR, '**', '*.xml'), recursive=True)
print(f"[INFO] Found {len(xml_files)} total XML annotation files in raw/.")

if not xml_files:
    print("[ERROR] No XML files found. Check your raw folder path.")
    exit(1)

random.seed(42)
random.shuffle(xml_files)

split_idx = int(len(xml_files) * 0.8)
train_xmls = xml_files[:split_idx]
test_xmls = xml_files[split_idx:]

def copy_files(xml_list, dest_folder):
    copied_count = 0
    for xml_path in xml_list:
        shutil.copy(xml_path, dest_folder)
        base = os.path.splitext(xml_path)[0]
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            img_candidate = base + ext
            if os.path.exists(img_candidate):
                shutil.copy(img_candidate, dest_folder)
                copied_count += 1
                break
    return copied_count

train_count = copy_files(train_xmls, TRAIN_DIR)
test_count = copy_files(test_xmls, TEST_DIR)

print(f"[SUCCESS] Copied {train_count} pairs to {TRAIN_DIR}")
print(f"[SUCCESS] Copied {test_count} pairs to {TEST_DIR}")
