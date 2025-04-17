import os
import shutil

# Centralized data root
DATA_ROOT = os.path.join('..', 'data')
GTSRB_TRAIN_PATH = os.path.join(DATA_ROOT, 'GTSRB', 'Train')
GTSRB_TEST_PATH = os.path.join(DATA_ROOT, 'GTSRB', 'Test')
DEST_BASE_PATH = os.path.join(DATA_ROOT, 'SpeedLimit')

# Destination dirs for speed-limit subset
TRAIN_DEST = os.path.join(DEST_BASE_PATH, 'Train')
TEST_DEST = os.path.join(DEST_BASE_PATH, 'Test')

# Create destination directories
for split in [TRAIN_DEST, TEST_DEST]:
    for cls in range(8):  # classes 0 through 7
        dir_path = os.path.join(split, str(cls))
        os.makedirs(dir_path, exist_ok=True)

# Copy training images for speed-limit classes
print("Copying training images for speed limits (classes 0-7)...")
train_count = 0
for cls in range(8):
    src_dir = os.path.join(GTSRB_TRAIN_PATH, str(cls))
    dst_dir = os.path.join(TRAIN_DEST, str(cls))
    if os.path.isdir(src_dir):
        for fname in os.listdir(src_dir):
            src_file = os.path.join(src_dir, fname)
            dst_file = os.path.join(dst_dir, fname)
            shutil.copy(src_file, dst_file)
            train_count += 1
    else:
        print(f"⚠️ Source directory not found: {src_dir}")
print(f"✅ Copied {train_count} training images to {TRAIN_DEST}")

# Copy test images for speed-limit classes
print("Copying test images for speed limits...")
test_count = 0
for cls in range(8):
    src_dir = os.path.join(GTSRB_TEST_PATH, str(cls))
    dst_dir = os.path.join(TEST_DEST, str(cls))
    if os.path.isdir(src_dir):
        for fname in os.listdir(src_dir):
            src_file = os.path.join(src_dir, fname)
            dst_file = os.path.join(dst_dir, fname)
            shutil.copy(src_file, dst_file)
            test_count += 1
    else:
        print(f"⚠️ Source directory not found: {src_dir}")
print(f"✅ Copied {test_count} test images to {TEST_DEST}")
