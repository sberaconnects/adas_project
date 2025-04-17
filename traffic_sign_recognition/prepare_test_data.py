import os
import pandas as pd
import shutil

# Centralized Paths
DATA_ROOT = os.path.join('..', 'data')
CSV_PATH = os.path.join(DATA_ROOT, 'TrafficSign', 'Test.csv')
IMG_BASE_PATH = os.path.join(DATA_ROOT, 'TrafficSign')
OUTPUT_PATH = os.path.join(DATA_ROOT, 'TrafficSign', 'Test')

# Load CSV
df = pd.read_csv(CSV_PATH)

# Create label folders
for label in df['ClassId'].unique():
    label_folder = os.path.join(OUTPUT_PATH, str(label))
    os.makedirs(label_folder, exist_ok=True)

# Move files to respective folders
moved_count = 0
for _, row in df.iterrows():
    src = os.path.join(IMG_BASE_PATH, row['Path'])
    dst = os.path.join(OUTPUT_PATH, str(
        row['ClassId']), os.path.basename(row['Path']))
    try:
        if os.path.exists(src):
            shutil.move(src, dst)
            moved_count += 1
        else:
            print(f"⚠️ File not found: {src}")
    except Exception as e:
        print(f"❌ Error moving file {src}: {e}")

print(f"✅ {moved_count} test images have been organized by class ID.")
