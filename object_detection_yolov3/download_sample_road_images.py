import os
import urllib.request

# Destination folder
DATA_DIR = os.path.join("..", "data", "images")
os.makedirs(DATA_DIR, exist_ok=True)

# Sample road scene image URLs
# Verified downloadable road scene image URLs
image_urls = [
    # COCO sample (dog, bike, car)
    "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg",
    # Urban traffic image
    "https://cdn.pixabay.com/photo/2018/01/20/08/33/road-3095838_1280.jpg",
    # Busy street scene
    "https://cdn.pixabay.com/photo/2016/11/18/16/27/london-1837614_1280.jpg"
]

# Download images


def download_images():
    for i, url in enumerate(image_urls):
        filename = f"road{i+1}.jpg"
        dest_path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(dest_path):
            print(f"⬇️  Downloading {filename}...")
            urllib.request.urlretrieve(url, dest_path)
            print(f"✅ Saved to {dest_path}")
        else:
            print(f"✔️  {filename} already exists.")


if __name__ == "__main__":
    download_images()
