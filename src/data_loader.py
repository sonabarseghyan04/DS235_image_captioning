import os
import json
from datasets import load_dataset


def load_and_save_data():
    """
    Downloads the 1,000-image test split of the Flickr8k dataset, saves the images,
    and generates a raw annotation JSON file.
    """
    output_dir = "data/flickr8k_sample"
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    raw_file_path = os.path.join(output_dir, "raw_annotations.json")

    print("Downloading full Flickr8k test dataset...")
    #Loading the entire test split (1,000 images)
    dataset = load_dataset("jxie/flickr8k", split="test")

    raw_data = []

    print(f"Saving 1,000 images to {images_dir} and extracting raw captions...")
    for i, row in enumerate(dataset):
        img_id = i
        image_name = f"image_{img_id:04d}.jpg"
        image_path = os.path.join(images_dir, image_name)

        #Saving image in RGB format
        image = row['image']
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(image_path)

        raw_data.append({
            "image_id": img_id,
            "file_name": image_name,
            "raw_captions": [
                row['caption_0'],
                row['caption_1'],
                row['caption_2'],
                row['caption_3'],
                row['caption_4']
            ]
        })

    with open(raw_file_path, 'w') as f:
        json.dump(raw_data, f, indent=4)

    print(f"Data loading complete! Raw annotations saved to: {raw_file_path}")


if __name__ == "__main__":
    load_and_save_data()