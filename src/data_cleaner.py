import os
import json
import re


def clean_caption(caption):
    """
    Cleans the raw human caption by lowercasing and removing excess whitespace.
    """
    caption = caption.lower().strip()
    caption = re.sub(r'\s+', ' ', caption)
    return caption


def create_ground_truth():
    input_file = "data/flickr8k_sample/raw_annotations.json"
    output_file = "results/ground_truth.json"

    if not os.path.exists(input_file):
        print(f"Error: Cannot find {input_file}")
        return

    print(f" Parsing {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    #COCO-format dictionaries
    coco_format = {
        "info": {"description": "Flickr8k Sample Ground Truth"},
        "images": [],
        "annotations": [],
        "type": "captions"
    }

    annotation_id = 0

    for item in raw_data:
        file_name = item.get("file_name")
        raw_captions = item.get("raw_captions", [])

        coco_format["images"].append({
            "id": file_name,
            "file_name": file_name
        })


        for cap in raw_captions:
            coco_format["annotations"].append({
                "image_id": file_name,
                "id": annotation_id,
                "caption": clean_caption(cap)
            })
            annotation_id += 1

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)

    print(f"Ground truth processing complete!")
    print(f"Total images mapped: {len(coco_format['images'])}")
    print(f"Total captions mapped: {len(coco_format['annotations'])}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    create_ground_truth()