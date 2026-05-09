import os
import time
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BlipForConditionalGeneration

def run_full_blip_inference():
    model_id = "Salesforce/blip-image-captioning-base"
    input_dir = "data/flickr8k_sample/images"
    output_dir = "results"
    output_file = os.path.join(output_dir, "blip_full_results.json")

    os.makedirs(output_dir, exist_ok=True)

    #Hardware Detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading {model_id} on {device}...")

    #Loading Processor and Model
    processor = AutoProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(
        model_id,
        use_safetensors=True
    ).to(device)

    if not os.path.exists(input_dir):
        print(f"Error: Cannot find {input_dir}")
        return

    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])

    print(f" Generating captions for {len(image_files)} images...")

    results = []

    start_time = time.time()

    for file_name in tqdm(image_files, desc="BLIP Inference"):
        img_path = os.path.join(input_dir, file_name)
        raw_image = Image.open(img_path).convert("RGB")

        inputs = processor(images=raw_image, return_tensors="pt").to(device)

        with torch.no_grad():
            #Using num_beams=3 for better quality evaluation results
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=3
            )

        caption = processor.batch_decode(output_tokens, skip_special_tokens=True)[0].strip()

        results.append({
            "image_id": file_name,
            "caption": caption
        })

    end_time = time.time()
    elapsed_time = end_time - start_time

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print("\n Inference complete!")
    print(f" Results successfully saved to: {output_file}")
    print(f" Total Inference Time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")


if __name__ == "__main__":
    run_full_blip_inference()