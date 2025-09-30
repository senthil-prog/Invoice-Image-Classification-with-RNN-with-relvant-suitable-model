from PIL import Image, ImageDraw, ImageFont
import os
import random

IMG_SIZE = 128
CLASSES = ["invoice", "receipt", "bill", "purchase_order"]
NUM_TRAIN = 20
NUM_VAL = 5

def generate_images(output_dir, num_images):
    os.makedirs(output_dir, exist_ok=True)
    for cls in CLASSES:
        cls_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(num_images):
            img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=255)
            draw = ImageDraw.Draw(img)
            draw.text((10, 50), f"{cls}_{i}", fill=0)
            img.save(os.path.join(cls_dir, f"{cls}_{i}.png"))

# Generate train and validation data
generate_images("data/train", NUM_TRAIN)
generate_images("data/validation", NUM_VAL)

print("Synthetic dataset generated!")
