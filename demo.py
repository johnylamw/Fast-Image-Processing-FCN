import argparse
import os
import random
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from dataset import ImageOperatorDataset
from utils import checkpoint_display_name, filter_pairs, get_device, load_model

OUTPUT_DIR = "output/demo"

# randomly samples image pairs from the test split
def choose_pairs(dataset, split_path, num_samples):
    pairs = filter_pairs(dataset, split_path)
    return random.sample(pairs, num_samples)

# runs a inferencing for a single image through the loaded model
def run_inference(model, input_path, device):
    image = Image.open(input_path).convert("RGB")
    x = F.to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x).clamp(0.0, 1.0)
    output = F.to_pil_image(y.squeeze(0).cpu())
    return image, output

# creates one labeled image panel
def panel(image, label, height=360, font_size=20):
    font = ImageFont.truetype("Arial.ttf", font_size)
    # font = header_font(font_size)
    width = round(image.width * height / image.height)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    header_height = font_size + 18
    canvas = Image.new("RGB", (width, height + header_height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), label, fill="black", font=font)
    return canvas

# creates one row comparing input, prediction(s), and gt
def make_comparison(input_image, predictions, target_path, gap=12):
    target = Image.open(target_path).convert("RGB")
    panels = [panel(input_image, "Input")]
    # iterate through all provided models
    for model_name, prediction in predictions:
        panels.append(panel(prediction, model_name))
    panels.append(panel(target, "GT"))
    total_width = sum(p.width for p in panels) + gap * (len(panels) - 1)
    max_height = max(p.height for p in panels)
    comparison = Image.new("RGB", (total_width, max_height), "white")
    draw = ImageDraw.Draw(comparison)

    x = 0
    for p in panels:
        comparison.paste(p, (x, 0))
        draw.rectangle((x, 0, x + p.width - 1, p.height - 1), outline=(210, 210, 210))
        x += p.width + gap

    return comparison


# stacks all sampled comparison rows and saves the final demo image
def save_stacked_comparisons(comparisons, output_path):
    total_height = sum(image.height for image in comparisons)
    max_width = max(image.width for image in comparisons)
    stacked = Image.new("RGB", (max_width, total_height), "white")

    y = 0
    for image in comparisons:
        stacked.paste(image, (0, y))
        y += image.height

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stacked.save(output_path)


# parses args and builds the demo image
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--num-samples", type=int, default=1)
    args = parser.parse_args()

    device = get_device()

    dataset = ImageOperatorDataset(args.dataset)
    dataset_name = os.path.basename(os.path.normpath(args.dataset))
    split_path = os.path.join("data_splits", dataset_name, "test_split.txt")
    pairs = choose_pairs(dataset, split_path, args.num_samples)

    models = []
    for checkpoint_path in args.checkpoints:
        model, _, _ = load_model(checkpoint_path, device)
        models.append((checkpoint_display_name(checkpoint_path), model))

    comparisons = []
    for input_path, target_path in pairs:
        predictions = []
        input_image = None
        for model_name, model in models:
            input_image, prediction = run_inference(model, input_path, device)
            predictions.append((model_name, prediction))
        comparisons.append(make_comparison(input_image, predictions, target_path))
        print(f"Input: {input_path}")
        print(f"Target: {target_path}")

    sample_names = "_".join(input_path.stem for input_path, _ in pairs)
    output_name = f"DEMO_{dataset_name}_[{sample_names}].png"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    save_stacked_comparisons(comparisons, output_path)
    print(f"Demo saved to {output_path}")


if __name__ == "__main__":
    main()
