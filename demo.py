import argparse
import os
import random
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from dataset import ImageOperatorDataset
from model import CAN24AN, CAN24AND, CAN32AN, CAN32AND

MODEL_VARIANTS = {
    "CAN24+AN": CAN24AN,
    "CAN32+AN": CAN32AN,
    "CAN24+AND": CAN24AND,
    "CAN32+AND": CAN32AND,
}

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# loads model weights based on checkpoint path
def load_model(checkpoint_path, device):
    # take the model name from the directory
    model_name = os.path.basename(os.path.dirname(checkpoint_path))
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))
    display_name = f"{dataset_name}/{model_name}"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MODEL_VARIANTS[model_name]().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, display_name

# randomly samples image pairs from the test manifest if available
def choose_pairs(dataset, checkpoint, num_samples):
    test_basenames = checkpoint.get("split_manifest", {}).get("test_basenames")
    if not test_basenames:
        pairs = dataset.pairs
    else:
        test_basenames = set(test_basenames)
        pairs = [
            (input_path, target_path)
            for input_path, target_path in dataset.pairs
            if input_path.stem in test_basenames
        ]
        if not pairs:
            pairs = dataset.pairs
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
    parser.add_argument("--output-dir", default="demos")
    parser.add_argument("--num-samples", type=int, default=1)
    args = parser.parse_args()

    device = get_device()

    _, first_checkpoint, _ = load_model(args.checkpoints[0], device)
    dataset = ImageOperatorDataset(args.dataset)
    pairs = choose_pairs(dataset, first_checkpoint, args.num_samples)

    models = []
    for checkpoint_path in args.checkpoints:
        model, _, model_name = load_model(checkpoint_path, device)
        models.append((model_name, model))

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

    dataset_name = os.path.basename(os.path.normpath(args.dataset))
    sample_names = "_".join(input_path.stem for input_path, _ in pairs)
    output_name = f"DEMO_{dataset_name}_[{sample_names}].png"
    output_path = os.path.join(args.output_dir, output_name)
    save_stacked_comparisons(comparisons, output_path)
    print(f"Demo saved to {output_path}")


if __name__ == "__main__":
    main()
