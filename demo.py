import argparse
import os
import random
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from dataset import ImageOperatorDataset
from utils import checkpoint_display_name, filter_pairs, get_device, load_model

OUTPUT_DIR = "output/demo"
DEMO_SHORT_EDGE = 720

# Arial is not present on every platform (e.g. Linux), so fall back to a
# guaranteed-available DejaVu font and finally PIL's bundled bitmap font.
FONT_CANDIDATES = [
    "Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
]


def load_font(font_size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


# randomly samples image pairs from the test split
def choose_pairs(dataset, split_path, num_samples):
    pairs = filter_pairs(dataset, split_path)
    return random.sample(pairs, num_samples)

# selects one image pair by input filename stem
def choose_pair_by_name(dataset, split_path, image_name):
    pairs = filter_pairs(dataset, split_path)
    for input_path, target_path in pairs:
        if input_path.stem == image_name:
            return [(input_path, target_path)]
    raise ValueError(f"No test split image matched {image_name}")

# runs a inferencing for a single image through the loaded model
def run_inference(model, input_path, device):
    image = Image.open(input_path).convert("RGB")
    if min(image.size) > DEMO_SHORT_EDGE:
        image = F.resize(image, DEMO_SHORT_EDGE)
    x = F.to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x).clamp(0.0, 1.0)
    output = F.to_pil_image(y.squeeze(0).cpu())
    return image, output

# creates one labeled image panel
def panel(image, label, height=360, font_size=20):
    font = load_font(font_size)
    width = round(image.width * height / image.height)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    header_height = font_size + 18
    canvas = Image.new("RGB", (width, height + header_height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), label, fill="black", font=font)
    return canvas

# creates one row comparing input, prediction(s), and gt
def make_comparison(input_image, predictions, target_path, columns=None, gap=12):
    target = Image.open(target_path).convert("RGB")
    panels = [panel(input_image, "Input")]
    # iterate through all provided models
    for model_name, prediction in predictions:
        panels.append(panel(prediction, model_name))
    panels.append(panel(target, "GT"))

    if columns is None:
        columns = len(panels)
    rows = [panels[index:index + columns] for index in range(0, len(panels), columns)]
    row_widths = [sum(p.width for p in row) + gap * (len(row) - 1) for row in rows]
    row_heights = [max(p.height for p in row) for row in rows]
    total_width = max(row_widths)
    total_height = sum(row_heights) + gap * (len(rows) - 1)
    comparison = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(comparison)

    y = 0
    for row, row_height in zip(rows, row_heights):
        x = 0
        for p in row:
            comparison.paste(p, (x, y))
            draw.rectangle((x, y, x + p.width - 1, y + p.height - 1), outline=(210, 210, 210))
            x += p.width + gap
        y += row_height + gap

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
    parser.add_argument("--columns", type=int, default=None)
    parser.add_argument("--image-name", default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the random test-split sampling so demo "
                             "figures are reproducible.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    device = get_device()

    dataset = ImageOperatorDataset(args.dataset)
    dataset_name = os.path.basename(os.path.normpath(args.dataset))
    split_path = os.path.join("data_splits", dataset_name, "test_split.txt")
    if args.image_name is None:
        pairs = choose_pairs(dataset, split_path, args.num_samples)
    else:
        pairs = choose_pair_by_name(dataset, split_path, args.image_name)

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
        comparisons.append(make_comparison(input_image, predictions, target_path, columns=args.columns))
        print(f"Input: {input_path}")
        print(f"Target: {target_path}")

    sample_names = "_".join(input_path.stem for input_path, _ in pairs)
    output_name = f"DEMO_{dataset_name}_[{sample_names}].png"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    save_stacked_comparisons(comparisons, output_path)
    print(f"Demo saved to {output_path}")


if __name__ == "__main__":
    main()
