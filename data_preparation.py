from PIL import Image
import argparse
import cv2
import os


def convert_tif_to_png(source_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print(f"Converting TIF -> PNG in {output_directory}")
    print("========================")
    for filename in os.listdir(source_directory):
        if filename.lower().endswith((".tif", ".tiff")):
            # load .tif file
            img = Image.open(os.path.join(source_directory, filename))
            output_filename = f"{os.path.splitext(filename)[0]}.png"

            # convert to png
            img.save(os.path.join(output_directory, output_filename), "PNG")
            print(f"{filename} saved as {output_filename}")


def convert_raw_to_processed(source_directory, output_directory, operator):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print(f"Converting Raw -> Processed Operator in {output_directory}")
    print("========================")
    for filename in os.listdir(source_directory):
        if filename.lower().endswith(".png") or filename.lower().endswith(".jpg"):
            img = cv2.imread(os.path.join(source_directory, filename))
            if img is None:
                print(f"Skipping {filename}: OpenCV could not read it")
                continue
            processed = apply_operator(
                img,
                operator,
            )
            output_filename = f"{os.path.splitext(filename)[0]}_{operator}.png"
            cv2.imwrite(os.path.join(output_directory, output_filename), processed)


def apply_operator(image, operator):
    # add more operators if needed
    if operator == "pencil":
        gray_sketch, _ = cv2.pencilSketch(
            image,
            sigma_s=60,
            sigma_r=0.07,
            shade_factor=0.05,
        )
        return gray_sketch
    raise ValueError(f"Unknown operator: {operator}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Adobe FiveK images as raw PNGs and processed operator outputs."
    )
    parser.add_argument(
        "--tif-source",
        default="fivek_data/MITAboveFiveK/processed/tiff16_a",
        help="Directory containing expert TIFF images.",
    )
    parser.add_argument(
        "--raw-output",
        default="dataset/adobe5k_processed/raw",
        help="Directory where converted PNG images are written.",
    )
    parser.add_argument(
        "--processed-output",
        default="dataset/adobe5k_processed/processed",
        help="Directory where processed PNG images are written.",
    )
    parser.add_argument(
        "--operator",
        choices=["pencil"],
        default="pencil",
        help="Processing operator to apply.",
    )
    parser.add_argument(
        "--skip-tif-conversion",
        action="store_true",
        help="Skip converting TIFF files to raw PNG files.",
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip applying the processing operator to raw PNG files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_tif_conversion:
        convert_tif_to_png(args.tif_source, args.raw_output)

    if not args.skip_processing:
        convert_raw_to_processed(
            args.raw_output,
            args.processed_output,
            args.operator,
        )


if __name__ == "__main__":
    main()
