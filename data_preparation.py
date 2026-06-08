from PIL import Image
import argparse
import cv2
import os
from functools import partial
from multiprocessing import Pool


def _worker_init():
    # Each task runs in its own process, so keep OpenCV single-threaded inside
    # each worker. Otherwise OpenCV's internal threads multiply by the number
    # of workers and oversubscribe the cores, hurting throughput.
    cv2.setNumThreads(1)


def _tif_to_png_worker(filename, source_directory, output_directory):
    if not filename.lower().endswith((".tif", ".tiff")):
        return "skipped"
    output_filename = f"{os.path.splitext(filename)[0]}.png"
    output_path = os.path.join(output_directory, output_filename)
    if os.path.isfile(output_path):
        return "skipped"  # resume support: already converted on a prior run
    img = Image.open(os.path.join(source_directory, filename))
    img.save(output_path, "PNG")
    return "converted"


def _raw_to_processed_worker(filename, source_directory, output_directory, operator):
    if not (filename.lower().endswith(".png") or filename.lower().endswith(".jpg")):
        return "skipped"
    output_filename = f"{os.path.splitext(filename)[0]}_{operator}.png"
    output_path = os.path.join(output_directory, output_filename)
    if os.path.isfile(output_path):
        return "skipped"  # resume support: already processed on a prior run
    img = cv2.imread(os.path.join(source_directory, filename))
    if img is None:
        print(f"Skipping {filename}: OpenCV could not read it")
        return "error"
    processed = apply_operator(img, operator)
    cv2.imwrite(output_path, processed)
    return "converted"


def _run_parallel(worker, filenames, workers, label):
    total = len(filenames)
    if total == 0:
        print("  nothing to do")
        return

    counts = {"converted": 0, "skipped": 0, "error": 0}

    def tally(result, done):
        counts[result] += 1
        if done % 100 == 0 or done == total:
            print(
                f"  [{done}/{total}] {label} "
                f"(new={counts['converted']} skipped={counts['skipped']} err={counts['error']})"
            )

    if workers <= 1:
        for done, fname in enumerate(filenames, 1):
            tally(worker(fname), done)
    else:
        with Pool(workers, initializer=_worker_init) as pool:
            for done, result in enumerate(
                pool.imap_unordered(worker, filenames, chunksize=1), 1
            ):
                tally(result, done)

    print(
        f"  Finished: {counts['converted']} new, "
        f"{counts['skipped']} skipped, {counts['error']} errors"
    )


def convert_tif_to_png(source_directory, output_directory, workers=1):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print(f"Converting TIF -> PNG in {output_directory} ({workers} workers)")
    print("========================")
    filenames = sorted(os.listdir(source_directory))
    worker = partial(
        _tif_to_png_worker,
        source_directory=source_directory,
        output_directory=output_directory,
    )
    _run_parallel(worker, filenames, workers, "tif->png")


def convert_raw_to_processed(source_directory, output_directory, operator, workers=1):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print(f"Converting Raw -> Processed Operator in {output_directory} ({workers} workers)")
    print("========================")
    filenames = sorted(os.listdir(source_directory))
    worker = partial(
        _raw_to_processed_worker,
        source_directory=source_directory,
        output_directory=output_directory,
        operator=operator,
    )
    _run_parallel(worker, filenames, workers, f"raw->{operator}")


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
        default="datasets/adobe5k_processed/raw",
        help="Directory where converted PNG images are written.",
    )
    parser.add_argument(
        "--processed-output",
        default="datasets/adobe5k_processed/processed",
        help="Directory where processed PNG images are written.",
    )
    parser.add_argument(
        "--operator",
        choices=["pencil"],
        default="pencil",
        help="Processing operator to apply.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="Number of parallel worker processes. (Default: CPU count - 2)",
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
        convert_tif_to_png(args.tif_source, args.raw_output, args.workers)

    if not args.skip_processing:
        convert_raw_to_processed(
            args.raw_output,
            args.processed_output,
            args.operator,
            args.workers,
        )


if __name__ == "__main__":
    main()
