"""
TF2 port of Chen, Xu & Koltun "Fast Image Processing with Fully-Convolutional
Networks" (CVPR 2017), adapted to the team's data layout for the "Reproduced"
criterion: run the authors' reference architecture/recipe and compare against
our from-scratch PyTorch CAN.

Origin: ported from the authors' released `parameterized.py` (TF1 + contrib.slim).
Changes made for a like-for-like comparison against our PyTorch model:
  * TF1 graph-mode + slim  ->  TF2 / Keras 3 (contrib.slim no longer exists).
  * Removed the parameterized-operator machinery: 4-channel input (RGB + tiled
    hyperparameter) reduced to plain 3-channel RGB; dropped the per-image .txt
    lambda files and the parameter-sweep video/export blocks.
  * prepare_data() now reads OUR split files (data_splits/<dataset>/*.txt) and
    OUR pair layout (inputs/<name>.png, targets/<name>_pencil.png), so the TF
    model trains/tests on exactly the same images as the PyTorch model.
  * Reconciled confounds vs our PyTorch pipeline: convert BGR->RGB on load, and
    apply the same random short-side resize (320-1440) during training.

Preserved from the reference (these define the "Reproduced" baseline):
  * Architecture: 9x [3x3 conv, dilation 1,2,4,8,16,32,64,128,1] + 1x1 -> RGB.
  * LeakyReLU(0.2); adaptive norm w0*x + w1*BN(x) with w0=1, w1=0 at init.
  * Identity kernel initialization (network starts ~identity).
  * Loss = mean squared error; optimizer = Adam(1e-4).
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import time
from pathlib import Path

# --- CUDA ptxas PATH guard (must run before `import tensorflow`) -------------
# Two problems on native-Windows TF 2.10 + Ampere, both about ptxas:
#  1) A system-wide CUDA toolkit on PATH (observed: CUDA v13.x under
#     "NVIDIA GPU Computing Toolkit") ships a ptxas.exe incompatible with this
#     TF 2.10 (CUDA 11) build. TF shells out to it on the first GPU kernel compile
#     and the process hard-crashes with no Python traceback. -> drop those dirs.
#  2) Without any compatible ptxas, TF logs "Profiling failure"/"Relying on driver"
#     and disables cuDNN autotune redzone checks. We ship a CUDA-11 ptxas via the
#     `nvidia-cuda-nvcc-cu11` wheel; if present, prepend it so TF uses it.
_parts = os.environ.get("PATH", "").split(os.pathsep)
_kept = [p for p in _parts if "NVIDIA GPU Computing Toolkit" not in p]
_ptxas_bin = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cuda_nvcc", "bin")
if os.path.isfile(os.path.join(_ptxas_bin, "ptxas.exe")):
    _kept.insert(0, _ptxas_bin)
os.environ["PATH"] = os.pathsep.join(_kept)

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------------
# Model (faithful port of the reference `build`)
# ----------------------------------------------------------------------------

def lrelu(x):
    return tf.maximum(x * 0.2, x)


class IdentityInitializer(keras.initializers.Initializer):
    """Delta at kernel center copying input channel i -> output channel i.

    Mirrors the reference identity_initializer(); the whole network therefore
    starts as an approximate identity map. This is a deliberate difference from
    our PyTorch model (default Kaiming init) and a likely explanation for any
    train-speed/accuracy gap, so it is preserved here on purpose.
    """

    def __call__(self, shape, dtype=None):
        array = np.zeros(shape, dtype=np.float32)
        cx, cy = shape[0] // 2, shape[1] // 2
        for i in range(min(shape[2], shape[3])):
            array[cx, cy, i, i] = 1.0
        return tf.constant(array, dtype=dtype or tf.float32)


class NativeConv2D(layers.Layer):
    """2D conv that calls tf.nn.conv2d's native `dilations` (cuDNN) directly.

    Keras Conv2D realizes dilated convolutions via SpaceToBatch: a dilation rate d
    moves a dxd block into the batch dim (batch -> batch * d^2), so the rate-128
    layer balloons batch to 16384. At the paper's training resolution (~1440 short
    side) that OOMs a 10 GB GPU and is very slow. tf.nn.conv2d's `dilations` arg
    uses cuDNN's native dilated convolution instead (no SpaceToBatch), matching how
    PyTorch's nn.Conv2d(dilation=...) behaves — fits full resolution and is fast.

    Same weights/params as Keras Conv2D (kernel + bias), so param count and the
    identity-init scheme are unchanged.
    """

    def __init__(self, filters, kernel_size=3, dilation_rate=1,
                 kernel_initializer="glorot_uniform", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.kernel_size, self.kernel_size, in_ch, self.filters),
            initializer=self.kernel_initializer, trainable=True)
        self.bias = self.add_weight(
            name="bias", shape=(self.filters,), initializer="zeros", trainable=True)

    def call(self, x):
        y = tf.nn.conv2d(
            x, self.kernel, strides=[1, 1, 1, 1], padding="SAME",
            dilations=[1, self.dilation_rate, self.dilation_rate, 1])
        return tf.nn.bias_add(y, self.bias)


class AdaptiveNorm(layers.Layer):
    """Reference nm(): w0*x + w1*BN(x), with w0=1.0 and w1=0.0 at init."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bn = layers.BatchNormalization()

    def build(self, input_shape):
        self.w0 = self.add_weight(name="w0", shape=(), initializer="ones", trainable=True)
        self.w1 = self.add_weight(name="w1", shape=(), initializer="zeros", trainable=True)

    def call(self, x, training=None):
        return self.w0 * x + self.w1 * self.bn(x, training=training)


def build_can(width: int = 32, norm: str = "adaptive") -> keras.Model:
    """CAN32 by default (depth 10, width 32). Matches paper ~75K params."""
    rates = [1, 2, 4, 8, 16, 32, 64, 128, 1]
    inp = keras.Input(shape=(None, None, 3))
    net = inp
    for rate in rates:
        net = NativeConv2D(
            width, kernel_size=3, dilation_rate=rate,
            kernel_initializer=IdentityInitializer(),
        )(net)
        if norm == "adaptive":
            net = AdaptiveNorm()(net)
        elif norm == "batch":
            net = layers.BatchNormalization()(net)
        elif norm != "none":
            raise ValueError(f"Unknown norm: {norm}")
        net = layers.Lambda(lrelu)(net)
    # output 1x1 conv keeps Keras' default (glorot) init, as in the reference
    out = NativeConv2D(3, kernel_size=1, dilation_rate=1)(net)
    return keras.Model(inp, out)


# ----------------------------------------------------------------------------
# Data (repointed at OUR layout + split files; lambda machinery removed)
# ----------------------------------------------------------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def read_split(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_pairs(dataset_dir: str, operator_suffix: str = "_pencil"):
    """Return {basename: (input_path, target_path)} for a dataset directory.

    Matches our ImageOperatorDataset pairing: inputs/<name>.<ext> against
    targets/<name><suffix>.<ext>.
    """
    dataset = Path(dataset_dir)
    input_dir, target_dir = dataset / "inputs", dataset / "targets"
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    if not target_dir.exists():
        raise FileNotFoundError(target_dir)

    target_by_stem = {
        p.stem.removesuffix(operator_suffix): p
        for p in target_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    }
    pairs = {}
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        target = target_by_stem.get(p.stem)
        if target is not None:
            pairs[p.stem] = (p, target)
    return pairs


def split_pairs(dataset_dir, split_path, operator_suffix="_pencil"):
    """Pairs whose basename is listed in split_path (mirrors filter_pairs)."""
    pairs = build_pairs(dataset_dir, operator_suffix)
    wanted = read_split(split_path)
    selected = [(pairs[b][0], pairs[b][1]) for b in wanted if b in pairs]
    if not selected:
        raise ValueError(f"No pairs matched split {split_path}")
    return selected


def load_image_rgb(path) -> np.ndarray:
    """Load as float32 RGB in [0,1]. cv2 reads BGR, so convert to match our
    PIL/RGB PyTorch pipeline (otherwise channel order is a silent confound)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.float32(rgb) / 255.0


def resize_short_side(img: np.ndarray, short: int) -> np.ndarray:
    """Resize so the shorter side == `short`, preserving aspect ratio.
    Mirrors torchvision F.resize(int)."""
    h, w = img.shape[:2]
    if h <= w:
        new_h, new_w = short, max(1, round(w * short / h))
    else:
        new_w, new_h = short, max(1, round(h * short / w))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------
# Train / evaluate / export
# ----------------------------------------------------------------------------

def train(args):
    dataset_name = os.path.basename(os.path.normpath(args.dataset))
    split_path = os.path.join(args.splits, dataset_name, "train_split.txt")
    pairs = split_pairs(args.dataset, split_path, args.operator_suffix)
    print(f"Training on {len(pairs)} pairs from {split_path}")

    model = build_can(width=args.width, norm=args.norm)
    print(f"Model params: {model.count_params()}")
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    out_dir = os.path.join(args.outputs, dataset_name, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer,
                               iteration=tf.Variable(0, dtype=tf.int64))
    manager = tf.train.CheckpointManager(ckpt, out_dir, max_to_keep=3)

    if args.resume and manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"Resumed from {manager.latest_checkpoint} (iter {int(ckpt.iteration)})")

    log_path = os.path.join(out_dir, f"{args.run_name}_train_log.csv")
    new_log = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    writer = csv.DictWriter(log_file, fieldnames=[
        "iteration", "loss", "seconds_per_iter", "height", "width"])
    if new_log:
        writer.writeheader()

    # in-memory cache (reference caches decoded images too); resize is per-iter.
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rng = random.Random(args.seed)

    @tf.function(reduce_retracing=True)
    def train_step(x, y):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(pred - y))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    start_iter = int(ckpt.iteration)
    start_time = time.time()
    for iteration in range(start_iter + 1, args.iterations + 1):
        idx = rng.randrange(len(pairs))
        if idx not in cache:
            in_path, tgt_path = pairs[idx]
            cache[idx] = (load_image_rgb(in_path), load_image_rgb(tgt_path))
        inp, tgt = cache[idx]

        # same random short-side resize as our PyTorch transform (320-1440)
        short = rng.randint(args.min_res, args.max_res)
        if args.max_pixels:  # optional cap to avoid OOM on extreme aspect ratios
            h, w = inp.shape[:2]
            ar = max(h, w) / min(h, w)
            short = min(short, max(args.min_res, int((args.max_pixels / ar) ** 0.5)))
        x = resize_short_side(inp, short)
        y = resize_short_side(tgt, short)

        t0 = time.time()
        loss = train_step(x[None], y[None])
        dt = time.time() - t0
        ckpt.iteration.assign(iteration)

        if iteration % 100 == 0 or iteration == args.iterations:
            print(f"iter {iteration}/{args.iterations} loss={float(loss):.6f} "
                  f"({dt:.3f}s, {x.shape[1]}x{x.shape[0]})")
        writer.writerow({"iteration": iteration, "loss": float(loss),
                         "seconds_per_iter": dt, "height": x.shape[0],
                         "width": x.shape[1]})
        log_file.flush()

        if iteration % args.ckpt_every == 0:
            manager.save()

    manager.save()
    log_file.close()
    print(f"Done. Checkpoints in {out_dir}")


def export_predictions(args):
    """Run the trained model on the test split and write predictions as PNGs,
    so they can be scored by our existing evaluate.py on identical metrics."""
    dataset_name = os.path.basename(os.path.normpath(args.dataset))
    split_path = os.path.join(args.splits, dataset_name, "test_split.txt")
    pairs = split_pairs(args.dataset, split_path, args.operator_suffix)

    out_dir = os.path.join(args.outputs, dataset_name, args.run_name)

    # Rebuild the architecture and restore weights from the training checkpoint.
    # This deliberately avoids Keras-version-specific model serialization (the
    # .keras v3 format / load_model safe_mode), so export runs unchanged on both
    # Keras 2 (TF<=2.10, native-Windows GPU) and Keras 3 (TF>=2.16). The functional
    # model's variables exist at build time, so the restore is eager.
    model = build_can(width=args.width, norm=args.norm)
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, out_dir, max_to_keep=3)
    if not manager.latest_checkpoint:
        raise FileNotFoundError(f"No checkpoint found in {out_dir}")
    # expect_partial(): the training checkpoint also holds optimizer+iteration,
    # which we intentionally don't restore for inference.
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    print(f"Restored {manager.latest_checkpoint}")

    pred_dir = os.path.join(out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for in_path, _ in pairs:
        img = load_image_rgb(in_path)
        if args.short_edge:
            img = resize_short_side(img, args.short_edge)
        pred = model(img[None], training=False).numpy()[0]
        pred = np.clip(pred, 0.0, 1.0) * 255.0
        bgr = cv2.cvtColor(np.uint8(pred), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(pred_dir, f"{in_path.stem}.png"), bgr)
    print(f"Wrote {len(pairs)} predictions to {pred_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="TF2 reference CAN (Reproduced baseline)")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--dataset", default="datasets/adobe5kA")
    t.add_argument("--splits", default="data_splits")
    t.add_argument("--outputs", default="tf_model_runs")
    t.add_argument("--run-name", default="CAN32+AN_tf")
    t.add_argument("--width", type=int, default=32)
    t.add_argument("--norm", choices=["adaptive", "batch", "none"], default="adaptive")
    t.add_argument("--iterations", type=int, default=500_000)
    t.add_argument("--min-res", type=int, default=320)
    t.add_argument("--max-res", type=int, default=1440)
    t.add_argument("--max-pixels", type=int, default=None)
    t.add_argument("--ckpt-every", type=int, default=10_000)
    t.add_argument("--operator-suffix", default="_pencil")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--resume", action="store_true")
    t.set_defaults(func=train)

    e = sub.add_parser("export")
    e.add_argument("--dataset", default="datasets/adobe5kA")
    e.add_argument("--splits", default="data_splits")
    e.add_argument("--outputs", default="tf_model_runs")
    e.add_argument("--run-name", default="CAN32+AN_tf")
    e.add_argument("--width", type=int, default=32,
                   help="Must match the value used at train time (rebuild target).")
    e.add_argument("--norm", choices=["adaptive", "batch", "none"], default="adaptive",
                   help="Must match the value used at train time (rebuild target).")
    e.add_argument("--short-edge", type=int, default=1080)
    e.add_argument("--operator-suffix", default="_pencil")
    e.set_defaults(func=export_predictions)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)