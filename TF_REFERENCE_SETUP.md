# Running the TF reference baseline (the "Reproduced" comparison)

Goal: train/run the authors' reference CAN architecture (`tf_reference.py`) on the
**same images and splits** as our PyTorch CAN, then score **both** with the same
metric code (`evaluate.py` / torchmetrics) and put the numbers side by side.

The decisions for this comparison:
- **Matched training budget** — TF trains for the same number of iterations the
  PyTorch CAN32+AN trained for (500,000), so any gap reflects framework/implementation
  rather than training time.
- **Init divergence documented, not matched** — TF keeps the authors' identity-kernel
  init (part of the paper recipe); PyTorch keeps Kaiming. This is a *known, documented*
  difference, not something we equalize.

---

## 0. Hard rule: TensorFlow lives in a SEPARATE environment

Do **not** add `tensorflow` to `pyproject.toml` / the uv project. It only shares
`dataset/` and `data_splits/` on disk. Keeping it out avoids CUDA dependency
conflicts with our PyTorch stack.

The TF version doesn't enter the comparison (metrics come from our torchmetrics scoring
the exported PNGs), only the architecture/recipe does — so any TF that runs the model is
valid. Record the actual TF/CUDA version in the writeup. `tf_reference.py` is
Keras-version-agnostic (export rebuilds the model and restores from the training
checkpoint instead of using the Keras-3 `.keras` format), so it runs on both modern TF
(2.16+/Keras 3) and legacy TF 2.10.

---

## 1. Create the separate TF environment

Pick the option matching the GPU you'll train on.

### Option A — Linux + modern TF (RECOMMENDED; required for RTX 50-series/Blackwell)
A Blackwell GPU (RTX 5090 = sm_120, needs CUDA 12.8) is **far newer than TF 2.10's CUDA
11.2** — TF 2.10 will not drive it. Use a current `tensorflow[and-cuda]` on Linux, which
bundles a CUDA 12.x runtime with Blackwell support and gives the GPU natively (no WSL
gymnastics on a real Linux box).
```bash
conda create -n tfref python=3.11 -y
conda activate tfref
pip install "tensorflow[and-cuda]" opencv-python numpy   # latest TF; verify it lists sm_120 support
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```
On this path the Windows-only guards in `tf_reference.py` (CUDA-toolkit PATH strip, ptxas
wheel) are harmless no-ops, and `numpy 2` is fine. 32 GB also removes the resolution cap —
you can train at the paper's full short-side range (see §3a). **If the smoke test (§2)
shows no GPU or crashes, the installed TF is too old for Blackwell — upgrade TF (or try
the nightly) until `list_physical_devices('GPU')` is non-empty and a train step runs.**

### Option B — native Windows + TF 2.10 (fallback; validated on the RTX 3080)
Last TF with native-Windows GPU; works on Ampere (3080) but is memory/speed limited and
**will not work on Blackwell**. In an **Anaconda Prompt**:
```bat
conda create -n tfref python=3.10 -y
conda activate tfref
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
pip install "tensorflow<2.11" "numpy<2" opencv-python nvidia-cuda-nvcc-cu11
```
> Windows TF 2.10 specifics (not needed on Option A):
> - `nvidia-cuda-nvcc-cu11` ships a CUDA-11 `ptxas.exe`; `tf_reference.py` auto-adds it to
>   PATH and strips any system CUDA toolkit (see its header) — prevents the hard GPU crash
>   and the "Profiling failure" autotune warnings on Ampere.
> - Pin `numpy<2`: TF 2.10 was built against NumPy 1.x; NumPy 2 makes it fail to import
>   (`_ARRAY_API not found`). opencv's `numpy>=2` metadata warning is cosmetic.

### Confirm the GPU is visible (one-liner)
```bat
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```
A non-empty list means TF sees the GPU. Empty = it will silently run on CPU (re-check the
cudatoolkit/cudnn versions above). Expect a few harmless TF deprecation warnings on 2.10.

> Keep this `tfref` env entirely separate from the uv/PyTorch project env — they share
> only `dataset/` and `data_splits/` on disk, never a Python process.

---

## 2. GPU smoke test FIRST (≈5 min, before any long run)

`tf_reference.py` was only exercised on a CPU TF build. Before a multi-hour run,
confirm on a tiny toy dataset that TF builds, a train step runs on GPU, checkpoints are
written, and export produces PNGs. Run from the repo root in the `tfref` env.

```bat
REM (a) generate ~6 fake pairs in the inputs/targets + split layout
python make_toy_dataset.py --root dataset/_toy --num 6 --test 2

REM (b) tiny train run: 200 iters at small resolution
python tf_reference.py train --dataset dataset/_toy --run-name CAN32+AN_tf_smoke --iterations 200 --min-res 96 --max-res 160 --ckpt-every 100

REM (c) export predictions for the toy test split
python tf_reference.py export --dataset dataset/_toy --run-name CAN32+AN_tf_smoke --short-edge 128
```

Pass criteria:
- (a) prints "Wrote 6 pairs ..." and creates `dataset/_toy/{inputs,targets}` +
  `data_splits/_toy/{train,test}_split.txt`.
- (b) prints `Model params: 76149`, loss lines, and writes
  `tf_model_runs/_toy/CAN32+AN_tf_smoke/` with checkpoint files (`ckpt-*`, `checkpoint`).
  Confirm it used the GPU (the GPU check in step 1 was non-empty).
- (c) prints `Restored ...` then "Wrote 2 predictions to .../predictions".

Then sanity-check the **scoring path** (this runs in the PyTorch/uv env, not the TF env):
```bash
# from the uv/PyTorch env:
<python> evaluate.py dataset/_toy \
    --predictions tf_model_runs/_toy/CAN32+AN_tf_smoke/predictions \
    --pred-label CAN32+AN_tf_smoke
```
It should print MSE/PSNR/SSIM and write `output/evaluate/_toy_CAN32+AN_tf_smoke_predeval.csv`.

Clean up the toy artifacts afterwards: `dataset/_toy`, `data_splits/_toy`,
`tf_model_runs/_toy`, `output/evaluate/_toy_*`.

---

## 3. The real run (adobe5kA — the paper's dataset, matched budget)

The comparison uses **adobe5kA**, matching the paper. The PyTorch baseline is the
already-trained `model_runs/adobe5kA/CAN32+AN/CAN32+AN_final.pt` (500k iters).

> **Dataset:** `datasets/adobe5kA/{inputs,targets}` — 5000 pairs, present and verified
> (every basename in `data_splits/adobe5kA/{train,test}_split.txt` is matched, so the
> trained PyTorch checkpoint aligns with the test split). If you ever need to rebuild it
> from scratch, the repo's scripts reproduce the canonical FiveK basenames:
> ```bat
> uv run python download_adobe5k.py --size 5000 --bypass-splits --experts a
> uv run python data_preparation.py --tif-source fivek_data\MITAboveFiveK\processed\tiff16_a --raw-output datasets\adobe5kA\inputs --processed-output datasets\adobe5kA\targets --operator pencil
> ```
> (`cv2.pencilSketch` is deterministic, so regenerated targets match — same OpenCV version.
> Note the full FiveK download is ~320 GB of TIFF+DNG.)

> **Cross-machine note (training on the friend's Linux RTX 5090).** Training and scoring
> can run on different machines — they only exchange files. Copy the repo code + the
> ~30 GB `datasets/adobe5kA/` + `data_splits/adobe5kA/` to the Linux box (e.g.
> `rsync -av`), train+export there (3a–3b), then copy the exported `predictions/` folder
> back here and score in the uv env (3c). The predictions are ~2500 PNGs at 1080px (small).

### 3a. Train the TF reference — 500k matched budget
```bash
conda activate tfref
# Linux + RTX 5090 (32 GB): full resolution, no cap needed
python tf_reference.py train --dataset datasets/adobe5kA --run-name CAN32+AN_tf --iterations 500000 --ckpt-every 10000 --resume
```
- **Resolution:** the default 320–1440 short-side range (matching the paper) fits easily in
  32 GB — no `--max-res`/`--max-pixels` cap needed (the `NativeConv2D` layer uses cuDNN's
  native dilation, not `SpaceToBatch`).
- *(For reference, the 10 GB Windows 3080 fallback needs `--max-res 1024 --max-pixels
  1500000` and runs ~2.7–3.4 s/iter → ~7–9 days. The 5090 avoids both the cap and the
  slowness.)*
- `--resume` continues from the newest 10k checkpoint if interrupted.

### 3b. Export TF predictions on the test split
```bash
python tf_reference.py export --dataset datasets/adobe5kA --run-name CAN32+AN_tf --short-edge 1080
# -> tf_model_runs/adobe5kA/CAN32+AN_tf/predictions/*.png  (copy this folder back to score)
```

### 3c. Score BOTH models with the SAME metric code — uv/PyTorch env (this machine)
```bat
REM PyTorch CAN32+AN (runs the checkpoint over the test split)
uv run python evaluate.py datasets/adobe5kA model_runs/adobe5kA/CAN32+AN/CAN32+AN_final.pt --short-edge 1080

REM TF reference (scores the exported PNGs — identical torchmetrics code)
uv run python evaluate.py datasets/adobe5kA --predictions tf_model_runs/adobe5kA/CAN32+AN_tf/predictions --pred-label CAN32+AN_tf
```
Outputs:
- `output/evaluate/adobe5kA_shortedge1080_eval.csv`    (PyTorch)
- `output/evaluate/adobe5kA_CAN32+AN_tf_predeval.csv`  (TF)

---

## 4. Side-by-side table

Both CSVs share a schema (`dataset, model, mse, psnr, ssim, time_ms, ...`). Combine the
two rows into the comparison table. `time_ms` is **blank** for the TF row — inference
ran in a different framework, so wall time is not comparable across them; report each
framework's own throughput from its training/eval log if timing is needed.

Dataset: **adobe5kA**, test split, short-edge 1080.

| Model            | Framework | Init     | Budget | MSE | PSNR | SSIM |
|------------------|-----------|----------|--------|-----|------|------|
| CAN32+AN         | PyTorch   | Kaiming  | 500k   | …   | …    | …    |
| CAN32+AN (ref)   | TF (Keras)| identity | 500k   | …   | …    | …    |

### Documented confounds (held constant or noted, per the comparison design)
- **Metric code:** identical — both rows scored by `evaluate.py`/torchmetrics. The TF
  model's PNGs are never scored by TF's own PSNR/SSIM.
- **Channel order:** reconciled — `tf_reference.py` converts cv2 BGR→RGB to match the
  PyTorch PIL/RGB pipeline.
- **Train-time resize:** reconciled — same random 320–1440 short-side resize (the 32 GB
  5090 trains at the paper's full range, so no resolution gap).
- **Weight init:** **known difference** (TF identity vs PyTorch Kaiming) — documented,
  not equalized.
- **Adaptive-norm BN params:** **known difference** — the TF adaptive norm's BN has
  affine γ,β (→ 76,149 params); the PyTorch `AdaptiveBatchNorm2D` uses `affine=False`
  (→ ≈74,997). A ~1.5% parameter gap (1,152 = 64×2×9), documented not equalized.
- **Framework / hardware:** PyTorch vs TensorFlow — **record the exact TF version and GPU**
  actually used (modern TF on the Linux RTX 5090). Part of the "framework effects" being
  measured, not a confound to remove.
- **Eval input resampler (residual):** the TF export resizes inputs with cv2
  `INTER_LINEAR`; the PyTorch eval resizes with PIL bilinear. Targets are PIL-resized on
  both sides inside the shared metric path, so only the model *input* resampler differs.
  Minor; noted for completeness.
