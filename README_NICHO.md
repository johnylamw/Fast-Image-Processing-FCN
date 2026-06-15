# Running the TF reference reproduction (CAN32+AN)

End-to-end steps to train the authors' reference CAN (TensorFlow port, `tf_reference.py`)
on **adobe5kA**, then score it against our PyTorch `CAN32+AN` with the **same** metric code
(`evaluate.py` / torchmetrics). Training runs on a Linux RTX 5090; scoring runs back on the
Windows machine that holds the data + PyTorch checkpoint. The two machines only exchange
files. For the full design rationale and confounds, see [TF_REFERENCE_SETUP.md](TF_REFERENCE_SETUP.md).

Roles:
- **You (Windows):** publish the branch, send the dataset, score at the end.
- **Friend (Linux + RTX 5090):** set up TF, smoke-test, train 500k, export predictions.

---

## Part A — Publish the branch (you, Windows, repo root)

Selective `add` so only the reproduction code is committed (not the unrelated
flickr2k/div2k working-tree deletions, and not the CRLF-only split change):

```bat
git checkout -b reproduction-tf-reference
git add .gitignore tf_reference.py make_toy_dataset.py TF_REFERENCE_SETUP.md README_NICHO.md evaluate.py download_adobe5k.py
git commit -m "Add TF reference (CAN32+AN) reproduction + folder-scoring eval"
git push -u origin reproduction-tf-reference
```

Notes:
- `datasets/` is git-ignored — the ~30 GB of Adobe data is **not** pushed (transferred in Part B).
- `third_party/` and `tf_model_runs/` are git-ignored — run outputs and the cloned FiveK
  loader never get committed.
- `data_splits/adobe5kA/` is already tracked, so the friend gets the splits from the clone.

---

## Part B — Send the data to the friend (~30 GB, separate from git)

The friend needs `datasets/adobe5kA/{inputs,targets}` at the repo root. Transfer by any
means; e.g. from the Windows machine:

```bash
rsync -av --progress datasets/adobe5kA/ FRIEND@HOST:/path/to/Fast-Image-Processing-FCN/datasets/adobe5kA/
```

(or zip + a cloud drive — it just has to end up at `<repo>/datasets/adobe5kA/`).

---

## Part C — Friend's EXACT steps (Linux + RTX 5090)

```bash
# 1. clone the branch
git clone -b reproduction-tf-reference https://github.com/johnylamw/Fast-Image-Processing-FCN.git
cd Fast-Image-Processing-FCN

# 2. drop the transferred data in place, then verify counts
ls datasets/adobe5kA/inputs  | wc -l    # expect 5000
ls datasets/adobe5kA/targets | wc -l    # expect 5000

# 3. modern TF env (Blackwell/sm_120 needs CUDA 12.x — NOT TF 2.10)
conda create -n tfref python=3.11 -y
conda activate tfref
pip install "tensorflow[and-cuda]" opencv-python numpy
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"   # MUST list a GPU

# 4. GPU SMOKE TEST FIRST (validates tf_reference.py on modern TF / Keras 3)
python make_toy_dataset.py --root datasets/_toy --num 6 --test 2
python tf_reference.py train  --dataset datasets/_toy --run-name smoke --iterations 200 --min-res 96 --max-res 160 --ckpt-every 100
python tf_reference.py export  --dataset datasets/_toy --run-name smoke --short-edge 128
#    PASS = prints "Model params: 76149" and "Wrote 2 predictions ...". If it errors, STOP and report the traceback.
rm -rf datasets/_toy data_splits/_toy tf_model_runs/_toy

# 5. THE REAL RUN — full 500k at the paper's resolution (fits in 32 GB)
python tf_reference.py train  --dataset datasets/adobe5kA --run-name CAN32+AN_tf --iterations 500000 --ckpt-every 10000 --resume

# 6. export test-split predictions
python tf_reference.py export --dataset datasets/adobe5kA --run-name CAN32+AN_tf --short-edge 1080

# 7. send this folder back:
#    tf_model_runs/adobe5kA/CAN32+AN_tf/predictions/      (~2500 PNGs, small)
```

---

## Part D — Score both models + build the table (you, Windows, uv env)

```bat
uv run python evaluate.py datasets/adobe5kA model_runs/adobe5kA/CAN32+AN/CAN32+AN_final.pt --short-edge 1080
uv run python evaluate.py datasets/adobe5kA --predictions tf_model_runs/adobe5kA/CAN32+AN_tf/predictions --pred-label CAN32+AN_tf
```

Outputs (same CSV schema → merge into the comparison table):
- `output/evaluate/adobe5kA_shortedge1080_eval.csv`    (PyTorch)
- `output/evaluate/adobe5kA_CAN32+AN_tf_predeval.csv`  (TF reference)

---

## Two things to stress to the friend

1. **Do not skip the smoke test (Part C step 4).** It is the gate that proves
   `tf_reference.py` runs on modern TF / Keras 3 (it was only executed on TF 2.10 during
   development). If step 4 errors, fix that before burning time on the 500k run.
2. **If step 3 lists no GPU**, the installed `tensorflow[and-cuda]` is too old for Blackwell
   (sm_120) — upgrade TF (or use a nightly) until the GPU lists and the smoke test passes.
