#### Downloading the Raw Adobe5k Dataset

Download all 5,000 MIT-Adobe FiveK images with expert variant A, without
restricting to only train, validation, or test:

```
uv run python download_adobe5k.py --size 5000 --bypass-splits --experts a
```

This downloads the DNG files and expert A TIFF files into:

```
fivek_data/MITAboveFiveK/raw/
fivek_data/MITAboveFiveK/targets/tiff16_a/
```

#### Preparing the Processed Dataset

Convert the expert A TIFF files to PNG and apply the pencil operator:

```
uv run python data_preparation.py \
  --tif-source fivek_data/MITAboveFiveK/processed/tiff16_a \
  --raw-output dataset/adobe5k_processed/inputs \
  --processed-output dataset/adobe5k_processed/targets \
  --operator pencil
```

This writes converted PNG inputs to:

```
dataset/adobe5k_processed/inputs/
```

and processed pencil outputs to:

```
dataset/adobe5k_processed/targets/
```

The processe target files are named like:

```
<image_name>_pencil.png
```

---
# ALTERNATIVE: Download jpg compressed Adobe5k here
https://www.kaggle.com/datasets/weipengzhang/adobe-fivek

---
### Adobe5k w/o the raws or tifs + other variants (i.e., div2k)
```
uv run python data_preparation.py \
  --skip-tif-conversion \
  --raw-output datasets/adobe5kA/inputs \
  --processed-output datasets/adobe5kA/targets \
  --operator pencil
  ```


---
Useful Commands:

```
uv run python data_preparation.py --help
uv run python data_preparation.py --skip-tif-conversion
uv run python data_preparation.py --skip-processing
```

---
# Model Training
```
uv run train.py \
--dataset datasets/adobe5kA \
--model CAN24+AN \
--iterations 500_000 \
--outputs model_runs \
--splits data_splits
```

Minimal-ish
```
uv run train.py \
--dataset datasets/adobe5kA \
--model CAN24+AN \
--iterations 500_000 
```


---
# Demo - 3 Samples on Div2k Test
```
uv run python demo.py \
  datasets/div2k \
  model_runs/adobe5kA/CAN24+AN/CAN24+AN_final.pt \
  model_runs/adobe5kA/CAN32+AN/CAN32+AN_final.pt \
  model_runs/div2k/CAN32+AN/CAN32+AN_final.pt \
  model_runs/flickr2k/CAN32+AN/CAN32+AN_final.pt \
  --num-samples 3
```

---
# Evaluate Model(s) on Div2k Test (MSE, SSIM, PSNR)
```
uv run python evaluate.py \
  datasets/div2k \
  model_runs/adobe5kA/CAN24+AN/CAN24+AN_final.pt \
  model_runs/adobe5kA/CAN32+AN/CAN32+AN_final.pt \
  model_runs/div2k/CAN32+AN/CAN32+AN_final.pt \
  model_runs/flickr2k/CAN32+AN/CAN32+AN_final.pt \
  --short-edge 1080
```

NOTE: Short-edge = the resizing of the image. The paper's experiment defaults to 1080 for eval.