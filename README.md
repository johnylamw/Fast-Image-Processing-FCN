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
--output runs
```