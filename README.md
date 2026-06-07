#### Downloading the Adobe5k Dataset

Download all 5,000 MIT-Adobe FiveK images with expert variant A, without
restricting to only train, validation, or test:

```
uv run python download_adobe5k.py --size 5000 --bypass-splits --experts a
```

This downloads the DNG files and expert A TIFF files into:

```
fivek_data/MITAboveFiveK/raw/
fivek_data/MITAboveFiveK/processed/tiff16_a/
```

#### Preparing the Processed Dataset

Convert the expert A TIFF files to PNG and apply the pencil operator:

```
uv run python data_preparation.py \
  --tif-source fivek_data/MITAboveFiveK/processed/tiff16_a \
  --raw-output dataset/adobe5k_processed/raw \
  --processed-output dataset/adobe5k_processed/processed \
  --operator pencil
```

This writes converted PNG inputs to:

```
dataset/adobe5k_processed/raw/
```

and processed pencil outputs to:

```
dataset/adobe5k_processed/processed/
```

The processed files are named like:

```
<image_name>_pencil.png
```


---
#### Other Variants (i.e. div2k):
Div2k can be extracted to `dataset/div2k/raw`
and can be processed via:
```
uv run python data_preparation.py \
  --skip-tif-conversion \
  --raw-output dataset/div2k/raw \
  --processed-output dataset/div2k/processed \
  --operator pencil
```



---

Useful Commands:

```
uv run python data_preparation.py --help
uv run python data_preparation.py --skip-tif-conversion
uv run python data_preparation.py --skip-processing
```
