<centering>
<h1 style='text-align: center'>Reproduction Blog: Fast Image Processing with Fully-Convolutional Networks</h1>
</centering><centering>
    
<p style='text-align:center; font-weight:200; font-size: 22px; margin-bottom: 0'>
    Robin Kruijf<sup >1</sup>, Ocean Wang<sup>1</sup>, Johnny Wu<sup>1</sup>, Nicholas Wu<sup>1</sup>
    </p>
    <p style='text-align:center; font-weight:500; font-size: 17px; margin-bottom: 0'>
    DSAIT 4025 Fundamental Research in Machine and Deep Learning
    </p>
    <p style='text-align:center; font-weight:100; font-size: 15px; margin-bottom: 0'>
    <sup>1</sup>TU Delft (Group 20)
    </p>
<p style='text-align:center; font-weight:100; font-size: 15px'>
    13 June 2026
    </p>
    
</centering>



## 1. Introduction and Motivation

Traditional image processing pipelines often rely on hand-designed operators for performing tone mapping, stylization, smoothing and detail enhancement. However, they are often computationally expensive and difficult to run in real time depending on on the transformation algorithm and the resolution of the image. Chen et al. address this by training fully-convolutional networks to approximate these operators with a single forward pass. 

This paper is interesting to reproduce because it claims that learned image-processing approximators can match expensive traditional image-processing operators while being much faster at inferencing. 

Our reproduction implements the Context Aggregation Network (CAN) architecture in PyTorch, including adaptive normalization, random-resolution training, and evaluation with MSE, PSNR, SSIM, and runtime. We also evaluate cross-dataset generalizability, model variants, and training progression over checkpoints.

## 2. Reproduction Scope

The original paper proposes using fully-convolutional networks to approximate image-processing operators. 
CAN uses dilated convolutions to increase receptive field without reducing image resolution. The paper also introduces adaptive normalization, which combines the identity mapping with batch normalization through learned parameters.

We do not reproduce the full set of ten image-processing operators from the paper. Instead, we focus on one target operator and evaluate whether the model architecture setup behave similarly to the original paper.

### 2.1 What We Faithfully Reproduced
- Context Aggregation Network architecture w/ dilated convolution layers
- Adaptive normalization
- Random-resolution training (MSE w/ Adam) for 500k iterations per model
- Same-dataset evaluation and splits on MIT-Adobe 5k with 2.5k/2.5k splits
- Cross-dataset generalization study
- Cross-resolution generalization study
- Architecture ablation across our main CAN variants: `CAN24+AN` and `CAN32+AN`
- Training on Pencil Sketch Image Pairs


### 2.2 Changes and Extensions
- Reimplemented the model and training pipeline in PyTorch (orignal: Tensorflow)
- For the cross-dataset study, we did not evaluate on the RAISE dataset used in the paper. Instead, we used Flickr2K and Div2K as alternative datasets for cross-dataset generalization.
- Added adaptive-dilation variants `CAN24+AND` and `CAN32+AND`
- Added multi-checkpoint evaluation for training progression
- Added qualitative demos comparing input, prediction, and ground truth.


### 2.3 What We Omitted
- Baseline models such as FCN-8s and encoder-decoder networks.
- Additional CAN variants from the paper such as `CAN32` and `CAN32+BN`
- The nine other image-processing operators evaluated in the paper
- Extension: Parameterized network variant which learns the image-operator with random sampling of parameters for image pairs of the operator in question. (i.e., pencil sketch $\sigma_s$)
- Extension: Single-network variant for learning multiple operators in one network.

### 2.4 Reproducibility Criteria
Our project satisfies the following reproducibility criteria:
- **Replicated:** We reimplemented the core CAN model and training setup from the paper description. The original CAN variant code was not released. Only the single-network and parameterized-network code was available.
- **New code variant**: We implemented the reproduction in PyTorch instead of TensorFlow.
- **New data**: We evaluated and trained the model on multiple datasets such as Flickr2k and Div2k.
- **New algorithm variant**: We introduced adaptive-dilation CAN variants.
- **Ablation study**: We compared different CAN variants and evaluated training progression over checkpoints.


## 3. Implementation

### 3.1 Model Variants

We implemented four CAN-based model variants. The first two are based on the architectures proposed in the original paper, while the last two are our adaptive-dilation extensions.

| Model | Depth | Width | Normalization | Notes |
|---|---:|---:|---:|---:|
| `CAN24+AN` | 9 | 24 | Adaptive normalization | Paper's primary model |
| `CAN32+AN` | 10 | 32 | Adaptive normalization | Larger paper model |
| `CAN24+AND` | 9 | 24 | Adaptive normalization | Our adaptive-dilation variant |
| `CAN32+AND` | 10 | 32 | Adaptive normalization | Larger adaptive-dilation variant |

### 3.2 Trained Models

| Model | Architecture | Training Dataset | Training Samples | Purpose |
|---|---:|---:|---:|---:|
| Adobe5kA `CAN24+AN` | `CAN24+AN` | Adobe5kA | 2500 | Main reproduction |
| Adobe5kA `CAN32+AN` | `CAN32+AN` | Adobe5kA | 2500 | Larger paper model |
| Adobe5kA `CAN24+AND` | `CAN24+AND` | Adobe5kA | 2500 | Adaptive-dilation variant |
| Adobe5kA `CAN32+AND` | `CAN32+AND` | Adobe5kA | 2500 | Larger adaptive-dilation variant |
| Flickr2K `CAN24+AN` | `CAN24+AN` | Flickr2K | 1325 | Cross-dataset generalization |
| Div2K `CAN24+AN` | `CAN24+AN` | Div2K | 450 | Cross-dataset generalization |


### 3.3 Datasets and Splits

We used paired image datasets where each input image has a corresponding pencil-sketch. In our case, the corresponding image is generated using `cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)`.

The network learns the mapping from the original image to the processed pencil-sketch output.

We used three datasets:
| Dataset | Total Images | Training Samples | Purpose |
|---|---:|---:|---|
| Adobe5kA | 5000 | 2500 | Main reproduction dataset |
| Flickr2K | 2650 | 1325 | Cross-dataset generalization |
| Div2K | 900 | 450 | Cross-dataset generalization |

For Adobe5kA, we follow the paper's 2.5k / 2.5k split. For Flickr2K and Div2K, we use fixed 50/50 train-test splits. The split files are stored in `data_splits/`, which ensures that training, evaluation, and demo generation all use the same held-out test images and for reproducibility purposes.

One difference from the original paper is that we did not use the RAISE dataset. Instead, Flickr2K and Div2K are used as alternative datasets for testing cross-dataset generalization.

### 3.4 Training Setup

All models are trained using MSE loss and the Adam optimizer. 
Following the original paper, training is iteration-based rather than epoch-based where only one image-pair is randomly sampled and trained per iterations. Each final model is trained for 500k iterations total.

During each training iteration, the sampled image is resized to a randomly sampled resolution between `320p - 1440p` while preserving aspect ratio. This follows the paper's approach on training the fully-convolutional model to learn across different input scales.

Similarly, we use a batch size of 1. This matches the random-resolution setup more naturally, since each training sample can also have a different spatial resolution. 

Checkpoints are saved every 10k iterations, and selected checkpoints are later used to evaluate training progression over time.

All training outputs are stored in:

```
model_runs/<dataset_name>/<model_name>
```

### 3.5 Evaluation Setup
Evaluation is performed on the fixed held-out test split. 
For the main qualitative result, we followed the paper's evaluation setup where images are resized to to 1080p resolution for the main quantitative results.

We report four metrics:

| Metric | Purpose |
|---|---|
| MSE | Pixel-wise reconstruction error |
| PSNR | Approximation accuracy in dB |
| SSIM | Structural similarity to the target image |
| Time | Average inference time per image |

- MSE, PSNR, and SSIM are computed using `torchmetrics`.
- Time is measured using `time.perf_counter()`

Evaluation results across different checkpoints are saved as CSV files in:
```output/evaluate/```


## 4. Experiments and Results

### 4.1 Same-Dataset Evaluation

We first evaluate each final model on the test split from the same dataset it was trained on. This is the closest setting to the main evaluation in the original paper because the model is tested on held-out images from the same image distribution.

For this experiment, images are resized to 1080p during evaluation, following the paper's evaluation protocol. Since our reproduction focuses only on the pencil-sketch operator, this table is analogous to the paper's Table 1 but capturing only one operator rather than the average across all ten.
#### Adobe5kA

| Model | MSE | PSNR (dB) | SSIM | Time (ms) | # Params |
|---|---:|---:|---:|---:|---:|
| `CAN24+AN` | TODO | TODO | TODO | TODO | TODO |
| `CAN32+AN` | TODO | TODO | TODO | TODO | TODO |
| `CAN24+AND` | TODO | TODO | TODO | TODO | TODO |
| `CAN32+AND` | TODO | TODO | TODO | TODO | TODO |

#### Additional Same-Dataset Results

| Dataset | Model | MSE | PSNR (dB) | SSIM | Time (ms) | # Params |
|---|---|---:|---:|---:|---:|---:|
| Flickr2K | TODO | TODO | TODO | TODO | TODO | TODO |
| Div2K | TODO | TODO | TODO | TODO | TODO | TODO |



This experiment serves as our main reproduction result.

- Higher PSNR and SSIM indicate that the model more closely approximates the pencil-sketch target.
- Lower MSE indicates lower pixel-level reconstruction error.
- Time measures average inference time per image on our hardware.

### 4.2 Cross-Dataset Generalization

We then evaluate whether a model trained on one dataset can generalize to another dataset.

This experiment follows the cross-dataset generalization study in the original paper, where models trained on MIT-Adobe and RAISE were evaluated across both datasets. In our reproduction, we did not use RAISE. Instead, we use Flickr2K and Div2K as alternative datasets for testing generalization.

To keep the comparison focused on dataset shift, we compare the same architecture `CAN32+AN` across training datasets.

| Test Dataset | Training Dataset | Model | MSE | PSNR (dB) | SSIM | Time (ms) |
|---|---|---|---:|---:|---:|---:|
| Adobe5kA | Flickr2K | TODO | TODO | TODO | TODO | TODO |
| Flickr2K | Adobe5kA | TODO | TODO | TODO | TODO | TODO |
| Adobe5kA | Div2K | TODO | TODO | TODO | TODO | TODO |
| Div2K | Adobe5kA | TODO | TODO | TODO | TODO | TODO |

This experiment tests whether the learned pencil-sketch approximation depends strongly on the training dataset. 

**TODO: explanation**

### 4.3 Cross-Resolution Generalization

The paper trains on randomly sampled image resolutions and evaluates the main results at 1080p. Since our training setup also uses random-resolution sampling, we evaluate whether the model behaves consistently across different evaluation resolutions.

For this experiment, we evaluate the same final checkpoint at multiple short-edge resolutions.

| Resolution | Model | MSE | PSNR (dB) | SSIM | Time (ms) |
|---:|---|---:|---:|---:|---:|
| TODO | TODO | TODO | TODO | TODO | TODO |

This experiment checks whether the model is sensitive to evaluation scale. We expect inference time to increase with resolution, while quality metrics may change depending on how well the learned operator transfers across image scales.

### 4.4 Architecture Ablation

We compare the different CAN variants to understand how architecture choices affect approximation quality and runtime. The two paper-based variants are `CAN24+AN` and `CAN32+AN`. The `CAN24+AND` and `CAN32+AND` models are our adaptive-dilation extensions.

All models are evaluated on the Adobe5kA test split so that the comparison focuses on architecture rather than dataset differences.

| Model | MSE | PSNR (dB) | SSIM | Time (ms) | # Params |
|---|---:|---:|---:|---:|---:|
| `CAN24+AN` | TODO | TODO | TODO | TODO | TODO |
| `CAN32+AN` | TODO | TODO | TODO | TODO | TODO |
| `CAN24+AND` | TODO | TODO | TODO | TODO | TODO |
| `CAN32+AND` | TODO | TODO | TODO | TODO | TODO |

This ablation lets us compare the effect of model capacity and receptive-field design. In particular, we compare whether the larger `CAN32+AN` model improves over `CAN24+AN`, and whether our adaptive-dilation variants improve or degrade performance. 

**TODO: talk about results**

### 4.5 Training Progression Over Checkpoints

Since we save intermediate checkpoints during training, we also evaluate how the model improves over time. This gives a more detailed view of convergence and whether the full 500k training iterations are necessary in our setting.

We evaluate selected checkpoints at:

- 10k iterations
- 20k iterations
- 50k iterations
- 100k iterations
- 250k iterations
- 500k iterations

| Iteration | Model | MSE | PSNR (dB) | SSIM | Time (ms) |
|---:|---|---:|---:|---:|---:|
| 10000 | TODO | TODO | TODO | TODO | TODO |
| 20000 | TODO | TODO | TODO | TODO | TODO |
| 50000 | TODO | TODO | TODO | TODO | TODO |
| 100000 | TODO | TODO | TODO | TODO | TODO |
| 250000 | TODO | TODO | TODO | TODO | TODO |
| 500000 | TODO | TODO | TODO | TODO | TODO |

**TODO: ADD DEMO IMAGES OF MODEL UNDER DIFFERENT ITERATIONS**
**TODO: EXPLANATION. Answer the question, did we really need to train 500k iterations?**

### 4.6 Training Log Analysis

Additionally, we analyze the training logs saved during optimization. These logs are used as supporting evidence in our study. The main claims are based on held-out test metrics, not training loss.

The plot shows training loss over iterations. This helps verify that the model is learning during training and that optimization is stable.

**TODO: Insert training loss vs iteration plot**

**TODO: EXPLANATION**

## 5. Discussion (WIP)

### 5.1 Did We Uphold the Paper's Main Claim?

### 5.2 Main Findings

### 5.3 Limitations

Although Section 2 defines the reproduction scope, several limitations affect how directly our results can be compared to the original paper.

- **Exact implementation details.** The main CAN variant code was not released by the original authors. More specifically, only the single-network and parameterized-network code was available. Our implementation is therefore based on the paper description and reimplemented in PyTorch. Therefore, small differences in initialization, padding behavior, optimizer details, preprocessing, and framework-specific layer behavior may affect exact numerical reproducibility.

- **Single operator.** The original paper evaluates ten image-processing operators, while we only reproduce the pencil-sketch operator. This means our results test the CAN architecture in one setting, but not across the full range of operators studied in the paper. Additionally, we use OpenCV's `cv2.pencilSketch()` rather than the exact pencil-sketch implementation used by the original authors.

- **Different cross-dataset setup.** The paper uses MIT-Adobe and RAISE for cross-dataset generalization. We use Adobe5kA, Flickr2K, and Div2K instead.

- **No baseline methods.** We do not reproduce the non-CAN baselines from the paper, such as FCN-8s and the encoder-decoder models. Our comparisons are therefore mainly between our own CAN variants.

- **Unequal dataset sizes.** Adobe5kA, Flickr2K, and Div2K have different numbers of training images. This means cross-dataset performance may reflect both dataset distribution and dataset size.

- **Runtime comparison.** Runtime depends on hardware, so our `Time (ms)` values are only directly comparable within our own experiments.


## 6. Conclusion

## Appendix

### A. Reproducibility Details

### B. Additional Results

### C. Additional Demo Images

### D. Team Contributions