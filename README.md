# SemEval 2022 Task 4 - Patronising and Condescending Language Detection

Binary classification of Patronising and Condescending Language (PCL) from news text, implemented as a weight-diversity ensemble of four fine-tuned RoBERTa-base models.

**Dev set F1:** 0.6364 | **Baseline F1:** 0.491 | **Decision threshold:** 0.44

---

## Repository Structure

```text
.
|-- data/
|   |-- dontpatronizeme_pcl.tsv              # Full annotated dataset (10,468 rows)
|   |-- train_semeval_parids-labels.csv      # Official train split par_ids
|   |-- dev_semeval_parids-labels.csv        # Official dev split par_ids
|   `-- task4_test.tsv                       # Official test set (no labels)
|
|-- Models/
|   |-- BaselineModel/                       # RoBERTa-base baseline (1 epoch, 1:2 downsampling)
|   `-- BestModel/
|       |-- threshold.json                   # Optimal decision threshold (0.44)
|       |-- seed42_w3.0/                     # Ensemble member 1
|       |-- seed59_w5.0/                     # Ensemble member 2
|       |-- seed123_w7.0/                    # Ensemble member 3
|       `-- seed456_w9.5/                    # Ensemble member 4
|
|-- 1_EDA.ipynb                              # Exploratory Data Analysis
|-- 2_training.ipynb                         # Model training (baseline + BestModel)
|-- 3_inference.ipynb                        # Evaluation and error analysis
|
|-- dev.txt                                  # Dev set predictions (2,094 lines)
`-- test.txt                                 # Test set predictions (3,832 lines)
```

---

## Approach

The proposed approach is a **weight-diversity ensemble** of four RoBERTa-base models, each trained independently on the full training set with a different random seed and positive-class weight. The positive-class weights (3.0, 5.0, 7.0, 9.5) span a range of precision-recall operating points, from conservative to aggressive. At inference time, softmax probabilities from all four models are averaged and a single decision threshold (0.44), tuned on the dev set, is applied.

Three key design decisions differentiate this from the baseline:

1. **Full training data with class-weighted loss** rather than 1:2 downsampling
2. **Weight-diversity ensemble** to reduce variance across operating points
3. **Dev-set threshold tuning** rather than a fixed threshold of 0.5

---

## Reproducing Results

### Requirements

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for `2_training.ipynb`. `3_inference.ipynb` runs on CPU or MPS (Apple Silicon).

### Data

The `data/` directory is already included in the repository. No additional download is required.

### Step 1 - Exploratory Data Analysis

Open and run `1_EDA.ipynb`. This notebook covers class distribution, token length analysis, lexical analysis, semantic exploration, and noise identification. No GPU required.

### Step 2 - Training

The `Models/` directory already contains saved model weights for both the baseline and all four ensemble members. `2_training.ipynb` will load these automatically without retraining.

To retrain from scratch, delete the contents of `Models/BaselineModel/` and `Models/BestModel/` and run `2_training.ipynb`. Each model will be retrained and saved back to the same directories.

Shared hyperparameters for BestModel ensemble members:

- `lr = 2e-5`
- `warmup_ratio = 0.1`
- `batch_size = 16`
- `max_length = 256`
- `num_epochs = 5`
- `load_best_model_at_end = True`

### Step 3 - Evaluation

Open and run `3_inference.ipynb`. This notebook is **fully standalone** - it loads all models and the decision threshold directly from disk with no dependency on `2_training.ipynb` runtime variables. It produces:

- `dev.txt` and `test.txt` with one binary prediction per line (0 = Non-PCL, 1 = PCL)
- Error analysis including false negative and false positive breakdowns by keyword
- Ensemble ablation study (F1 as models are added)
- Weight sensitivity analysis (precision-recall tradeoff per model)

---

## Prediction Files

Both files contain one prediction per line (0 or 1).

- `dev.txt` - 2,094 lines, official dev set
- `test.txt` - 3,832 lines, official test set

---

## Results

| Model | Dev F1 | Precision | Recall |
|-------|--------|-----------|--------|
| Baseline (threshold=0.5) | 0.491 | 0.37 | 0.74 |
| BestModel (threshold=0.44) | 0.636 | 0.640 | 0.633 |
