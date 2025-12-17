# Dataset Instructions

## Datasets

This project uses customer support datasets for training the NLP triage system.

### 1. Customer Support on Twitter (Kaggle)

**URL**: https://www.kaggle.com/datasets/skywalker123/customer-support-on-twitter

**Description**: Real customer service conversations from Twitter

**Download Instructions**:
1. Go to the Kaggle dataset page
2. Click "Download" (requires Kaggle account)
3. Extract the CSV file
4. Place it in `data/raw/customer_support_twitter.csv`

Alternatively, use Kaggle API:
```bash
kaggle datasets download -d skywalker123/customer-support-on-twitter
unzip customer-support-on-twitter.zip -d data/raw/
```

### 2. Amazon Reviews (Kaggle)

**URL**: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

**Description**: Product reviews with ratings

**Download Instructions**:
1. Go to the Kaggle dataset page
2. Click "Download"
3. Extract the CSV file
4. Place it in `data/raw/amazon_reviews.csv`

Alternatively, use Kaggle API:
```bash
kaggle datasets download -d bittlingmayer/amazonreviews
unzip amazonreviews.zip -d data/raw/
```

## Sample Data

If you don't have access to the datasets, the system will automatically generate sample data for development and testing purposes. This allows you to:
- Test the entire pipeline
- Develop and debug models
- Demonstrate the system capabilities

To use sample data, simply run the training script without downloading the datasets:
```bash
python train.py
```

## Data Structure

After downloading, your `data/` directory should look like:
```
data/
├── raw/
│   ├── customer_support_twitter.csv
│   └── amazon_reviews.csv (optional)
└── processed/
    ├── processed_train.csv (generated)
    ├── processed_val.csv (generated)
    └── processed_test.csv (generated)
```

## Data Processing

The preprocessing pipeline will:
1. Load raw data
2. Clean and normalize text
3. Create category labels
4. Split into train/val/test sets
5. Save processed data

Run preprocessing:
```bash
python train.py
```

Or run just the preprocessing step in the EDA notebook:
```bash
jupyter notebook notebooks/01_eda.ipynb
```
