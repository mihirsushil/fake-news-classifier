# Liar Predictor

A machine learning model that classifies statements as either true or false using natural language processing and logistic regression.

## Current Status

⚠️ **Note**: This repository contains the code but the CSV data files are missing. You'll need to provide your own dataset or obtain the original data files to run the model.

## Features

- **Text Preprocessing**: Converts statements to lowercase and removes extra whitespace
- **TF-IDF Vectorization**: Uses n-gram features (1-2 words) with minimum document frequency filtering
- **Binary Classification**: Converts multi-class labels into binary (Fake: 0, Real: 1)
- **Stratified Sampling**: Ensures balanced class distribution across train/validation/test splits
- **Model Pipeline**: Combines vectorization and classification in a single scikit-learn pipeline
- **Performance Evaluation**: Provides detailed classification reports with precision, recall, and F1-scores

## Requirements

```bash
pip install pandas scikit-learn numpy
```

## Usage

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd liar_predictor
   ```

2. **Add your data files**:
   - Place your CSV files in the project directory
   - Ensure they have the expected column names (see Data Format section)

3. **Run the model**:
   ```bash
   python3 model_predictor.py
   ```

## Data Format

The model expects CSV files with these columns:
- `id`: Unique identifier
- `label`: Original multi-class labels (0-5)
- `statement`: The text to be classified
- `true_counts`, `mostly_true_counts`, `half_true_counts`, `mostly_false_counts`, `false_counts`, `pants_on_fire_counts`: Various truth metrics

## Model Architecture

- **Text Vectorization**: TF-IDF with n-gram range (1,2) and min_df=3
- **Classifier**: Logistic Regression with balanced class weights
- **Training Strategy**: 
  - Initial training on 80% of data
  - Final training on 90% of data (train + validation)
  - Evaluation on held-out test set

## File Structure

```
liar_predictor/
├── model_predictor.py    # Main model implementation
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation
```

## Missing Data Files

The following CSV files are required but not included:
- `train.csv` - Training dataset
- `valid.csv` - Validation dataset  
- `test.csv` - Test dataset

## Future Improvements

- Experiment with different ML algorithms
- Incorporate additional features from the count columns
- Implement cross-validation for more robust evaluation
- Add hyperparameter tuning capabilities
- Create a web interface for real-time predictions

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
