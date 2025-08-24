# Liar Predictor

A machine learning model that classifies statements as either true or false using natural language processing and logistic regression.

## Overview

This project implements a binary classification system for detecting false statements. It uses TF-IDF vectorization to convert text statements into numerical features and trains a logistic regression model to predict whether a statement is likely to be true or false.

## Features

- **Text Preprocessing**: Converts statements to lowercase and removes extra whitespace
- **TF-IDF Vectorization**: Uses n-gram features (1-2 words) with minimum document frequency filtering
- **Binary Classification**: Converts multi-class labels into binary (Fake: 0, Real: 1)
- **Stratified Sampling**: Ensures balanced class distribution across train/validation/test splits
- **Model Pipeline**: Combines vectorization and classification in a single scikit-learn pipeline
- **Performance Evaluation**: Provides detailed classification reports with precision, recall, and F1-scores

## Dataset

The model is trained on a dataset containing:
- **Statement**: The text to be classified
- **Label**: Original multi-class labels (0-5)
- **Truth Counts**: Various metrics related to statement verification
- **Binary Label**: Converted to 0 (Fake) or 1 (Real)

## Requirements

```bash
pip install pandas scikit-learn
```

## Usage

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd liar_predictor
   ```

2. **Run the model**:
   ```bash
   python3 model_predictor.py
   ```

## Model Architecture

- **Text Vectorization**: TF-IDF with n-gram range (1,2) and min_df=3
- **Classifier**: Logistic Regression with balanced class weights
- **Training Strategy**: 
  - Initial training on 80% of data
  - Final training on 90% of data (train + validation)
  - Evaluation on held-out test set

## Performance

The model achieves approximately **62-63% accuracy** on the test set, with:
- **Fake Detection**: ~70% precision, ~61% recall
- **Real Detection**: ~55% precision, ~67% recall

## File Structure

```
liar_predictor/
├── model_predictor.py    # Main model implementation
├── train.csv            # Training dataset
├── valid.csv            # Validation dataset  
├── test.csv             # Test dataset
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

## Future Improvements

- Experiment with different ML algorithms (Random Forest, XGBoost, Neural Networks)
- Incorporate additional features from the count columns
- Implement cross-validation for more robust evaluation
- Add hyperparameter tuning capabilities
- Create a web interface for real-time predictions

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
