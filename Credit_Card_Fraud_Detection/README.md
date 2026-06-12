# Credit Card Fraud Detection

## Objective

This project is a complete end-to-end machine learning solution for detecting fraudulent credit card transactions. It includes data analysis, preprocessing, feature scaling, class imbalance handling with SMOTE, model training, model evaluation, visualization, and a Streamlit web app.

## Dataset Information

Dataset: Kaggle Credit Card Fraud Detection  
Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The original dataset contains anonymized credit card transactions. Most columns are PCA-transformed features named `V1` to `V28`. The key columns are:

- `Time`: Time elapsed from the first transaction.
- `Amount`: Transaction amount.
- `Class`: Target column. `0` means legitimate transaction and `1` means fraud.

Place the real Kaggle file here:

```text
Credit_Card_Fraud_Detection/dataset/creditcard.csv
```

If the Kaggle CSV is not present, the project automatically uses generated demo data so that `main.py` and `app.py` can still run for demonstration.

## Technologies Used

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- imbalanced-learn
- Streamlit
- Jupyter Notebook

## Project Structure

```text
Credit_Card_Fraud_Detection/
|
├── dataset/
│   └── creditcard.csv
|
├── notebooks/
│   └── fraud_detection.ipynb
|
├── src/
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── model_training.py
│   ├── sample_data.py
│   └── visualization.py
|
├── outputs/
│   ├── confusion_matrix.png
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── amount_by_class.png
│   └── model_results.txt
|
├── app.py
├── main.py
├── run_app.bat
├── requirements.txt
└── README.md
```

## Data Preprocessing Steps

1. Load the Kaggle dataset or generated demo data.
2. Check dataset shape, missing values, duplicate rows, and class counts.
3. Fill missing numeric values with median values.
4. Fill missing non-numeric values with mode values.
5. Remove duplicate rows.
6. Split features and target column.
7. Apply stratified train-test splitting.
8. Scale features using `StandardScaler`.
9. Apply SMOTE only on the training data to avoid data leakage.

## Handling Class Imbalance

Fraud transactions are rare compared with legitimate transactions. This project uses SMOTE to oversample the minority fraud class in the training set. The test set remains untouched so evaluation stays realistic.

## Model Training

The project trains and compares:

- Logistic Regression
- Random Forest Classifier

## Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

The best model is selected using F1-Score because fraud detection needs a balance between catching fraud cases and avoiding false alarms.

## Visualizations

The project generates:

- Class distribution chart
- Correlation heatmap
- Transaction amount by class chart
- Confusion matrix for each model
- Best model confusion matrix

All visual outputs are saved in the `outputs/` folder.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete ML pipeline:

```bash
python main.py
```

Run strictly with the real Kaggle CSV and fail if it is missing:

```bash
python main.py --no-demo
```

Run the Streamlit app:

```bash
streamlit run app.py
```

On Windows, you can also double-click:

```text
run_app.bat
```

## Results

After running `main.py`, the terminal displays the model comparison table and best-performing model. Detailed results are saved in:

```text
outputs/model_results.txt
```

Generated plots are saved in:

```text
outputs/
```

## Conclusion

This project follows a complete internship-level machine learning workflow suitable for GitHub and LinkedIn portfolio submission. It includes clean modular code, a notebook, visualizations, model comparison, and an interactive Streamlit app for easy demonstration.
