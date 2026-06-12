# 💳 Credit Card Fraud Detection

An end-to-end Machine Learning project to detect fraudulent credit card transactions using the [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  
This project demonstrates **data preprocessing, model training, evaluation, and visualization** with an interactive Streamlit dashboard.

---

## 📂 Project Structure

Credit_Card_Fraud_Detection/
│── dataset/                # Kaggle dataset (creditcard.csv)
│── notebooks/              # Jupyter notebook for experiments
│── outputs/                # Model results & visualizations
│── src/                    # Source code (preprocessing, training, evaluation, visualization)
│── app.py                  # Streamlit dashboard
│── main.py                 # Command-line runner
│── README.md               # Project documentation


---

## ⚙️ Features
- Preprocessing with **StandardScaler** and **SMOTE** for class imbalance.
- Models: **Logistic Regression** & **Random Forest**.
- Evaluation metrics: Accuracy, Precision, Recall, F1-score.
- Visualizations: Class distribution, correlation heatmap, confusion matrices.
- Interactive **Streamlit dashboard** for dataset upload and live analysis.

---

## 📊 Results
- **Dataset Size:** 284,807 rows × 31 columns  
- **Best Model:** Logistic Regression  
- **Accuracy:** ~99.8%  
- **Recall (Fraud):** 100%  
- **Precision (Fraud):** ~94%  

---

## 🖼️ Screenshots

### 📌 Class Distribution

![Class Distribution](outputs/class_distribution.png)

### 📌 Amount by Class

![Amount by Class](outputs/amount_by_class.png)

### 📌 Correlation Heatmap

![Correlation Heatmap](outputs/correlation_heatmap.png)

### 📌 Confusion Matrix (Overall)

![Confusion Matrix](outputs/confusion_matrix.png)

### 📌 Confusion Matrix - Logistic Regression

![Confusion Matrix Logistic](outputs/confusion_matrix_logistic_regression.png)

### 📌 Confusion Matrix - Random Forest

![Confusion Matrix Random Forest](outputs/confusion_matrix_random_forest.png)

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection


2. Create virtual environment
bash
python -m venv .venv
.\.venv\Scripts\activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Run pipeline (CLI)
bash
python main.py

5. Run dashboard (Streamlit)
bash
streamlit run app.py

📌 Notes
Kaggle dataset (creditcard.csv) is not included due to size.

Download it from here and place inside dataset/ folder.

👨‍💻 Author
Ansh Pandey

GitHub: AnshPandey-85 (github.com in Bing)

LinkedIn: Ansh Pandey (linkedin.com in Bing)



