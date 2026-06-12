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

<img width="1536" height="1024" alt="Copilot_20260612_115736" src="https://github.com/user-attachments/assets/f56d704e-ffae-4e62-99ec-fc1db8d04803" />

### 📌 Class Distribution

<img width="1541" height="958" alt="image" src="https://github.com/user-attachments/assets/007f4675-4661-4e05-bcfd-f73dd48e8f4e" />

### 📌 Amount by Class

<img width="1532" height="847" alt="image" src="https://github.com/user-attachments/assets/ec410a15-8bab-4523-896a-c4c4880f43dc" />

### 📌 Correlation Heatmap

<img width="1293" height="967" alt="image" src="https://github.com/user-attachments/assets/2b9df693-c3b8-4e9f-82c8-97aa9ce2c299" />

### 📌 Confusion Matrix - Logistic Regression

<img width="1168" height="972" alt="image" src="https://github.com/user-attachments/assets/561675f9-01fb-4793-aa9d-0d1ed0013e11" />

### 📌 Confusion Matrix - Random Forest

<img width="1162" height="968" alt="image" src="https://github.com/user-attachments/assets/6c3771c2-471b-4d03-9eca-77c73c78a075" />

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



