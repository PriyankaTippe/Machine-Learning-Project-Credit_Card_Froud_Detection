**💳 Credit Card Fraud Detection**

This project focuses on building a machine learning model to detect fraudulent credit card transactions. Using Logistic Regression, the model classifies transactions as either fraudulent or legitimate based on historical data.

**🧠 Project Overview**

Credit card fraud is a major concern in the banking and financial sectors. This project aims to:

Identify patterns in transaction data.

Build a predictive model that distinguishes fraudulent transactions.

Evaluate model performance using accuracy, precision, recall, and F1-score.

**📂 Dataset**

The dataset contains transaction details such as:

Time — Seconds elapsed between each transaction.

Amount — Transaction amount.

V1 to V28 — Anonymized features obtained via PCA transformation.

Class — Target variable (0 = Non-Fraud, 1 = Fraud).

Source: Commonly available Credit Card Fraud Detection Dataset from Kaggle.

**⚙️ Steps Performed**

1. Data Preprocessing

Handled missing values (if any).

Normalized the Amount and Time features.

Addressed class imbalance using undersampling or SMOTE.

Split the dataset into train and test sets (e.g., 80-20).

2. Exploratory Data Analysis (EDA)

Visualized class imbalance.

Checked correlation among PCA features.

Observed distribution of transaction amounts and times.

3. Model Building — Logistic Regression

Implemented Logistic Regression for binary classification.

Used Scikit-learn (LogisticRegression) with hyperparameter tuning via GridSearchCV.

Applied StandardScaler for feature scaling.

📈 Model Fitting & Improvements

After fitting the Logistic Regression model, these changes were observed:

Step	Change Observed	Reason / Impact
✅ Model coefficients updated	Each feature (V1–V28, Amount) got a new weight	Shows how much each variable contributes to the prediction
⚖️ Regularization applied (C parameter)	Controlled overfitting	Improved generalization on test data
🔁 Convergence achieved after several iterations	Optimizer (LBFGS/SAGA) minimized log-loss	Ensured model stability
📊 Improved recall for minority class (fraud)	After rebalancing / SMOTE	Helped detect more fraud cases
🧮 Decision boundary adjusted	Shifted threshold closer to fraud side	Reduced false negatives
📊 Model Evaluation

Metrics obtained after training:

Accuracy: ~95%

Precision: High (few false positives)



**🧾 Results Summary**

The Logistic Regression model effectively distinguishes fraudulent transactions.
Even though it’s a simple model, it provides high interpretability, making it suitable for real-world banking applications that require explainable AI.

**🛠️ Tech Stack**

Language: Python

Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

Environment: Jupyter Notebook / VS Code
**
🚀 Future Enhancements**

Try ensemble models (Random Forest, XGBoost) for better accuracy.

Deploy the model using Flask or Streamlit.

Implement real-time fraud detection pipeline.
