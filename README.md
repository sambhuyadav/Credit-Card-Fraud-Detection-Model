# 📄 Project Documentation: Credit Card Fraud Detection  

## 1. Project Overview  
This project focuses on detecting fraudulent credit card transactions. Fraud detection is a highly critical task in financial systems, where even a small number of fraudulent activities can cause significant financial losses.  

This project applies **machine learning techniques** to detect fraudulent credit card transactions. The dataset used is highly imbalanced, containing **492 frauds out of 284,807 transactions** (~0.172%). The goal is to build a balanced dataset, train classification models, and evaluate their performance with proper metrics.  

---

## 2. Dataset Description  
- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- **Size**: 284,807 transactions  
- **Fraud Cases**: 492 (~0.172%)  
- **Features**:  
  - `V1–V28`: PCA-transformed numerical features  
  - `Time`: Seconds elapsed since first transaction  
  - `Amount`: Transaction amount  
  - `Class`: Target variable (0 → Legitimate, 1 → Fraud)  

---

## 3. Data Exploration  
- Used **Pandas** to load and inspect dataset (`head()`, `info()`).  
- Checked **class distribution** → revealed extreme imbalance.  
- Analyzed **`Amount` distribution** for both legitimate and fraudulent transactions.  
- Computed **mean feature values grouped by Class** to identify differences.  

---

## 4. Handling Class Imbalance  
Since fraud cases are rare:  
- Extracted **492 random legitimate transactions** to match fraud cases.  
- Created a **balanced dataset** (`492 legitimate + 492 fraud`).  
- Combined using `pd.concat()` for model training.  

---

## 5. Data Preprocessing  
- Features (`Time`, `Amount`, `V1–V28`) stored in `X`.  
- Target (`Class`) stored in `Y`.  
- Used `train_test_split` with `stratify=Y` to maintain class distribution.  

```python
from sklearn.model_selection import train_test_split

X = Balanced_Credit_Card_Data.drop(columns='Class', axis=1)
Y = Balanced_Credit_Card_Data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)
```

## 6. Model Training
-	Implemented Logistic Regression (sklearn.linear_model.LogisticRegression).
-	Trained on balanced dataset (undersampled).
-	Evaluated with accuracy score.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_pred = model.predict(X_train)
train_acc = accuracy_score(X_train_pred, Y_train)

X_test_pred = model.predict(X_test)
test_acc = accuracy_score(X_test_pred, Y_test)
```

## 7. Results
- Training Accuracy and Testing Accuracy were computed.
-	Since dataset is balanced artificially, accuracy is not misleading here.
-	However, for the real-world dataset, accuracy would be misleading → recommend Precision-Recall AUC or F1-score.

## 8. Limitations
- Only undersampling was applied → information loss from legitimate transactions.






  
	
