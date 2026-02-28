# 01_supervised_learning

Supervised Learning is a type of Machine Learning where the model learns patterns from **labeled data** — meaning each input comes with a corresponding output. The goal is for the model to **predict outputs for new, unseen data** based on what it has learned.


## 🔹 Key Concepts

- **Features (X):** Input variables used for prediction.
- **Target (y):** Output variable the model aims to predict.
- **Training set:** Data used to train the model.
- **Test set:** Data used to evaluate the model’s performance.
- **Overfitting:** Model performs well on training data but poorly on unseen data.
- **Underfitting:** Model performs poorly on both training and test data.


## 📂 Subfolders

### `regression/`
Used when the target variable is **continuous/numeric**.  
**Algorithms covered:**
- Linear Regression  
- Polynomial Regression  
- Ridge & Lasso Regression  
- Support Vector Regression (SVR)  

**Example Use-Cases:**
- Predicting house prices  
- Forecasting sales or stock prices  

### `classification/`
Used when the target variable is **categorical/discrete**.  
**Algorithms covered:**
- Logistic Regression  
- Decision Trees & Random Forest  
- K-Nearest Neighbors (KNN)  
- Support Vector Machines (SVM)  
- Gradient Boosting Methods (XGBoost, LightGBM)  

**Example Use-Cases:**
- Email spam detection  
- Customer churn prediction  
- Disease diagnosis (e.g., diabetes prediction)


## 🔹 Tips & Best Practices

1. **Feature Scaling:** Algorithms like KNN, SVM, and Logistic Regression benefit from scaling features.  
2. **Train/Test Split:** Always split your data (e.g., 80/20) to evaluate performance fairly.  
3. **Cross-Validation:** Helps in understanding model stability and preventing overfitting.  
4. **Regularization:** Ridge, Lasso, and ElasticNet prevent overfitting in regression tasks.  
5. **Evaluation Metrics:**  
   - Regression: MAE, MSE, RMSE, R² Score  
   - Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC  


## 📖 Recommended Learning Flow

1. Start with **Linear Regression** and evaluate with MSE/R².  
2. Move to **Logistic Regression** for binary classification tasks.  
3. Explore **Tree-Based Methods** for both regression and classification.  
4. Experiment with **ensemble methods** (Random Forest, XGBoost) for improved performance.  


This folder contains **notebooks, scripts, and examples** to help you practice and master supervised learning algorithms end-to-end.