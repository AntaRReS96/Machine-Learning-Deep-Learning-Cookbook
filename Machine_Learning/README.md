# **Machine Learning with Scikit-Learn**  
### *Uczenie maszynowe z Scikit-Learn*

*Based on "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron (3rd Edition)*

---

## **📚 Overview / Przegląd**

This comprehensive collection of Jupyter notebooks covers fundamental machine learning concepts and implementations using Scikit-Learn, following the structure and approach of Aurélien Géron's renowned book. Each notebook includes both theoretical explanations and practical implementations with real-world datasets.

Ta kompleksowa kolekcja notatników Jupyter obejmuje podstawowe koncepcje uczenia maszynowego i implementacje przy użyciu Scikit-Learn, zgodnie ze strukturą i podejściem słynnej książki Aurélien Géron. Każdy notatnik zawiera zarówno wyjaśnienia teoretyczne, jak i praktyczne implementacje z rzeczywistymi zbiorami danych.

---

## **🗂️ Directory Structure / Struktura katalogów**

```
Machine_Learning/
├── 01_Classification/          # Supervised Learning - Classification
│   ├── Logistic_Regression.ipynb
│   ├── SVM_Classification.ipynb
│   ├── Decision_Trees.ipynb
│   ├── Random_Forest.ipynb
│   └── README.md
├── 02_Regression/              # Supervised Learning - Regression
│   ├── Linear_Regression.ipynb
│   ├── Polynomial_Regression.ipynb
│   └── README.md
├── 03_Clustering/              # Unsupervised Learning - Clustering
│   ├── KMeans_Clustering.ipynb
│   └── README.md
├── 04_Dimensionality_Reduction/ # Unsupervised Learning - Dimensionality Reduction
│   ├── PCA_Analysis.ipynb
│   └── README.md
├── 05_Ensemble_Methods/        # Ensemble Learning
│   ├── Ensemble_Learning.ipynb
│   └── README.md
├── 06_Unsupervised_Learning/   # Advanced Unsupervised Techniques
│   ├── Anomaly_Detection.ipynb
│   └── README.md
├── 07_Time_Series/            # Time Series Analysis
│   ├── Time_Series_Analysis.ipynb
│   └── README.md
├── Projects/                   # End-to-End Projects
│   └── README.md
└── README.md                   # This file
```

---

## **📖 Notebooks Overview / Przegląd notatników**

### **Part I: The Fundamentals of Machine Learning**

#### **🎯 Classification Algorithms**
- **Logistic Regression** - Linear classifier with probability outputs
- **Support Vector Machines** - Maximum margin classification with kernel trick
- **Decision Trees** - Tree-based decisions with feature importance
- **Random Forest** - Ensemble of decision trees with bootstrap sampling
- **k-Nearest Neighbors (k-NN)** - Instance-based learning with distance metrics
- **Naive Bayes** - Probabilistic classifier based on Bayes' theorem

#### **📈 Regression Algorithms**
- **Linear Regression** - Fundamental regression technique
- **Polynomial Regression** - Non-linear relationships with polynomial features
- **Regularized Regression** - Ridge, Lasso, Elastic Net for overfitting prevention
- **SGD Regression** - Stochastic Gradient Descent for large-scale datasets

#### **🔍 Clustering Algorithms**
- **K-Means Clustering** - Centroid-based clustering with elbow method
- **DBSCAN** - Density-based clustering for arbitrary shaped clusters
- **Hierarchical Clustering** - Tree-based clustering with dendrograms
- **Gaussian Mixture Models** - Probabilistic clustering with soft assignments

#### **📉 Dimensionality Reduction**
- **Principal Component Analysis (PCA)** - Variance-preserving dimensionality reduction

#### **🤝 Ensemble Methods**
- **Bagging, Boosting, and Stacking** - Combining multiple models for better performance
- **Random Forest and Gradient Boosting** - Advanced ensemble techniques

#### **🔍 Unsupervised Learning**
- **Anomaly Detection** - Identifying outliers and novel patterns
- **Gaussian Mixture Models** - Probabilistic clustering

#### **⏰ Time Series Analysis**
- **Forecasting Techniques** - ARIMA, seasonal decomposition, ML approaches
- **Feature Engineering** - Lag features, rolling statistics, seasonality

---

## **🔧 Technical Requirements / Wymagania techniczne**

### **Required Libraries / Wymagane biblioteki**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import *
import scipy.stats as stats
```

### **Installation / Instalacja**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

---

## **📊 Datasets Used / Używane zbiory danych**

- **Iris Dataset** - Classification benchmark
- **Boston Housing** - Regression analysis
- **Wine Quality** - Multi-class classification
- **Synthetic Datasets** - Generated for specific algorithm demonstrations
- **Time Series Data** - Generated patterns for forecasting

---

## **🎓 Learning Path / Ścieżka nauki**

### **Beginner Level / Poziom początkujący**
1. Start with **Linear Regression** - understand basic concepts
2. Move to **Logistic Regression** - introduction to classification
3. Learn **Decision Trees** - interpretable algorithms
4. Practice with **K-Means Clustering** - unsupervised learning basics

### **Intermediate Level / Poziom średniozaawansowany**
1. **Support Vector Machines** - advanced classification
2. **Random Forest** - ensemble methods introduction
3. **PCA** - dimensionality reduction
4. **Anomaly Detection** - unsupervised problem solving

### **Advanced Level / Poziom zaawansowany**
1. **Ensemble Methods** - combining multiple algorithms
2. **Time Series Analysis** - sequential data processing
3. **Advanced Clustering** - complex unsupervised techniques
4. **End-to-End Projects** - complete ML pipeline

---

## **🔬 Key Concepts Covered / Omawiane kluczowe koncepcje**

### **Supervised Learning**
- **Bias-Variance Tradeoff** - Model complexity vs. generalization
- **Overfitting & Underfitting** - Model performance issues
- **Cross-Validation** - Model evaluation techniques
- **Feature Selection** - Choosing relevant predictors
- **Model Evaluation** - Metrics and validation strategies

### **Unsupervised Learning**
- **Clustering Validation** - Silhouette analysis, elbow method
- **Dimensionality Reduction** - PCA, manifold learning
- **Anomaly Detection** - Statistical and ML approaches
- **Pattern Discovery** - Finding hidden structures

### **Model Selection & Tuning**
- **Hyperparameter Optimization** - Grid search, random search, Bayesian optimization
- **Pipeline Creation** - Preprocessing and modeling workflows with scikit-learn
- **Cross-Validation** - Robust model evaluation strategies
- **Model Comparison** - Choosing the best algorithm for your problem
- **Performance Metrics** - Accuracy, precision, recall, F1-score, ROC-AUC

---

## **🌟 Best Practices / Najlepsze praktyki**

### **Data Preparation**
- **Data Cleaning** - Handling missing values, outliers
- **Feature Scaling** - Standardization, normalization
- **Feature Engineering** - Creating meaningful variables
- **Train-Test Split** - Proper data splitting strategies

### **Model Development**
- **Start Simple** - Begin with basic algorithms
- **Iterative Improvement** - Gradually increase complexity
- **Validation Strategy** - Robust model evaluation
- **Documentation** - Clear code and result interpretation

### **Code Organization**
- **Modular Code** - Reusable functions and classes
- **Reproducibility** - Fixed random seeds, version control
- **Visualization** - Clear plots and data exploration
- **Error Handling** - Robust code with proper exceptions

---

## **📚 Additional Resources / Dodatkowe zasoby**

### **Books / Książki**
- "Hands-On Machine Learning" by Aurélien Géron
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman

### **Online Resources / Zasoby online**
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Coursera Machine Learning Course](https://www.coursera.org/learn/machine-learning)

### **Practice Platforms / Platformy do ćwiczeń**
- [Kaggle Competitions](https://www.kaggle.com/competitions)
- [Google Colab](https://colab.research.google.com/)
- [Jupyter Notebook](https://jupyter.org/)

---

## **🤝 Contributing / Współpraca**

Contributions are welcome! Please feel free to submit pull requests, report issues, or suggest improvements.

Wkład jest mile widziany! Prosimy o przesyłanie pull requestów, zgłaszanie problemów lub sugerowanie ulepszeń.

---

## **📄 License / Licencja**

This project is licensed under the MIT License - see the LICENSE file for details.

Ten projekt jest licencjonowany na licencji MIT - szczegóły w pliku LICENSE.

---

*"The best way to learn machine learning is by doing machine learning."*  
*"Najlepszym sposobem na naukę uczenia maszynowego jest uczenie maszynowe."*

**Happy Learning! / Miłej nauki!** 🎉 