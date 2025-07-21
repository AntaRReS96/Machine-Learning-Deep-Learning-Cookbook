# **Ensemble Learning and Random Forests**
### *Uczenie zespołowe i lasy losowe*

*Chapter 7 from "Hands-On Machine Learning" by Aurélien Géron*

---

## **Overview / Przegląd**

Ensemble learning combines multiple machine learning models to create a stronger predictor than any individual model alone. This approach leverages the "wisdom of crowds" principle and often leads to better predictive performance, increased robustness, and reduced overfitting.

Uczenie zespołowe łączy wiele modeli uczenia maszynowego, aby stworzyć silniejszy predyktor niż jakikolwiek pojedynczy model. To podejście wykorzystuje zasadę "mądrości tłumu" i często prowadzi do lepszej wydajności predykcyjnej, zwiększonej odporności i zmniejszonego przeuczenia.

---

## **📁 Notebooks in This Directory / Notatniki w tym katalogu**

### **[Ensemble_Learning.ipynb](Ensemble_Learning.ipynb)**
Comprehensive guide to ensemble methods including:
- Voting classifiers (hard and soft voting)
- Bagging and Random Forest implementation
- Boosting algorithms (AdaBoost, Gradient Boosting)
- Stacking and meta-learning
- Performance comparison and analysis

---

## **🎯 Main Ensemble Methods / Główne metody zespołowe**

### **1. Voting Classifiers / Klasyfikatory głosujące**
- **Hard Voting**: Majority vote decision
- **Soft Voting**: Average of predicted probabilities
- **Weighted Voting**: Different weights for different models

### **2. Bagging (Bootstrap Aggregating)**
- **Bootstrap Sampling**: Random sampling with replacement
- **Parallel Training**: Independent model training
- **Aggregation**: Voting or averaging predictions
- **Examples**: Random Forest, Extra Trees

### **3. Boosting**
- **Sequential Training**: Each model corrects previous errors
- **Weighted Samples**: Focus on misclassified instances
- **Examples**: AdaBoost, Gradient Boosting, XGBoost

### **4. Stacking (Stacked Generalization)**
- **Base Models**: First level predictors
- **Meta-Model**: Learns to combine base predictions
- **Cross-Validation**: Prevents overfitting in meta-model

---

## **🌳 Random Forest Deep Dive / Szczegółowa analiza lasów losowych**

### **Key Features / Kluczowe cechy**
- **Bootstrap Sampling**: Each tree trained on different subset
- **Random Feature Selection**: Subset of features at each split
- **Majority Voting**: Final prediction from all trees
- **Feature Importance**: Aggregate importance scores

### **Hyperparameters / Hiperparametry**
- `n_estimators`: Number of trees
- `max_depth`: Maximum tree depth
- `max_features`: Features to consider for splits
- `min_samples_split`: Minimum samples to split
- `bootstrap`: Whether to use bootstrap sampling

---

## **📊 Algorithm Comparison / Porównanie algorytmów**

| Method | Training | Prediction | Pros | Cons |
|--------|----------|------------|------|------|
| **Voting** | Parallel | Fast | Simple, diverse models | Equal weight assumption |
| **Bagging** | Parallel | Fast | Reduces variance | May not help with bias |
| **Boosting** | Sequential | Fast | Reduces bias and variance | Sequential, overfitting risk |
| **Stacking** | Two-level | Medium | Flexible combination | Complex, overfitting risk |

---

## **🔬 Mathematical Foundations / Podstawy matematyczne**

### **AdaBoost Weight Update**
$$
w_i^{(m+1)} = w_i^{(m)} \exp\left(-\alpha_m y_i h_m(x_i)\right)
$$

### **Gradient Boosting**
$$
F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)
$$

Where $h_m(x)$ fits the negative gradient of the loss function.

### **Random Forest Feature Importance**
$$
\text{Importance}(X_j) = \frac{1}{T} \sum_{t=1}^{T} \sum_{i \in \text{splits on } X_j} p_i \Delta_i
$$

---

## **⚖️ Bias-Variance Decomposition / Dekompozycja bias-wariancja**

### **How Ensembles Help / Jak zespoły pomagają**
- **Bagging**: Primarily reduces variance
- **Boosting**: Reduces both bias and variance
- **Stacking**: Can reduce both depending on meta-model

### **Mathematical Insight**
For independent models with equal variance σ²:
$$
\text{Var}(\text{average}) = \frac{\sigma^2}{n}
$$

---

## **🚀 Best Practices / Najlepsze praktyki**

### **Model Diversity / Różnorodność modeli**
1. **Different Algorithms**: Combine tree-based, linear, kernel methods
2. **Different Features**: Use different feature subsets
3. **Different Training Data**: Bootstrap, cross-validation folds
4. **Different Hyperparameters**: Various configurations

### **Ensemble Size / Rozmiar zespołu**
- **Sweet Spot**: Usually 50-500 models
- **Diminishing Returns**: Performance plateaus after certain size
- **Computational Trade-off**: More models = slower prediction

### **Overfitting Prevention / Zapobieganie przeuczeniu**
- **Cross-Validation**: For meta-model training
- **Regularization**: In boosting algorithms
- **Early Stopping**: Monitor validation performance

---

## **📈 Performance Metrics / Metryki wydajności**

### **Individual vs Ensemble / Pojedynczy vs zespół**
- Compare base model performance
- Measure ensemble improvement
- Analyze computation time trade-offs

### **Diversity Measures / Miary różnorodności**
- **Disagreement**: How often models disagree
- **Q-Statistic**: Pairwise diversity measure
- **Correlation**: Between model predictions

---

## **🎯 When to Use Ensembles / Kiedy używać zespołów**

### **Good Candidates / Dobre kandydaci**
- High-variance models (decision trees)
- Models with different strengths
- Sufficient computational resources
- Need for robust predictions

### **Challenges / Wyzwania**
- **Interpretability**: Harder to explain decisions
- **Computational Cost**: Training and prediction time
- **Memory Usage**: Storing multiple models
- **Hyperparameter Tuning**: More parameters to optimize

---

## **🔗 Real-World Applications / Zastosowania w świecie rzeczywistym**

- **Netflix Prize**: Winning solution used ensemble methods
- **Kaggle Competitions**: Top solutions often use ensembles
- **Finance**: Risk assessment and fraud detection
- **Medical Diagnosis**: Combining different diagnostic approaches
- **Computer Vision**: Object detection and image classification