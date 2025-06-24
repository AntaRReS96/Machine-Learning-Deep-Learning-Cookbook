# Regression Algorithms
### Algorytmy regresji

This directory contains Jupyter notebooks demonstrating various regression algorithms from scikit-learn.

## Contents | Zawartość

### 📓 Notebooks | Notatniki

1. **[Linear_Regression.ipynb](Linear_Regression.ipynb)**
   - **English**: Simple and multiple linear regression with normal equations
   - **Polish**: Prosta i wielokrotna regresja liniowa z równaniami normalnymi

2. **[Polynomial_Regression.ipynb](Polynomial_Regression.ipynb)**
   - **English**: Non-linear relationships using polynomial features
   - **Polish**: Związki nieliniowe przy użyciu cech wielomianowych

3. **[Ridge_Lasso_Regression.ipynb](Ridge_Lasso_Regression.ipynb)** *(Coming Soon)*
   - **English**: Regularized regression to prevent overfitting
   - **Polish**: Regresja z regularyzacją zapobiegająca przeuczeniu

4. **[SVR_Regression.ipynb](SVR_Regression.ipynb)** *(Coming Soon)*
   - **English**: Support Vector Regression with different kernels
   - **Polish**: Regresja wektorów nośnych z różnymi jądrami

### 🎯 Key Topics Covered | Główne omawiane tematy

#### English
- **Linear Regression**: Least squares, normal equations, R-squared, RMSE
- **Polynomial Regression**: Feature engineering, basis functions, overfitting
- **Regularization**: Ridge (L2), Lasso (L1), Elastic Net penalties
- **Model Selection**: Cross-validation, bias-variance tradeoff, learning curves
- **Evaluation Metrics**: MSE, RMSE, MAE, R², adjusted R²

#### Polish
- **Regresja liniowa**: Najmniejsze kwadraty, równania normalne, R-kwadrat, RMSE
- **Regresja wielomianowa**: Inżynieria cech, funkcje bazowe, przeuczenie
- **Regularyzacja**: Kary Ridge (L2), Lasso (L1), Elastic Net
- **Wybór modelu**: Walidacja krzyżowa, kompromis bias-wariancja, krzywe uczenia
- **Metryki oceny**: MSE, RMSE, MAE, R², skorygowane R²

### 📊 Algorithm Comparison | Porównanie algorytmów

| Algorithm<br>Algorytm | Use Case<br>Przypadek użycia | Advantages<br>Zalety | Limitations<br>Ograniczenia |
|------------------------|-------------------------------|----------------------|---------------------------|
| **Linear** | Simple relationships<br>Proste związki | Interpretable, fast<br>Interpretowalny, szybki | Linear assumptions<br>Założenia liniowości |
| **Polynomial** | Non-linear patterns<br>Wzorce nieliniowe | Flexible<br>Elastyczny | Overfitting risk<br>Ryzyko przeuczenia |
| **Ridge** | Many features<br>Wiele cech | Prevents overfitting<br>Zapobiega przeuczeniu | Shrinks coefficients<br>Zmniejsza współczynniki |
| **Lasso** | Feature selection<br>Selekcja cech | Automatic selection<br>Automatyczna selekcja | Can eliminate important features<br>Może wyeliminować ważne cechy |

### 📚 Required Libraries | Wymagane biblioteki

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 🎓 Mathematical Foundations | Podstawy matematyczne

#### English
- **Normal Equation**: `θ = (X^T X)^(-1) X^T y`
- **Cost Function**: `J(θ) = (1/2m) Σ(h_θ(x_i) - y_i)²`
- **Ridge Penalty**: `J(θ) = MSE + α Σθ_j²`
- **Lasso Penalty**: `J(θ) = MSE + α Σ|θ_j|`

#### Polish
- **Równanie normalne**: `θ = (X^T X)^(-1) X^T y`
- **Funkcja kosztu**: `J(θ) = (1/2m) Σ(h_θ(x_i) - y_i)²`
- **Kara Ridge**: `J(θ) = MSE + α Σθ_j²`
- **Kara Lasso**: `J(θ) = MSE + α Σ|θ_j|`

### 🚀 How to Use | Jak używać

1. **English**: 
   - Begin with Linear Regression to understand fundamentals
   - Progress to Polynomial for non-linear relationships
   - Use regularized methods for complex datasets
   - Compare different approaches on the same data

2. **Polish**:
   - Zacznij od regresji liniowej, aby zrozumieć podstawy
   - Przejdź do wielomianowej dla związków nieliniowych
   - Używaj metod regularyzowanych dla złożonych zbiorów danych
   - Porównuj różne podejścia na tych samych danych

### 📈 Performance Evaluation | Ocena wydajności

#### Metrics | Metryki

- **MSE (Mean Squared Error)**: `Σ(y_true - y_pred)² / n`
- **RMSE (Root Mean Squared Error)**: `√MSE`
- **MAE (Mean Absolute Error)**: `Σ|y_true - y_pred| / n`
- **R² (Coefficient of Determination)**: `1 - SS_res/SS_tot`

### 🔧 Advanced Topics | Tematy zaawansowane

- **Feature Engineering**: Polynomial features, interaction terms
- **Regularization Path**: How coefficients change with regularization strength
- **Cross-Validation**: K-fold validation for model selection
- **Learning Curves**: Training vs validation performance over time

### 🔗 Related Topics | Powiązane tematy

- [Classification Algorithms](../Classification/README.md)
- [Clustering Algorithms](../Clustering/README.md)
- [Feature Selection Techniques](../Feature_Selection/README.md)

---

**Note | Uwaga**: All notebooks include step-by-step explanations, mathematical derivations, and practical implementations with real datasets.

**Wszystkie notatniki zawierają objaśnienia krok po kroku, wyprowadzenia matematyczne i praktyczne implementacje z rzeczywistymi zbiorami danych.**
