# **Dimensionality Reduction**
### *Redukcja wymiarowości*

*Chapter 8 from "Hands-On Machine Learning" by Aurélien Géron*

---

## **Overview / Przegląd**

Dimensionality reduction is a crucial technique in machine learning that helps combat the curse of dimensionality, reduce computational complexity, and enable better data visualization. This section covers the main approaches and algorithms for reducing the number of features while preserving the most important information.

Redukcja wymiarowości to kluczowa technika w uczeniu maszynowym, która pomaga walczyć z przekleństwem wymiarowości, zmniejsza złożoność obliczeniową i umożliwia lepszą wizualizację danych. Ta sekcja obejmuje główne podejścia i algorytmy redukcji liczby cech przy zachowaniu najważniejszych informacji.

---

## **📁 Notebooks in This Directory / Notatniki w tym katalogu**

### **[PCA_Analysis.ipynb](PCA_Analysis.ipynb)**
Comprehensive guide to Principal Component Analysis including:
- Mathematical foundations and eigenvalue decomposition
- Explained variance ratio and choosing optimal components
- Data visualization in 2D and 3D
- Comparison with original features
- Real-world applications with Iris and high-dimensional datasets

---

## **🎯 Key Concepts / Kluczowe koncepcje**

### **The Curse of Dimensionality / Przekleństwo wymiarowości**
- High-dimensional spaces become sparse
- Distance metrics become less meaningful
- Increased computational complexity
- Risk of overfitting

### **Main Approaches / Główne podejścia**

#### **1. Projection Methods / Metody projekcji**
- **PCA (Principal Component Analysis)** - Maximizes variance preservation
- **Random Projection** - Fast approximation method
- **Linear Discriminant Analysis (LDA)** - Supervised dimensionality reduction

#### **2. Manifold Learning / Uczenie rozmaitości**
- **LLE (Locally Linear Embedding)** - Preserves local relationships
- **t-SNE** - Excellent for visualization
- **Isomap** - Geodesic distance preservation

---

## **📊 When to Use Each Method / Kiedy używać każdej metody**

| Method | Best For | Advantages | Disadvantages |
|--------|----------|------------|---------------|
| **PCA** | General purpose, linear relationships | Fast, interpretable | Linear assumptions |
| **LLE** | Non-linear manifolds | Preserves local structure | Sensitive to noise |
| **t-SNE** | Data visualization | Great for clustering visualization | Computationally expensive |
| **Random Projection** | Very high dimensions | Very fast | Less precise |

---

## **🔬 Mathematical Foundations / Podstawy matematyczne**

### **PCA Optimization Objective**
$$
\max_{\mathbf{w}} \frac{\mathbf{w}^T \mathbf{C} \mathbf{w}}{\mathbf{w}^T \mathbf{w}}
$$

Where $\mathbf{C}$ is the covariance matrix.

### **Eigenvalue Decomposition**
$$
\mathbf{C} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T
$$

Where $\mathbf{Q}$ contains eigenvectors and $\boldsymbol{\Lambda}$ contains eigenvalues.

---

## **🎨 Visualization Benefits / Korzyści wizualizacji**

- **2D/3D Plotting**: Visualize high-dimensional data
- **Cluster Discovery**: Identify natural groupings
- **Outlier Detection**: Spot anomalous data points
- **Feature Relationships**: Understand feature interactions

---

## **⚡ Performance Considerations / Względy wydajnościowe**

### **Computational Complexity**
- **PCA**: O(n³) for eigendecomposition
- **Randomized PCA**: O(n²k) for k components
- **Incremental PCA**: O(nk) memory efficient

### **Memory Usage**
- Standard PCA requires full data in memory
- Incremental PCA processes data in batches
- Random projection requires minimal memory

---

## **🚀 Best Practices / Najlepsze praktyki**

1. **Data Preprocessing**:
   - Always standardize features before PCA
   - Handle missing values appropriately
   - Remove or transform outliers

2. **Component Selection**:
   - Use explained variance ratio
   - Elbow method for optimal components
   - Consider domain knowledge

3. **Validation**:
   - Test on held-out data
   - Monitor downstream task performance
   - Validate assumptions (linearity for PCA)

---

## **📈 Applications / Zastosowania**

- **Data Visualization**: High-dimensional data exploration
- **Feature Engineering**: Creating composite features
- **Noise Reduction**: Removing less important dimensions
- **Data Compression**: Reducing storage requirements
- **Preprocessing**: For other ML algorithms