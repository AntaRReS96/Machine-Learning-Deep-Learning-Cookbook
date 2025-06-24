# Clustering Algorithms
### Algorytmy klastrowania

This directory contains Jupyter notebooks demonstrating various unsupervised clustering algorithms from scikit-learn.

## Contents | Zawartość

### 📓 Notebooks | Notatniki

1. **[KMeans_Clustering.ipynb](KMeans_Clustering.ipynb)**
   - **English**: K-Means clustering with elbow method for optimal k selection
   - **Polish**: Grupowanie K-średnich z metodą łokcia do wyboru optymalnego k

2. **[Hierarchical_Clustering.ipynb](Hierarchical_Clustering.ipynb)** *(Coming Soon)*
   - **English**: Agglomerative and divisive hierarchical clustering with dendrograms
   - **Polish**: Hierarchiczne grupowanie aglomeracyjne i rozdzielcze z dendrogramami

3. **[DBSCAN_Clustering.ipynb](DBSCAN_Clustering.ipynb)** *(Coming Soon)*
   - **English**: Density-based clustering for arbitrary cluster shapes
   - **Polish**: Grupowanie oparte na gęstości dla dowolnych kształtów klastrów

### 🎯 Key Topics Covered | Główne omawiane tematy

#### English
- **K-Means**: Centroid-based clustering, WCSS minimization, elbow method
- **Hierarchical Clustering**: Linkage criteria, dendrogram interpretation, cluster merging
- **DBSCAN**: Density-based clustering, noise handling, core points identification
- **Cluster Evaluation**: Silhouette score, Adjusted Rand Index, inertia analysis
- **Visualization**: Cluster assignments, centroids, decision boundaries

#### Polish
- **K-Means**: Grupowanie oparte na centroidach, minimalizacja WCSS, metoda łokcia
- **Grupowanie hierarchiczne**: Kryteria łączenia, interpretacja dendrogramów, łączenie klastrów
- **DBSCAN**: Grupowanie oparte na gęstości, obsługa szumu, identyfikacja punktów rdzeniowych
- **Ocena klastrów**: Współczynnik sylwetki, skorygowany indeks Rand, analiza inercji
- **Wizualizacja**: Przypisania klastrów, centroidy, granice decyzyjne

### 📊 Algorithm Comparison | Porównanie algorytmów

| Algorithm<br>Algorytm | Best for<br>Najlepszy dla | Assumptions<br>Założenia | Parameters<br>Parametry |
|------------------------|---------------------------|--------------------------|-------------------------|
| **K-Means** | Spherical clusters<br>Klastry sferyczne | Equal cluster sizes<br>Równe rozmiary klastrów | Number of clusters (k)<br>Liczba klastrów (k) |
| **Hierarchical** | Nested clusters<br>Klastry zagnieżdżone | No specific shape<br>Brak określonego kształtu | Linkage criteria<br>Kryteria łączenia |
| **DBSCAN** | Arbitrary shapes<br>Dowolne kształty | Varying densities<br>Różne gęstości | eps, min_samples |

### 📚 Required Libraries | Wymagane biblioteki

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

### 🚀 How to Use | Jak używać

1. **English**: 
   - Start with K-Means for basic understanding
   - Each notebook is self-contained with synthetic datasets
   - Experiment with different parameters and visualizations

2. **Polish**:
   - Zacznij od K-Means dla podstawowego zrozumienia
   - Każdy notatnik jest samodzielny z syntetycznymi zbiorami danych
   - Eksperymentuj z różnymi parametrami i wizualizacjami

### 📈 Performance Metrics | Metryki wydajności

#### English
- **Silhouette Score**: Measures how well-separated clusters are
- **Adjusted Rand Index**: Compares clustering results with ground truth
- **Inertia (WCSS)**: Within-cluster sum of squares for K-means
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance

#### Polish
- **Współczynnik sylwetki**: Mierzy, jak dobrze oddzielone są klastry
- **Skorygowany indeks Rand**: Porównuje wyniki grupowania z prawdą podstawową
- **Inercja (WCSS)**: Suma kwadratów w klastrach dla K-means
- **Indeks Calinski-Harabasz**: Stosunek wariancji między klastrami do wariancji w klastrach

### 🔧 Advanced Topics | Tematy zaawansowane

- **Feature Scaling**: Impact on distance-based algorithms
- **Curse of Dimensionality**: Challenges in high-dimensional clustering
- **Cluster Validation**: Internal and external validation methods
- **Ensemble Clustering**: Combining multiple clustering algorithms

### 🔗 Related Topics | Powiązane tematy

- [Classification Algorithms](../Classification/README.md)
- [Regression Algorithms](../Regression/README.md)
- [Dimensionality Reduction](../Dimensionality_Reduction/README.md)

---

**Note | Uwaga**: All notebooks include comprehensive explanations, mathematical foundations, and practical examples with real-world datasets.

**Wszystkie notatniki zawierają wyczerpujące objaśnienia, podstawy matematyczne i praktyczne przykłady z rzeczywistymi zbiorami danych.**
