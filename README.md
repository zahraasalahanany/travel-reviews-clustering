# Travel Reviews Clustering

An exploratory **unsupervised machine learning** project examining travel destination ratings and clustering reviewer profiles using **K-Means** and **DBSCAN**. The notebook also includes exploratory data analysis (EDA), feature scaling, **PCA** visualizations, and scikit-learn pipelines.

## Project overview

The goal is to explore similarities among travelers based on their average ratings across ten destination categories, including restaurants, museums, parks, beaches, and more.

### What's included
- Data exploration, summary statistics, and checks for duplicates
- Distribution, box, correlation heatmap, scatter, and pair plots
- K-Means clustering and the elbow method
- DBSCAN clustering and an epsilon/outlier-percentage exploration
- Two-dimensional PCA plots for visualizing clustering results
- Scikit-learn pipelines that combine scaling with clustering
- A Random Forest cross-validation experiment to predict K-Means-generated cluster labels

## Dataset

**Travel Reviews**, UCI Machine Learning Repository, donated by Shini Renjith (2018). The dataset contains **980 rows**, ten numeric review-category features, and a user identifier. The included data file is `data/tripadvisor_review.csv`.

- Dataset: https://archive.ics.uci.edu/dataset/484/travel+reviews
- Citation: Renjith, S. (2018). *Travel Reviews* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56K6W
- License: **CC BY 4.0**. The source and attribution are provided here in accordance with the license.

## Project structure

```text
travel-reviews-clustering/
├── data/
│   └── tripadvisor_review.csv
├── travel_review.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting started

1. Clone or download this repository and open a terminal in its folder.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `travel_review.ipynb` and run the cells in order from the repository root. The data-loading cell uses the relative path `data/tripadvisor_review.csv`.

## Important interpretation notes

- The notebook's initial K-Means experiment fits the **raw numeric features**, whereas the later K-Means pipeline applies **StandardScaler** before clustering. Those results therefore use different preprocessing approaches and should not be interpreted as directly identical.
- PCA is used to **visualize** clusters in two dimensions; it is not used as the clustering input in the original experiments.
- The Random Forest cross-validation experiment predicts labels assigned by K-Means. Its accuracy measures agreement with these **pseudo-labels**, not the independently verified accuracy or quality of the clusters.
- Cluster counts and DBSCAN hyperparameters are exploratory choices, not proven optimal settings. For a stronger evaluation, compare clustering metrics such as silhouette score and assess cluster stability and interpretability.

## Technologies

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook.
