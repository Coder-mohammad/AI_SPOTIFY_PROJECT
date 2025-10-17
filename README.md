#  Spotify Genre Segmentation Project

##  Overview
This project performs **genre-based segmentation** of Spotify songs using **machine learning** and **unsupervised clustering** techniques.  
The goal is to analyze audio features, identify natural groupings among songs, and visualize the relationships between genres in a 2D or 3D space.

---

##  Features
- Data preprocessing and cleaning using **pandas** and **NumPy**  
- Feature scaling with **StandardScaler**  
- Dimensionality reduction using **PCA** and **t-SNE**  
- Clustering with **K-Means**, **DBSCAN**, and **Agglomerative Clustering**  
- Model evaluation using **Silhouette Score** and **Davies-Bouldin Index**  
- 2D visualizations using **Matplotlib**

---

##  Algorithms Used
| Technique | Purpose |
|------------|----------|
| PCA | Reduces high-dimensional data |
| t-SNE | Non-linear embedding for visualization |
| K-Means | Partitioning-based clustering |
| DBSCAN | Density-based clustering |
| Agglomerative | Hierarchical clustering |

---

##  Project Structure
│
├── spotify_songs.csv              # Dataset
├── spotify_genre_segmentation.py  # Main Python script
├── figures/                       # Output visualizations
└── README.md                      # Project description
---

##  Requirements
Install the dependencies using pip:
```bash
pip install pandas numpy scikit-learn matplotlib
� Outputs
	•	Cluster plots (saved in the figures/ folder)
	•	Model comparison table with accuracy metrics
	•	Visualizations of song embeddings and cluster boundaries

⸻

Results & Insights
	•	The best clustering model is determined using silhouette and Davies-Bouldin scores.
	•	Similar genres are grouped closer in the reduced 2D embedding.
	•	Insights can help identify hidden genre patterns and recommendation trends.

⸻

 Author

Mohammad Daulah
B.Tech – Artificial Intelligence & Machine Learning
Aditya University

⸻
