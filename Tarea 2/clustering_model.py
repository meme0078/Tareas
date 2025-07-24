import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Ruta del archivo CSV
file_path = '/Users/melissasanchez/Desktop/ArtificialIntelligence/Tarea/Tarea 2/Tasty Bytes/recipe_site_traffic_2212.csv'
df = pd.read_csv(file_path)

# Variables utilizadas
features = ['calories', 'sugar']
df_clean = df[features].dropna()

# Escalamiento y modelo de clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

kmeans = KMeans(n_clusters=3, random_state=42)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)
df.loc[df_clean.index, 'cluster'] = df_clean['cluster']

# Ruta de salida
output_dir = '/Users/melissasanchez/Desktop/ArtificialIntelligence/Tarea/Tarea 2/Tasty Bytes/ClusteringResultados'
os.makedirs(output_dir, exist_ok=True)

# 1. Scatterplot de clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='calories', y='sugar', hue='cluster', palette='Set2', s=80)
plt.title("Clustering: Calorías vs Azúcar")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/clustering_scatter.png")
plt.close()

# 2. Promedio por cluster
cluster_means = df.groupby('cluster')[features].mean()
cluster_means.plot(kind='bar', figsize=(10, 6), colormap='Set2')
plt.title("Promedio de calorías y azúcar por cluster")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{output_dir}/cluster_means.png")
plt.close()

# 3. Boxplot de calorías por cluster
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='cluster', y='calories', palette='Set2')
plt.title("Boxplot: Calorías por Cluster")
plt.tight_layout()
plt.savefig(f"{output_dir}/boxplot_calories_by_cluster.png")
plt.close()

# 4. Boxplot de azúcar por cluster
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='cluster', y='sugar', palette='Set2')
plt.title("Boxplot: Azúcar por Cluster")
plt.tight_layout()
plt.savefig(f"{output_dir}/boxplot_sugar_by_cluster.png")
plt.close()

# 5. Conteo por categoría y cluster (si existe)
if 'category' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='category', hue='cluster', palette='Set2')
    plt.xticks(rotation=45)
    plt.title("Conteo por categoría y cluster")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_cluster_count.png")
    plt.close()

print(f"✅ ¡Gráficas generadas correctamente en: {output_dir}")

