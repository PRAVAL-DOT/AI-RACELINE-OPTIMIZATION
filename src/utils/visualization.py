import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def generate_style_visualizations(embeddings: np.ndarray, labels: np.ndarray, mapping_csv_path: str, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)
    df_map = pd.read_csv(mapping_csv_path)
    label_to_driver = dict(zip(df_map['Label'], df_map['Driver']))
    
    # Fixed syntax: Calculate array outside list comprehension loop
    unique_labels = np.unique(labels)
    driver_names = [label_to_driver.get(i, f"ID_{i}") for i in sorted(unique_labels)]
    
    print("Computing t-SNE projections...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1), random_state=42, init='pca')
    components = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=components[:, 0], y=components[:, 1], 
        hue=[label_to_driver[lbl] for lbl in labels], 
        palette="tab20", legend="full", alpha=0.8
    )
    plt.title("Driver Style Latent Space (t-SNE Projective Geometry Mapping)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(os.path.join(output_dir, "driver_style_tsne.png"), dpi=300)
    plt.close()

    norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cosine_sim = np.dot(norm_emb, norm_emb.T)
    
    driver_sim_matrix = np.zeros((len(driver_names), len(driver_names)))
    for idx_i, lbl_i in enumerate(sorted(unique_labels)):
        for idx_j, lbl_j in enumerate(sorted(unique_labels)):
            mask_i = (labels == lbl_i)
            mask_j = (labels == lbl_j)
            if np.any(mask_i) and np.any(mask_j):
                driver_sim_matrix[idx_i, idx_j] = np.mean(cosine_sim[mask_i][:, mask_j])

    plt.figure(figsize=(10, 8))
    sns.heatmap(driver_sim_matrix, xticklabels=driver_names, yticklabels=driver_names, cmap="vlag", annot=True, fmt=".2f")
    plt.title("Mean Cross-Driver Style Cosine Similarity Index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "driver_cosine_similarity.png"), dpi=300)
    plt.close()
    print(f"Visualizations saved to metrics output folder: {output_dir}/")