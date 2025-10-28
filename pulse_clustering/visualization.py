import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

# -------------------------------
# Load saved clustering results
# -------------------------------
clusters = np.load("clusters.npy", allow_pickle=True)
rep_pairs = np.load("rep_pairs.npy", allow_pickle=True)
max_intervals = np.load("max_intervals.npy", allow_pickle=True)

print(f"Total clusters: {len(clusters)}")

# -------------------------------
# Create parent output folder
# -------------------------------
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
output_dir = os.path.join(parent_dir, "cluster_visualizations")
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# 1. Cluster sizes histogram
# -------------------------------
cluster_sizes = [len(c) for c in clusters]
plt.figure(figsize=(10, 5))
plt.bar(range(len(cluster_sizes)), cluster_sizes, color='#9c2542', edgecolor='black')
plt.xlabel("Cluster Index")
plt.ylabel("Number of Segments")
plt.title("Cluster Sizes")
plt.savefig(os.path.join(output_dir, "cluster_sizes.png"))
plt.show()

# -------------------------------
# 2-4. Visualizations per cluster
# -------------------------------
for idx, cluster in enumerate(clusters):
    cluster_name = f"Cluster_{idx + 1:02d}"
    cluster_dir = os.path.join(output_dir, cluster_name)
    os.makedirs(cluster_dir, exist_ok=True)

    cluster_arr = np.array(cluster)
    mean_seg = np.mean(cluster_arr, axis=0)

    # Overlay segments
    plt.figure(figsize=(12, 4))
    for seg in cluster_arr:
        plt.plot(seg, color='gray', alpha=0.3)
    plt.plot(mean_seg, color='#9b111e', linewidth=2, label='Cluster Mean')
    plt.title(f"{cluster_name} - {len(cluster_arr)} Segments")
    plt.xlabel("Time")
    plt.ylabel("ABP Signal")
    plt.legend()
    plt.savefig(os.path.join(cluster_dir, f"{cluster_name}_overlay.png"))
    plt.close()

    # Correlation heatmap
    cluster_centered = cluster_arr - cluster_arr.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(cluster_centered, axis=1)
    corr_matrix = (cluster_centered @ cluster_centered.T) / (norms[:, None] * norms[None, :] + 1e-8)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, cmap='magma', vmin=-1, vmax=1)
    plt.title(f"Correlation Heatmap - {cluster_name}")
    plt.savefig(os.path.join(cluster_dir, f"{cluster_name}_heatmap.png"))
    plt.close()

    # Mean ± std
    std_seg = np.std(cluster_arr, axis=0)
    plt.figure(figsize=(12, 4))
    plt.plot(mean_seg, color='blue', label='Mean')
    plt.fill_between(range(len(mean_seg)), mean_seg - std_seg, mean_seg + std_seg,
                     color='blue', alpha=0.2, label='±1 Std')
    plt.title(f"{cluster_name} - Mean ± Std")
    plt.xlabel("Time")
    plt.ylabel("ABP Signal")
    plt.legend()
    plt.savefig(os.path.join(cluster_dir, f"{cluster_name}_mean_std.png"))
    plt.close()

    # Representative pair
    if idx < len(rep_pairs):
        (seg1, seg2), corr = rep_pairs[idx]
        plt.figure(figsize=(12, 4))
        plt.plot(seg1, label='Segment 1')
        plt.plot(seg2, label='Segment 2')
        plt.title(f"{cluster_name} Representative Pair (Corr={corr:.2f})")
        plt.xlabel("Time")
        plt.ylabel("ABP Signal")
        plt.legend()
        plt.savefig(os.path.join(cluster_dir, f"{cluster_name}_rep_pair.png"))
        plt.close()

    # Kadane interval (use first segment as example)
    if idx < len(max_intervals):
        max_sum, start, end = max_intervals[idx]
        seg = cluster_arr[0]
        plt.figure(figsize=(12, 4))
        plt.plot(seg, label='Segment')
        plt.axvspan(start, end, color='red', alpha=0.3, label='Max Interval (Kadane)')
        plt.title(f"{cluster_name} Kadane Interval")
        plt.xlabel("Time")
        plt.ylabel("ABP Signal")
        plt.legend()
        plt.savefig(os.path.join(cluster_dir, f"{cluster_name}_kadane.png"))
        plt.close()

# -------------------------------
# 5. PCA 2D projection of all segments
# -------------------------------
all_segments = np.vstack(clusters)
labels = np.hstack([[i] * len(c) for i, c in enumerate(clusters)])
pca = PCA(n_components=2)
proj = pca.fit_transform(all_segments)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='plasma', s=15)
plt.title("2D PCA Projection of All Segments Colored by Cluster")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label="Cluster Index")
plt.savefig(os.path.join(output_dir, "pca_projection.png"))
plt.show()

print(f"All visualizations saved to '{output_dir}'")
