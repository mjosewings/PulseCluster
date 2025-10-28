import numpy as np

# ---------- Load processed data ----------
data = np.load("Processed_VitalDB_Data.npz", allow_pickle=True)
abp_segments = data['abp_segments']  # List of arrays per subject
all_abp = [seg for subj in abp_segments for seg in subj]  # Flatten all segments
all_abp = np.array(all_abp)  # Shape: (num_segments, samples_per_segment)
print(f"Total segments: {all_abp.shape[0]}")

# ---------- Optimized Recursive Clustering ----------
def recursive_cluster_vec(segments, threshold=0.3, min_size=10, max_depth=50, depth=0):
    if len(segments) <= min_size or depth >= max_depth:
        return [segments]

    mean_seg = np.mean(segments, axis=0)
    segments_centered = segments - segments.mean(axis=1, keepdims=True)
    mean_centered = mean_seg - mean_seg.mean()
    norms_segments = np.linalg.norm(segments_centered, axis=1)
    norm_mean = np.linalg.norm(mean_centered)
    similarity = (segments_centered @ mean_centered) / (norms_segments * norm_mean + 1e-8)

    group1 = segments[similarity >= threshold]
    group2 = segments[similarity < threshold]

    clusters = []
    if len(group1) > 0:
        clusters += recursive_cluster_vec(group1, threshold, min_size, max_depth, depth + 1)
    if len(group2) > 0:
        clusters += recursive_cluster_vec(group2, threshold, min_size, max_depth, depth + 1)

    return clusters

clusters = recursive_cluster_vec(all_abp)
print(f"Total clusters: {len(clusters)}")

# ---------- Optimized Closest-Pair per Cluster ----------
def closest_pair_vec(cluster):
    cluster_centered = cluster - cluster.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(cluster_centered, axis=1)
    corr_matrix = (cluster_centered @ cluster_centered.T) / (norms[:, None] * norms[None, :] + 1e-8)
    np.fill_diagonal(corr_matrix, -np.inf)
    i, j = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
    return (cluster[i], cluster[j]), corr_matrix[i, j]

representative_pairs = [closest_pair_vec(c) for c in clusters]

# ---------- Kadane's Algorithm ----------
def kadane(arr):
    max_sum = curr_sum = arr[0]
    start = end = s = 0
    for i in range(1, len(arr)):
        if curr_sum + arr[i] < arr[i]:
            curr_sum = arr[i]
            s = i
        else:
            curr_sum += arr[i]
        if curr_sum > max_sum:
            max_sum = curr_sum
            start, end = s, i
    return max_sum, start, end

max_intervals = [kadane(seg) for seg in all_abp]

# ---------- Save results ----------
np.save("clusters.npy", np.array(clusters, dtype=object))
np.save("rep_pairs.npy", np.array(representative_pairs, dtype=object))
np.save("max_intervals.npy", np.array(max_intervals, dtype=object))

print("Clusters, representative pairs, and Kadane's intervals saved successfully.")
