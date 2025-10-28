import numpy as np

# Load preprocessed data
data = np.load("Processed_VitalDB_Data.npz", allow_pickle=True)
abp_segments = data['abp_segments']  # List of arrays per subject
sbp_segments = data['sbp_segments']
dbp_segments = data['dbp_segments']
demographics = data['demographics']
subject_ids = data['subject_ids']

# Flatten segments into a single list of 1D arrays
all_abp = [seg for subj in abp_segments for seg in subj]
print(f"Total segments: {len(all_abp)}")