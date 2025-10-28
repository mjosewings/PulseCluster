import h5py
import numpy as np
import random
from collections import defaultdict
import os

# -----------------------------
# Config
# -----------------------------
files = [
    'VitalDB_CalBased_Test_Subset.mat',
    'VitalDB_AAMI_Cal_Subset.mat',
    'VitalDB_AAMI_Test_Subset.mat',
    'VitalDB_Train_Subset.mat',
    'VitalDB_CalFree_Test_Subset.mat'
]

data_dir = "data"
subjects_per_file = 20  # adjust for ~100 total across all files
max_segments_per_subject = 1000
abp_sample_length = 625  # samples per segment

# -----------------------------
# Storage containers
# -----------------------------
all_abp_segments = []
all_sbp_segments = []
all_dbp_segments = []
all_demographics = []
subject_ids = []


# -----------------------------
# Helper functions
# -----------------------------
def extract_numeric_field(field_raw, num_segments):
    """Extract numeric field per segment, handling single-row repeats."""
    if field_raw.shape[0] == 1:
        return np.full(num_segments, np.mean(field_raw))
    return np.squeeze(field_raw)[:num_segments]


def extract_gender(f, gender_raw, num_segments):
    """Convert HDF5 gender field into numeric per-segment array (1=M, 0=F)."""
    if gender_raw.shape[0] == 1:
        first_ref = gender_raw[0, 0]
        if isinstance(first_ref, np.ndarray):
            first_ref = first_ref.item()
        gender_bytes = f[first_ref][()]
        gender_str = gender_bytes.tobytes().decode('utf-8').rstrip('\x00').strip()
        gender_numeric = 1 if gender_str == 'M' else 0
        return np.full(num_segments, gender_numeric)

    # Per-segment references
    gender_per_seg = np.zeros(num_segments)
    if gender_raw.dtype == object:
        for i in range(num_segments):
            ref = gender_raw[i]
            if isinstance(ref, np.ndarray):
                ref = ref.item()
            gender_bytes = f[ref][()]
            gender_str = gender_bytes.tobytes().decode('utf-8').rstrip('\x00').strip()
            gender_per_seg[i] = 1 if gender_str == 'M' else 0
    else:
        gender_per_seg = np.squeeze(gender_raw == b'M').astype(float)[:num_segments]
    return gender_per_seg


# -----------------------------
# Main Processing Loop
# -----------------------------
for file_idx, filename in enumerate(files, 1):
    path = os.path.join(data_dir, filename)
    print(f"\n{'=' * 60}")
    print(f"Processing file {file_idx}/{len(files)}: {filename}")
    print(f"{'=' * 60}")

    if not os.path.isfile(path):
        print(f"WARNING: File not found: {path}. Skipping.")
        continue

    try:
        with h5py.File(path, 'r') as f:
            subset = f['Subset']
            signals = subset['Signals']
            num_segments_raw = signals.shape[0]

            # ABP channel (2)
            abp_all = signals[:, 2, :]

            # SBP/DBP labels
            sbp_per_seg = np.squeeze(subset['SBP'][:])
            dbp_per_seg = np.squeeze(subset['DBP'][:])

            # Trim to labelled segments
            num_labelled = len(sbp_per_seg)
            num_segments = min(num_segments_raw, num_labelled)
            abp_all = abp_all[:num_segments]
            sbp_per_seg = sbp_per_seg[:num_segments]
            dbp_per_seg = dbp_per_seg[:num_segments]

            # Demographics
            age_per_seg = extract_numeric_field(subset['Age'][:], num_segments)
            bmi_per_seg = extract_numeric_field(subset['BMI'][:], num_segments)
            height_per_seg = extract_numeric_field(subset['Height'][:], num_segments)
            weight_per_seg = extract_numeric_field(subset['Weight'][:], num_segments)
            gender_per_seg = extract_gender(f, subset['Gender'][:], num_segments)

            demographics_per_seg = np.column_stack(
                (age_per_seg, bmi_per_seg, gender_per_seg, height_per_seg, weight_per_seg))

            # Group segments by unique demographic profile (subject)
            subject_groups = defaultdict(list)
            for i in range(num_segments):
                key = tuple(demographics_per_seg[i])
                subject_groups[key].append(i)

            unique_subjects = list(subject_groups.keys())
            num_subjects = len(unique_subjects)
            print(f"Number of unique subjects: {num_subjects}")

            # Select subjects
            if num_subjects > subjects_per_file:
                selected_subjects = random.sample(unique_subjects, subjects_per_file)
            else:
                selected_subjects = unique_subjects

            # Sample subject ID
            sample_subject_id = f'file{file_idx}'
            if 'Subject' in subset:
                try:
                    subject_refs = np.squeeze(subset['Subject'][:])
                    first_ref = subject_refs[0].item() if isinstance(subject_refs[0], np.ndarray) else subject_refs[0]
                    sample_subject_id = f[first_ref][()].tobytes().decode('utf-8').rstrip('\x00').strip()
                except Exception:
                    pass

            # Extract segments per subject
            for subj_idx, subj_key in enumerate(selected_subjects, 1):
                seg_indices = subject_groups[subj_key]
                all_demographics.append(np.array(subj_key))
                subject_ids.append(f"{sample_subject_id}_{file_idx}_{subj_idx}")

                num_to_select = min(max_segments_per_subject, len(seg_indices))
                selected_segs = random.sample(seg_indices, num_to_select)

                abp_list = [abp_all[i][:abp_sample_length] for i in selected_segs]
                sbp_list = [sbp_per_seg[i] for i in selected_segs]
                dbp_list = [dbp_per_seg[i] for i in selected_segs]

                all_abp_segments.append(np.array(abp_list))
                all_sbp_segments.append(np.array(sbp_list))
                all_dbp_segments.append(np.array(dbp_list))

            print(f"File {file_idx} extraction complete: {len(selected_subjects)} subjects processed.")

    except Exception as e:
        print(f"ERROR processing file {filename}: {e}. Skipping.")
        continue

# -----------------------------
# Save data
# -----------------------------
print(f"\n{'=' * 60}\nSaving processed data...\n{'=' * 60}")

np.savez_compressed(
    "pulse_clustering/Processed_VitalDB_Data.npz",
    abp_segments=np.array(all_abp_segments, dtype=object),
    sbp_segments=np.array(all_sbp_segments, dtype=object),
    dbp_segments=np.array(all_dbp_segments, dtype=object),
    demographics=np.array(all_demographics, dtype=object),
    subject_ids=np.array(subject_ids, dtype=object)
)

print("Saved processed data to 'Processed_VitalDB_Data.npz'")
