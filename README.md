# SPARK

**SPARK** is a Python implementation for identifying resting-state functional networks from fMRI data using sparse dictionary learning and K-hubness analysis.


## Repository Structure

```
network-analysis/
└── SPARK/
    ├── install_spark.sh
    ├── run_spark.sh
    ├── run_spark_slurm.sh
    ├── requirements.txt
    ├── requirements-dev.txt
    ├── pipeline_steps1_6.py
    ├── step1_load_data.py
    ├── step2_estimate_scale.py
    ├── step3_bootstrap.py
    ├── step4_dictionary.py
    ├── step5_clustering.py
    ├── step6_kmap_atoms.py
    └── utils.py
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/multifunkim/network-analysis.git
cd network-analysis/SPARK
```

### Standard installation

```bash
bash install_spark.sh
```

### Developer installation

```bash
bash install_spark.sh --developer
```

Developer mode additionally registers the **SPARK Python** Jupyter kernel.

---

# Running SPARK

### Local workstation

Edit the configuration section at the beginning of

```text
run_spark.sh
```

Then execute

```bash
bash run_spark.sh
```

---

### SLURM clusters

Edit the configuration section at the beginning of

```text
run_spark_slurm.sh
```

including your SLURM parameters if needed (`account`, `time`, `cpus`, `memory`, etc.).

Submit the job

```bash
sbatch run_spark_slurm.sh
```

---

# Configuration

Before running SPARK, update the following variables:

```bash
input_dir="/path/to/input"
mask_path="/path/to/mask.nii.gz"
output_base="/path/to/output"
suffix="_processed"
```

### Input directory

Directory containing the preprocessed fMRI files.

Example

```
sub-HC043_ses-01_processed.nii.gz
sub-HC044_ses-01_processed.nii.gz
...
```

### Mask

Gray matter mask used during SPARK.

### Output directory

SPARK automatically creates one output folder for each subject.

### File suffix

SPARK searches for all files ending with

```
*_processed.nii
*_processed.nii.gz
```

For HCP datasets, for example,

```bash
suffix="_rfMRI_smoothed_k8"
```

---

# Automatic Processing

The launcher automatically

- searches all matching fMRI files
- extracts subject identifiers
- creates output directories
- skips completed subjects
- prevents duplicate processing using lock files
- removes lock files after successful completion

---

# Output

For each subject SPARK generates

```
Subject/

    step1/
    step2/
    step3/
    step4/
    step5/

    KMAP_Subject/

        k_hubness_Subject.nii.gz

        atom_001.nii.gz
        atom_002.nii.gz
        ...
```

---

# Citation

Citation information will be added after publication.
