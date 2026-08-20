# SPARK

Here we present the python implementation of **SPARK** (SParsity-based Analysis of Reliable _K_-hubness). SPARK can be used for the identification of resting-state functional networks from BOLD fMRI data using sparse dictionary learning and K-hubness analysis. SPARK provides a set of individually consistent resting state networks, and proposes a novel measure of hubness, "_k_-hubness, by counting the number of functional networks spatiotemporally overlapping in each voxel. This method is fully data-driven, voxel-wise multivariate analysis of BOLD fMRI data based on the [data driven sparse GLM](https://ieeexplore.ieee.org/document/5659483) Parameters of our sparse dictionary learning process are automatically estimated. Statistical reproducibility of the hub estimation is ensured using a boostrap resampling based strategy.

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
└── inputs/ # created by user
└── outputs/ # created by user
└── masks/ # created by user


```

We suggest that the program user generates an /inputs, /outputs, and /masks folder at the same level as /SPARK/ of the repository structure after installing the program into their prefered environment.

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

To run on a slurm cluster, the user must edit the first section at the beginning of:

```text
run_spark_slurm.sh
```
Where the following configuration elements are included at the top of the file.

```text
    #SBATCH --time=72:00:00
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=20
    #SBATCH --mem-per-cpu=2G
    #SBATCH --account=YOUR_ACCOUNT
    #SBATCH --job-name=spark_batch
    #SBATCH --mail-type=BEGIN,END,FAIL
    #SBATCH --mail-user=YOUR_EMAIL
```

The user may need to update the SLURM paramaters to meet their needs (i.e.: `account`, `time`, `cpus`, `memory`, etc.). We have optimized the parameters for working on the [Digital Research Alliance of Canada](https://www.alliancecan.ca/en) computer cluster platform, which runs on the Slurm cluster management and job scheduling system. These parameters will need to be updated if the SPARK user is using a different scheduler.

To submit the job to the slurm scheduler, the user must run the following line:

```bash
sbatch run_spark_slurm.sh
```

---

# Configuration

Before running SPARK, the following variables must be updated inside either *run_spark.sh* or *run_spark_slurm.sh* files:

```bash
input_dir="/path/to/input"
mask_path="/path/to/mask.nii.gz"
output_base="/path/to/output"
suffix="_processed"
```
We suggest the folders follow our recommendations found in the [repository structure](#repository-structure) section of this README, where these directories are created outside of the /SPARK folder to keep the core codes of the program organized. 

---

### Input directory

The input directory contains the preprocessed fMRI files. SPARK accepts file types ending in the following ways:

```
sub-XXX_ses-XX_processed.nii.gz
sub-XXX_ses-XX_processed.nii
...
```
---
### Grey Matter Mask

For volumetric analysis, SPARK requires a Mask of the Grey Matter with the same voxel size and spatial dimensions as the fMRI data. It is the region for which SPARK will perform its analysis. Anything outside of the mask will be excluded from the analysis.

---

### File suffix

#### Group Level Analysis
To allow for the processing of multiple subjects at once, SPARK will search for all the files in the input folder ending with a specific suffix. For group analyses, all the fMRI files should be registered to a standard template space, and all subjects will be using the same [grey matter mask](#grey-matter-mask). 
For example, in the following example: 

```
*_processed.nii
*_processed.nii.gz
```
all of the fMRI files ending in _processed_ will be run at the same time. The user can change the suffix to fit their data. Here is another example:

```
suffix="_rfMRI_smoothed_k8"
```

#### Individual Level Analysis
We also support individual-level analysis, where SPARK is run in the subject's native functional space. In these cases, the configuration file will be as follows. If launching SPARK in the subject's native space, the grey matter mask must be individualized for the subject.

```
input_dir="/path/to/input/fMRI/file/.nii"
mask_path="/path/to/subject/level/mask.nii.gz"
suffix=''
```
---

### Output directory

SPARK automatically creates one output folder for each subject. The following is the stucture of the outputs folder.

```network-analysis/
└── outputs/
    ├── sub-XXX_ses-XX
        ├── boot/
        ├── dict/
        ├── KMAP_sub-XX_ses-XX/
            k_hubness_Subject.nii.gz
            atom_000.nii.gz
            atom_001.nii.gz
            atom_002.nii.gz
            ...
        ├── clusters.mat
        ├── scale.mat
        ├── step1.log/
        ├── step2.log/
        ├── step3.log/
        ├── step4.log/
        ├── step5.log/

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

# Citation
If you use this library for your publications, please cite as:
    
Kangjoo Lee, Jean-Marc Lina, Jean Gotman and Christophe Grova, “SPARK: Sparsity-based analysis of reliable k-hubness and overlapping network structure in brain functional connectivity”, Neuroimage, vol. 134, pp. 434–449, April 2016, Link.

Additional references:
Kangjoo Lee, Hui Ming Khoo, Jean-Marc Lina, François Dubeau, Jean Gotman and Christophe Grova, “Disruption, emergence and lateralization of brain network hubs in mesial temporal lobe epilepsy”, Neuroimage: Clinical, vol. 20, pp. 71–84, June 2018, Link.

Kangjoo Lee, Corey Horien, David O’Connor, Bronwen Garand-Sheridan, Fuyuze Tokoglu, Dustin Scheinost, Evelyn M.R. Lake, R. Todd Constable, “Arousal impacts distributed hubs modulating the integration of brain functional connectivity”, Neuroimage (2022), Link.

