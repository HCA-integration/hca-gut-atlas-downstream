# HCA Gut Atlas Downstream analysis

The readme below documents locations and versioning for objects used by the HCA Gut Cell Atlas group

## Getting started

We use a Conda environment to control package versions 
1. **Install Miniconda (if you don't already have it)**
   - [Miniconda Installation Instructions](https://docs.conda.io/en/latest/miniconda.html)

2. **Clone this repository**
git clone https://github.com/HCA-integration/hca-gut-atlas-downstream.git
cd hca-gut-atlas-downstream

3. **Create the environment from `environment.yml`** 
conda env create -f environment.yml

the yml file details package versions and names the resulting env "gca"

4. **Activate the new environment**
conda activate gca

5. **Validate installation (optional)**
python -c "import scanpy; import numpy; import pandas; print('Python libraries OK')" Rscript -e "library(Seurat); library(dplyr); library(compositions); print('R libraries OK')"

Once the environment is activated, you’ll have all the packages (R and Python) needed for the project.

## File Descriptions

- **`gca_concatenated_anndata_March7_2025.h5ad`**: Initial NON-integrated h5ad object
- **`gca_concatenated_metadata_March7_2025.csv`**: Cell-level metadata csv for all datasets in the h5ad (also present in anndata.obs in .h5ad above)

---

## Version History

### v0.2 (Initial Release)

**Release Date**: 2025-03-07
**Files**:
- `gca_concatenated_anndata_March7_2025.h5ad`
- `gca_concatenated_metadata_March7_2025.csv`
  
**Changes**:
- Added several additional datasets for a total of 25
- Low threshold QC was done (>100 counts per cell, >50 genes per cell).
- Contains 336 donors, 752 samples, 3322132 cells, 37592 genes, and 112 harmonized author celltypes

**Notes**:
- For usage instructions, see [../../code/README.md](../../code/README.md).
- MD5 Checksum: `abc123...`

---

### v0.1 (First pass at integration)

**Release Date**: 2024-06-24
**Files**:
- `healthy_concat_qc_leiden_clust.h5ad`
- `healthy_concat_qc_obs.csv`
  
**Changes**:
- Created the first merged and integrated H5AD from 18 studies including PanGI integrated data
- Basic QC was done (removed doublets, low-quality cells) and data was integrated using scIB pipeline
- Contains 168 donors, 1267745 cells, and 452 samples

**Notes**:
- This object was used for the first GCA working group Meetup in Cambridge, UK in June, 2024. 
- This contains integrated clusters where one snRNA and several scRNA studies were integrated

---

## Future Updates

When you introduce new versions:
1. **Add a new heading to the versions list** (e.g., `### v1.2`)  
    2. **State the release date**  
    3. **List the updated files**  
    4. **Describe the changes** (e.g., new samples, removed outliers, etc.)  
    5. **Include checks or integrity verifications** 
        - See tests/validate_data_integrity.py for methods to validate the integrated object contains HCA and CELLxGENE requirements
6. **Add any relevant notes** or references to scripts used  

---

## Download Instructions

- The data files are stored in [our S3 bucket / HPC location / external server].
- You can pull the latest version via `aws s3 cp s3://bucket/h5ad_version_1.1.h5ad data/` (or other location).

---

## Data Integrity Validation

- Use [`tests/validate_data_integrity.py`](../tests/test_data_integrity.py) to verify that the H5AD matches expected cell/gene counts and metadata columns.

---

## Contributing

- When you update or add new data, **please**:
  1. Upload the new file to the remote data store (Currently our shared Google Drive - "hca_gut_working_group").
  2. Update this README with:
     - **Version**  
     - **Changes**  
     - **Checks**  
     - **Any usage notes**  
  3. Submit a pull request with the new version details.
