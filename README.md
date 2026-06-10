# Biomotion Study

This repository contains the analysis code (Python) and supporting data files required to reproduce the results of the study titled "TITLE", by Niko, Nahid, and Johannes Schultz, which investigated the effects of acute lorazepam administration on neural representations of social biological motion using 7T fMRI. The full fMRI dataset can be found on OpenNeuro (Link: __https://openneuro.org/datasets/__).

The repository includes scripts for preprocessing, design matrix generation, GLMsingle-based single-trial beta estimation, univariate sanity checks, representational similarity analysis (RSA), reliability assessment, and group-level statistical inference.


## Repository Structure

### `data/`

Contains supporting data files required to reproduce the analyses.

- **`biomotion_ROI/`**: Functional ROI biomotion mask from an independent study.
- **`design_matrices/`**: Design matrices and accompanying .json files.
- **`events_tsv/`**: Event files used for design-matrix generation.
- **`model_RDMs/`**: Model representational dissimilarity matrices (RDMs) used in RSA.
- **`unblinding_file.csv`**: Group assignment (Lorazepam or Placebo).

### `scripts/`

Contains all analysis code used in the project.

- **`01_preprocessing/`**: fMRIPrep slurm scripts (with and without surface reconstruction).
- **`02_design_matrices/`**: Generation and quality control of condition-level design matrices.
- **`03_smoothing/`**: Spatial smoothing (3 mm FWHM) of preprocessed BOLD data.
- **`04_GLMsingle/`**: Single-trial beta estimation using GLMsingle.
- **`05_ROIs_Masks/`**: Generation of cortex masks and group masks.
- **`06_univar_sanity_analysis/`**: Univariate biological-motion versus scrambled-motion sanity check.
- **`07_model_RDMs/`**: Construction of model representational dissimilarity matrices.
- **`08_searchlight_RSA/`**: Searchlight representational similarity analyses and reliability estimation.
- **`09_group_inference_RSA/`**: Group-level statistical inference using permutation testing and threshold-free cluster enhancement (TFCE).


## Final Analysis Sample

The original dataset contained 63 participants. Several participants were excluded before the final pipeline (sub-114, sub-139, and sub-159):

The final primary analysis sample consisted of:

- lorazepam: n = 29
- placebo: n = 28


## Installation

The analysis environment can be created using the provided Conda environment file:

```bash
conda env create -f environment.yml
conda activate biomotion-glmsingle
```

Some analyses additionally require external neuroimaging tools installed outside the Conda environment, including fMRIPrep and FSL.


## Pipeline Overview

The complete analysis consisted of the following steps:

### 1. BIDS Conversion

Raw DICOM data were converted to BIDS format using HeuDiConv.

### 2. fMRIPrep Preprocessing

Functional data were preprocessed using fMRIPrep. The preprocessing included motion correction, susceptibility distortion correction, co-registration to the structural image, spatial normalization to MNI152NLin2009cAsym space at 1.7 mm isotropic resolution, and FreeSurfer surface reconstruction where available.

Slice-timing correction was omitted. Five dummy scans were flagged in fMRIPrep using `--dummy-scans 5`; these volumes were not removed by fMRIPrep itself, but were handled during downstream modeling.

### 3. Event Timing and Design Matrix Generation

The event files were already aligned to the dummy-scan convention. Time zero corresponded to 5 TRs after scan start, because the experiment began after detection of the sixth scanner trigger (see Data_acqusition_code.m).

Condition-level design matrices were generated for the post-dummy modeled time window. These matrices were later aligned to the upsampled BOLD time series during GLMsingle preparation.

Relevant script:

```text
scripts/02_design_matrices/build_design_from_events.py
```

### 4. Optional Spatial Smoothing

A smoothed variant of the preprocessed BOLD data was generated using spatial-only Gaussian smoothing with a 3 mm FWHM kernel. The subject-specific brain mask was applied after smoothing.

Relevant script:

```text
scripts/03_smoothing/smooth_subjects.py
```

### 5. GLMsingle Single-Trial Beta Estimation (Prince et al., 2022)

GLMsingle was used to estimate single-trial neural response amplitudes from whole-brain fMRI data. The functional time series were temporally upsampled from TR = 1.9 s to approximately 0.633 s before model fitting.

Three GLMsingle branches were evaluated:

- A_baseline: GLMsingle HRF fitting, GLMdenoise, and fractional ridge regression without external confounds
- B_aCompCor: WM/CSF aCompCor regressors added as external nuisance regressors, GLMdenoise disabled, fractional ridge regression retained
- C_confounds: 24-parameter motion model plus spike/outlier regressors added as external nuisance regressors

Both smoothed and unsmoothed BOLD inputs were tested. The final downstream analysis used the unsmoothed B_aCompCor branch because it provided the best split-half reliability while still recovering a plausible biological-motion response.

Relevant scripts:

```text
scripts/04_glmsingle/
```

### 6. Cortex ROI and Group Mask Generation

Subject-specific cortex masks were generated by intersecting functional run coverage with gray-matter/cortical tissue definitions. For subjects with FreeSurfer reconstructions, cortical ribbon masks were warped to MNI space. For subjects without FreeSurfer reconstructions, MNI gray-matter probability maps thresholded at 30% were used.

Group-level masks were then created by retaining voxels present in the cortex ROI of a minimum proportion of participants.

Relevant scripts:

```text
scripts/05_rois_masks/build_cortex_rois_gm_only.py 
scripts/05_rois_masks/build_group_mask_from_rois.py
```

### 7. Univariate Sanity Check

A biological-motion-versus-scrambled contrast was computed from the GLMsingle betas in the placebo group. This analysis was used as a sanity check to verify that each GLMsingle branch recovered a canonical biological-motion response.

Relevant scripts:

```text
scripts/06_univar_sanity_analysis/
```

### 8. Model RDM Generation

Three model representational dissimilarity matrices (RDMs) were generated to test specific representational hypotheses:

- bio_vs_scrambled: biological motion versus scrambled motion
- castaldi_fvp: between-emotion versus within-emotion structure among walkers
- castaldi_gtvfp: emotion-neutral versus emotion-emotion structure among walkers

Relevant script:

```text
scripts/07_model_rdms/model_rdms.py
```

### 9. Searchlight RSA

Searchlight RSA was performed using multi-predictor rank-based regression. Searchlight centers were constrained to a cortical group mask, while local neighborhoods were defined using a k-nearest-neighbor approach.

Settings:
- rank-based regression RSA
- cov20 cortical mask
- k = 60 nearest valid voxels
- maximum voxel distance constraint = 5 voxels

Relevant script:

```text
scripts/08_searchlight_rsa/rsa_searchlight_compare_ABC_glm_multi.py
```

### 10. Split-Half Reliability and Pipeline Selection

Split-half reliability maps were computed to evaluate the stability of RSA outputs across runs. The final GLMsingle branch was selected primarily based on post-RSA reliability, with the univariate biomotion sanity check used as an additional guardrail.

Relevant script:

```text
scripts/08_searchlight_rsa/summarize_searchlight_reliability.py
```

### 11. TFCE Group Mask and Inference

A TFCE-safe group mask was created by retaining voxels with full subject coverage within the cov20 cortical analysis mask.
Group-level inference was performed on subject-level RSA beta maps using two-sample permutation testing with TFCE. Subject maps were smoothed with approximately 6 mm FWHM before inference.

TFCE outputs were stored as corrected p-value maps, where `corrp = 1 - pFWE`.

Relevant scripts:

```text
scripts/09_group_inference_rsa/build_tfce_masks_and_qc_thresholds.py 
scripts/09_group_inference_rsa/group_tfce_two_group.py
```

Group-level RSA inference and TFCE procedures were adapted from Thornton et al. (2019) and the accompanying code provided by the authors (https://osf.io/hp5wc/overview).

### 12. Biomotion ROI-Restricted TFCE

In addition to cortex-wide TFCE inference, analyses were repeated within an independently defined biomotion ROI (BM>BMscram_0p05FWEclus_0p001u_k40.nii). The biomotion ROI was intersected with the TFCE-safe analysis mask before group inference was rerun.

Relevant script:

```text
scripts/09_group_inference_rsa/biomotion-roi_tfce_mask.py
```

## Reproducing the Analyses

After installing the Conda environment, the analyses can be reproduced by running the scripts in numerical order within the `scripts/` directory. Each folder corresponds to one major stage of the analysis pipeline.

Most scripts require project-specific file paths to be updated before execution. Example event files, design matrices, and model RDMs are provided in the `data/` directory.


## References

Prince, J. S., et al. (2022). GLMsingle: Improving single-trial fMRI response estimates. *eLife*.

Thornton, M. A., et al. (2019). People represent their own mental states more distinctly than those of others. *Nature Communications*, 10, 2117. https://doi.org/10.1038/s41467-019-10083-6


## Citation

If you use code from this repository, please cite the associated publication.


## License

This repository is released under the MIT License. See `LICENSE` for details.
